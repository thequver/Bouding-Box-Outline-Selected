bl_info = {
    "name": "Bounding Box Outline Selected",
    "author": "Assistant, Quver",
    "version": (1, 0, 1),
    "blender": (3, 6, 0),
    "location": "3D View > Overlay (Object section)",
    "description": "Replace selection outline with bounding boxes",
    "category": "3D View",
}

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from math import inf

_draw_handle = None
_overlay_targets = []
_state_per_area = {}  # area_ptr -> {"enabled": bool, "prev_outline": bool}

# ---------- Helpers ----------

def _tag_redraw_all():
    wm = bpy.context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

def _area_key(area):
    return int(area.as_pointer()) if area else 0

def _is_enabled(area):
    st = _state_per_area.get(_area_key(area))
    return bool(st and st.get("enabled"))

def _set_enabled(area, enabled):
    if not area or area.type != 'VIEW_3D':
        return
    key = _area_key(area)
    space = area.spaces.active
    if not space or space.type != 'VIEW_3D':
        return
    overlay = space.overlay

    st = _state_per_area.get(key, {"enabled": False, "prev_outline": True})
    if enabled:
        st["prev_outline"] = bool(getattr(overlay, "show_outline_selected", True))
        if hasattr(overlay, "show_outline_selected"):
            overlay.show_outline_selected = False
        st["enabled"] = True
    else:
        if hasattr(overlay, "show_outline_selected"):
            overlay.show_outline_selected = bool(st.get("prev_outline", True))
        st["enabled"] = False
    _state_per_area[key] = st
    area.tag_redraw()

def _theme_colors(context):
    try:
        tv = context.preferences.themes[0].view_3d
        active = getattr(tv, "object_active", (1.0, 0.5, 0.0))
        selected = getattr(tv, "object_selected", (1.0, 1.0, 0.0))
    except Exception:
        active, selected = (1.0, 0.5, 0.0), (1.0, 1.0, 0.0)
    if len(active) == 3: active = (*active, 1.0)
    if len(selected) == 3: selected = (*selected, 1.0)
    return active, selected

def _outline_like_width(context):
    # Approximate Blender outline thickness using UI line width
    try:
        w = context.preferences.view.ui_line_width  # 'THIN', 'AUTO', 'THICK'
        if w == 'THIN':  return 2.0
        if w == 'THICK': return 3.6
        return 2.8  # AUTO
    except Exception:
        return 2.8

def _extend(mn, mx, v):
    x, y, z = v
    if x < mn[0]: mn[0] = x
    if y < mn[1]: mn[1] = y
    if z < mn[2]: mn[2] = z
    if x > mx[0]: mx[0] = x
    if y > mx[1]: mx[1] = y
    if z > mx[2]: mx[2] = z

def _collect_aabbs(context, depsgraph):
    # Combined AABB for each selected evaluated object, including instanced geometry
    selected_eval = {
        ob.evaluated_get(depsgraph)
        for ob in context.view_layer.objects
        if ob.select_get() and ob.visible_get()
    }
    if not selected_eval:
        return {}

    result = {}  # eval_owner -> (mn, mx)

    def ensure(owner):
        if owner not in result:
            result[owner] = ([inf, inf, inf], [-inf, -inf, -inf])
        return result[owner]

    for inst in depsgraph.object_instances:
        base = inst.object or getattr(inst, "instance_object", None)
        if base is None:
            continue

        owner = inst.parent if getattr(inst, "is_instance", False) else inst.object
        if owner is None:
            owner = getattr(inst, "instance_parent", None)
        if owner not in selected_eval:
            continue

        bb = getattr(base, "bound_box", None)
        if not bb:
            continue

        mn, mx = ensure(owner)
        mw = inst.matrix_world
        for c in bb:
            w = mw @ Vector(c)
            _extend(mn, mx, (w.x, w.y, w.z))

    # Clean invalid
    return {
        k: (mn, mx)
        for k, (mn, mx) in result.items()
        if not any(v == inf for v in mn) and not any(v == -inf for v in mx)
    }

def _corners_from_aabb(mn, mx):
    x0, y0, z0 = mn
    x1, y1, z1 = mx
    return [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]

# ---------- Draw callback ----------

def _draw_callback_3d():
    ctx = bpy.context
    area = ctx.area
    space = ctx.space_data
    if not area or area.type != 'VIEW_3D' or not space or space.type != 'VIEW_3D':
        return
    if not _is_enabled(area):
        return

    depsgraph = ctx.evaluated_depsgraph_get()
    active_color, selected_color = _theme_colors(ctx)
    shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')

    gpu.state.depth_test_set('LESS_EQUAL')  # occluded
    gpu.state.blend_set('ALPHA')
    try:
        gpu.state.line_width_set(_outline_like_width(ctx))
    except Exception:
        pass

    indices = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    aabbs = _collect_aabbs(ctx, depsgraph)
    active_obj = ctx.view_layer.objects.active
    active_eval = active_obj.evaluated_get(depsgraph) if active_obj else None

    for owner_eval, (mn, mx) in aabbs.items():
        corners = _corners_from_aabb(mn, mx)
        batch = batch_for_shader(shader, 'LINES', {"pos": corners}, indices=indices)
        color = active_color if owner_eval == active_eval else selected_color
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)

    gpu.state.blend_set('NONE')

# ---------- Operator (toggle) ----------

class VIEW3D_OT_bbsel_toggle(bpy.types.Operator):
    bl_idname = "view3d.bbsel_toggle"
    bl_label = "Bounding Box Outline Selected"
    bl_description = "Toggle drawing bounding boxes instead of 'Outline Selected' for this viewport"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return context.area and context.area.type == 'VIEW_3D'

    def execute(self, context):
        area = context.area
        _set_enabled(area, not _is_enabled(area))
        return {'FINISHED'}

# ---------- Overlay UI injection ----------

def _overlay_draw_line(self, context):
    area = context.area
    if not area or area.type != 'VIEW_3D':
        return
    enabled = _is_enabled(area)
    row = self.layout.row(align=True)
    row.use_property_split = False
    row.use_property_decorate = False
    icon = 'CHECKBOX_HLT' if enabled else 'CHECKBOX_DEHLT'
    row.operator("view3d.bbsel_toggle",
                 text="Bounding Box Outline Selected",
                 icon=icon,
                 depress=enabled)

def _append_to_overlay():
    targets = []
    # Prefer the Object section (where 'Outline Selected' is)
    panel = getattr(bpy.types, "VIEW3D_PT_overlay_object", None)
    if panel and hasattr(panel, "append"):
        try:
            panel.append(_overlay_draw_line)
            targets.append(panel)
            return targets
        except Exception:
            pass
    # Fallback only if the Object section isn't available
    panel = getattr(bpy.types, "VIEW3D_PT_overlay", None)
    if panel and hasattr(panel, "append"):
        try:
            panel.append(_overlay_draw_line)
            targets.append(panel)
        except Exception:
            pass
    return targets

# ---------- Register ----------

def register():
    global _draw_handle, _overlay_targets
    bpy.utils.register_class(VIEW3D_OT_bbsel_toggle)
    _overlay_targets = _append_to_overlay()
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback_3d, (), "WINDOW", "POST_VIEW"
        )

def unregister():
    global _draw_handle, _overlay_targets
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None
    for panel in _overlay_targets:
        try:
            panel.remove(_overlay_draw_line)
        except Exception:
            pass
    _overlay_targets.clear()
    try:
        bpy.utils.unregister_class(VIEW3D_OT_bbsel_toggle)
    except Exception:
        pass

if __name__ == "__main__":
    register()