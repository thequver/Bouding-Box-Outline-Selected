bl_info = {
    "name": "Bounding Box Outline Selected",
    "author": "Assistant, Quver",
    "version": (1, 1, 2),
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
import time

# In-session state
_state_per_area = {}    # area_ptr -> {"enabled": bool, "prev_overlay": bool, "prev_shading": bool, "saved": bool}
_state_per_screen = {}  # screen_ptr -> {"prev_overlay": bool, "saved": bool}

# Cache
_cache = {
    "dg_version": -1,
    "selected_key": None,
    "xform_key": None,
    "prefs_sig": None,
    "aabbs": {},         # owner_ptr -> (mn, mx)
}
_dg_version = 0
_last_update_time = 0.0  # throttling gate (Object Mode only)

_draw_handle = None
_overlay_targets = []

# Static box edges
_BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

# -------------------- Preferences --------------------

def _prefs():
    ad = bpy.context.preferences.addons.get(__name__)
    return getattr(ad, "preferences", None)

def _prefs_sig(prefs):
    if not prefs:
        return ('AUTO', 10000, 256)
    return (prefs.accuracy_mode, prefs.auto_threshold, prefs.sample_limit_per_owner)

def _interval():
    try:
        p = _prefs()
        fps = int(p.refresh_fps) if p else 30
        fps = max(1, min(1000, fps))
        return 1.0 / float(fps)
    except Exception:
        return 1.0 / 30.0

class BBSEL_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    accuracy_mode: bpy.props.EnumProperty(
        name="Accuracy",
        items=[
            ('AUTO', "Auto", "Accurate for small counts, sampled for huge counts"),
            ('ACCURATE', "Accurate", "Consider all instances (slow for huge counts)"),
            ('SAMPLED', "Sampled", "Process only a limited number of instances per owner"),
        ],
        default='AUTO',
    )
    auto_threshold: bpy.props.IntProperty(
        name="Auto Threshold", default=10000, min=1000, max=10_000_000,
    )
    sample_limit_per_owner: bpy.props.IntProperty(
        name="Samples per Owner", default=256, min=1, max=100000,
    )
    use_all_views: bpy.props.BoolProperty(
        name="Use in all 3D Views",
        description="Overlay toggle applies to every 3D Viewport and is saved in the .blend",
        default=True,
    )
    refresh_fps: bpy.props.EnumProperty(
        name="Refresh Rate",
        description="How often to refresh selection bounding boxes (Object Mode)",
        items=[('15', "15 FPS", ""), ('30', "30 FPS", ""), ('60', "60 FPS", ""), ('120', "120 FPS", "")],
        default='30',
    )

    def draw(self, _context):
        layout = self.layout
        col = layout.column(align=True)
        col.prop(self, "use_all_views")
        col.prop(self, "refresh_fps", text="Refresh Rate")
        col.separator()
        col.prop(self, "accuracy_mode")
        if self.accuracy_mode == 'AUTO':
            col.prop(self, "auto_threshold")
        if self.accuracy_mode in {'AUTO', 'SAMPLED'}:
            col.prop(self, "sample_limit_per_owner")


# -------------------- Helpers --------------------

def _area_key(area): return int(area.as_pointer()) if area else 0
def _screen_key(screen): return int(screen.as_pointer()) if screen else 0

def _get_state(area):
    return _state_per_area.setdefault(
        _area_key(area),
        {"enabled": False, "prev_overlay": True, "prev_shading": True, "saved": False}
    )

def _get_screen_state(screen):
    return _state_per_screen.setdefault(
        _screen_key(screen),
        {"prev_overlay": True, "saved": False}
    )

def _set_screen_overlay(screen, enable_bbsel):
    # Controls bpy.data.screens["..."].overlay.show_outline_selected
    try:
        ov = getattr(screen, "overlay", None)
        if not ov:
            return
        st = _get_screen_state(screen)
        if enable_bbsel:
            if not st["saved"]:
                st["prev_overlay"] = bool(getattr(ov, "show_outline_selected", True))
                st["saved"] = True
            if hasattr(ov, "show_outline_selected"):
                ov.show_outline_selected = False
        else:
            if hasattr(ov, "show_outline_selected"):
                ov.show_outline_selected = bool(st.get("prev_overlay", True))
            st["saved"] = False
    except Exception:
        pass

def _set_outline_for_area(area, enable_bbsel):
    # Toggle native outline in this viewport (Overlay and Shading) if available
    try:
        st = _get_state(area)
        for sp in area.spaces:
            if sp.type != 'VIEW_3D':
                continue
            ov = getattr(sp, "overlay", None)
            sd = getattr(sp, "shading", None)
            if enable_bbsel:
                if not st["saved"]:
                    st["prev_overlay"] = bool(getattr(ov, "show_outline_selected", True)) if ov else True
                    st["prev_shading"] = bool(getattr(sd, "show_object_outline", True)) if sd else True
                    st["saved"] = True
                if ov and hasattr(ov, "show_outline_selected"):
                    ov.show_outline_selected = False
                if sd and hasattr(sd, "show_object_outline"):
                    sd.show_object_outline = False
            else:
                if ov and hasattr(ov, "show_outline_selected"):
                    ov.show_outline_selected = bool(st.get("prev_overlay", True))
                if sd and hasattr(sd, "show_object_outline"):
                    sd.show_object_outline = bool(st.get("prev_shading", True))
                st["saved"] = False
    except Exception:
        pass

def _enable_area(area, enabled):
    if not area or area.type != 'VIEW_3D':
        return
    _set_outline_for_area(area, enabled)
    _get_state(area)["enabled"] = bool(enabled)
    area.tag_redraw()

def _apply_everywhere(enabled):
    try:
        for screen in bpy.data.screens:
            _set_screen_overlay(screen, enabled)
    except Exception:
        pass
    try:
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    _enable_area(area, enabled)
    except Exception:
        pass

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
    try:
        w = context.preferences.view.ui_line_width  # THIN/AUTO/THICK
        if w == 'THIN':  return 2.0
        if w == 'THICK': return 3.6
        return 2.8
    except Exception:
        return 2.8

def _selected_eval_set(context, depsgraph):
    # Faster than scanning the whole view layer
    return {
        ob.evaluated_get(depsgraph)
        for ob in context.selected_objects
        if ob.visible_get()
    }

def _selected_key(selected_set):
    return tuple(sorted(int(o.as_pointer()) for o in selected_set))

def _xform_key(selected_set):
    # Lightweight transform signature (quantized matrices)
    parts = []
    for ob in sorted(selected_set, key=lambda o: o.as_pointer()):
        mw = ob.matrix_world
        vals = (
            mw[0][0], mw[0][1], mw[0][2],
            mw[1][0], mw[1][1], mw[1][2],
            mw[2][0], mw[2][1], mw[2][2],
            mw[0][3], mw[1][3], mw[2][3],
        )
        q = tuple(int(v * 1e6) for v in vals)
        parts.append((int(ob.as_pointer()),) + q)
    return tuple(parts)

def _is_enabled_anywhere():
    p = _prefs()
    if p and p.use_all_views:
        try:
            return any(getattr(s, "bbsel_enable_all", False) for s in bpy.data.scenes)
        except Exception:
            return bool(getattr(bpy.context.scene, "bbsel_enable_all", False))
    return any(st.get("enabled") for st in _state_per_area.values())

def _tag_redraw_all_3d_views():
    try:
        for win in bpy.context.window_manager.windows:
            if not win.screen:
                continue
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


# -------------------- AABB collection (Object Mode only, throttled) --------------------

def _collect_aabbs_cached(context, depsgraph):
    global _cache, _last_update_time

    # Skip completely in Edit Mode
    if context.mode.startswith("EDIT"):
        return {}

    selected_eval = _selected_eval_set(context, depsgraph)
    if not selected_eval:
        return {}

    p = _prefs()
    psig = _prefs_sig(p)
    sel_key = _selected_key(selected_eval)
    xf_key = _xform_key(selected_eval)

    changed_dg   = (_cache["dg_version"] != _dg_version)
    changed_sel  = (_cache["selected_key"] != sel_key)
    changed_xf   = (_cache["xform_key"] != xf_key)
    changed_pref = (_cache["prefs_sig"] != psig)

    needs_recompute = changed_dg or changed_sel or changed_xf or changed_pref

    now = time.monotonic()
    if not needs_recompute or (now - _last_update_time) < _interval():
        return _cache["aabbs"]

    # Recompute AABBs (with optional sampling / early-exit)
    mode, auto_threshold, sample_limit = psig
    sampling = (mode == 'SAMPLED')
    total_hits = 0
    owners_done = 0
    owners_needed = len(selected_eval)

    owners = {owner: {"mn": [inf, inf, inf], "mx": [-inf, -inf, -inf], "count": 0, "done": False}
              for owner in selected_eval}

    for inst in depsgraph.object_instances:
        base = inst.object or getattr(inst, "instance_object", None)
        if base is None:
            continue

        owner = inst.parent if getattr(inst, "is_instance", False) else inst.object
        if owner is None:
            owner = getattr(inst, "instance_parent", None)

        st = owners.get(owner)
        if st is None:
            continue

        if mode == 'AUTO' and not sampling and total_hits >= auto_threshold:
            sampling = True
        if sampling and st["done"]:
            if owners_done >= owners_needed:
                break
            continue

        bb = getattr(base, "bound_box", None)
        if not bb:
            continue

        mw = inst.matrix_world
        mn, mx = st["mn"], st["mx"]
        for c in bb:
            w = mw @ Vector(c)
            x, y, z = w.x, w.y, w.z
            if x < mn[0]: mn[0] = x
            if y < mn[1]: mn[1] = y
            if z < mn[2]: mn[2] = z
            if x > mx[0]: mx[0] = x
            if y > mx[1]: mx[1] = y
            if z > mx[2]: mx[2] = z

        st["count"] += 1
        total_hits += 1

        if sampling and st["count"] >= sample_limit:
            st["done"] = True
            owners_done += 1
            if owners_done >= owners_needed:
                break

    result = {}
    for owner, st in owners.items():
        mn, mx = st["mn"], st["mx"]
        if any(v == inf for v in mn) or any(v == -inf for v in mx):
            continue
        result[int(owner.as_pointer())] = (mn, mx)

    # Update cache
    _cache["dg_version"] = _dg_version
    _cache["selected_key"] = sel_key
    _cache["xform_key"] = xf_key
    _cache["prefs_sig"] = psig
    _cache["aabbs"] = result
    _last_update_time = now
    return result


# -------------------- Draw callback (batched draws) --------------------

def _build_batches(aabbs, active_eval_ptr):
    if not aabbs:
        return None, None

    sel_pos, sel_idx = [], []
    act_pos, act_idx = [], []

    def add_box(target_pos, target_idx, mn, mx):
        base = len(target_pos)
        target_pos.extend((
            (mn[0], mn[1], mn[2]), (mx[0], mn[1], mn[2]),
            (mx[0], mx[1], mn[2]), (mn[0], mx[1], mn[2]),
            (mn[0], mn[1], mx[2]), (mx[0], mn[1], mx[2]),
            (mx[0], mx[1], mx[2]), (mn[0], mx[1], mx[2]),
        ))
        target_idx.extend(tuple((base + a, base + b) for (a, b) in _BOX_EDGES))

    for owner_ptr, (mn, mx) in aabbs.items():
        if active_eval_ptr and owner_ptr == active_eval_ptr:
            add_box(act_pos, act_idx, mn, mx)
        else:
            add_box(sel_pos, sel_idx, mn, mx)

    shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
    batch_sel = batch_for_shader(shader, 'LINES', {"pos": sel_pos}, indices=sel_idx) if sel_idx else None
    batch_act = batch_for_shader(shader, 'LINES', {"pos": act_pos}, indices=act_idx) if act_idx else None
    return batch_sel, batch_act

def _draw_callback_3d():
    ctx = bpy.context
    area = ctx.area
    sp = ctx.space_data
    if not area or area.type != 'VIEW_3D' or not sp or sp.type != 'VIEW_3D':
        return

    # Do not display in Edit Mode at all
    if ctx.mode.startswith("EDIT"):
        return

    p = _prefs()
    if p and p.use_all_views:
        if not bool(getattr(ctx.scene, "bbsel_enable_all", False)):
            return
    else:
        if not _get_state(area).get("enabled"):
            return

    depsgraph = ctx.evaluated_depsgraph_get()
    active_color, selected_color = _theme_colors(ctx)
    shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')

    aabbs = _collect_aabbs_cached(ctx, depsgraph)
    if not aabbs:
        return

    active_obj = ctx.view_layer.objects.active
    active_eval_ptr = int(active_obj.evaluated_get(depsgraph).as_pointer()) if active_obj else None

    batch_sel, batch_act = _build_batches(aabbs, active_eval_ptr)
    if not batch_sel and not batch_act:
        return

    gpu.state.depth_test_set('LESS_EQUAL')  # occluded
    gpu.state.blend_set('ALPHA')
    try:
        gpu.state.line_width_set(_outline_like_width(ctx))
    except Exception:
        pass

    if batch_sel:
        shader.bind()
        shader.uniform_float("color", selected_color)
        batch_sel.draw(shader)
    if batch_act:
        shader.bind()
        shader.uniform_float("color", active_color)
        batch_act.draw(shader)

    gpu.state.blend_set('NONE')


# -------------------- Operator (overlay button) --------------------

class VIEW3D_OT_bbsel_toggle(bpy.types.Operator):
    bl_idname = "view3d.bbsel_toggle"
    bl_label = "Bounding Box Outline Selected"
    bl_description = "Toggle drawing bounding boxes instead of 'Outline Selected' (per viewport or all viewports)"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return context.area and context.area.type == 'VIEW_3D'

    def execute(self, context):
        p = _prefs()
        use_all = bool(p.use_all_views) if p else False

        if use_all:
            new_state = not bool(getattr(context.scene, "bbsel_enable_all", False))
            try:
                for sce in bpy.data.scenes:
                    sce.bbsel_enable_all = new_state
            except Exception:
                context.scene.bbsel_enable_all = new_state
            _apply_everywhere(new_state)
        else:
            area = context.area
            st = _get_state(area)
            _enable_area(area, not bool(st.get("enabled")))

        _tag_redraw_all_3d_views()
        return {'FINISHED'}


# -------------------- Overlay UI injection --------------------

def _overlay_draw_line(self, context):
    area = context.area
    if not area or area.type != 'VIEW_3D':
        return

    p = _prefs()
    enabled = bool(getattr(context.scene, "bbsel_enable_all", False)) if (p and p.use_all_views) \
              else bool(_get_state(area).get("enabled"))

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
    panel = getattr(bpy.types, "VIEW3D_PT_overlay_object", None)
    if panel and hasattr(panel, "append"):
        try:
            panel.append(_overlay_draw_line)
            targets.append(panel)
            return targets
        except Exception:
            pass
    panel = getattr(bpy.types, "VIEW3D_PT_overlay", None)
    if panel and hasattr(panel, "append"):
        try:
            panel.append(_overlay_draw_line)
            targets.append(panel)
        except Exception:
            pass
    return targets


# -------------------- Cleanup legacy props --------------------

def _cleanup_legacy_props():
    # Remove stray ID properties left by older versions on SpaceView3D
    try:
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                for sp in area.spaces:
                    if sp.type != 'VIEW_3D':
                        continue
                    for k in ("bbsel_enable", "bbsel_prev_outline", "bbsel_prev_outline_state"):
                        if k in sp.keys():
                            try:
                                del sp[k]
                            except Exception:
                                pass
    except Exception:
        pass


# -------------------- Handlers --------------------

def _depsgraph_update_post(*_args):
    global _dg_version
    _dg_version += 1

def _load_post(_dummy):
    # Clean legacy props and re-apply global state if saved ON
    _cleanup_legacy_props()
    try:
        enabled = bool(getattr(bpy.context.scene, "bbsel_enable_all", False))
        if enabled:
            _apply_everywhere(True)
    except Exception:
        pass
    # Reset throttling so first draw happens immediately
    global _last_update_time
    _last_update_time = 0.0


# -------------------- Register --------------------

classes = (
    BBSEL_AddonPreferences,
    VIEW3D_OT_bbsel_toggle,
)

def register():
    global _draw_handle, _overlay_targets
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.bbsel_enable_all = bpy.props.BoolProperty(
        name="Bounding Box Outline Selected (All Views)",
        description="If enabled, draw bounding box selection in all 3D Views (saved in file)",
        default=False,  # feature starts OFF by default
    )

    _overlay_targets = _append_to_overlay()

    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback_3d, (), "WINDOW", "POST_VIEW"
        )

    if _depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_depsgraph_update_post)
    if _load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_load_post)

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
        bpy.app.handlers.depsgraph_update_post.remove(_depsgraph_update_post)
    except Exception:
        pass
    try:
        bpy.app.handlers.load_post.remove(_load_post)
    except Exception:
        pass

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    if hasattr(bpy.types.Scene, "bbsel_enable_all"):
        del bpy.types.Scene.bbsel_enable_all


if __name__ == "__main__":
    register()
