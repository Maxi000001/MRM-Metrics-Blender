# -*- coding: utf-8 -*-
# ============================================================
#  MRM Metrics Pro  –  Professional Measurement Addon
#  Author  : Malik + Claude (Anthropic)
#  Version : 5.0.0
#  Blender : 4.2+ / 5.0
#  License : MIT
# ============================================================
"""
MRM Metrics Pro — Professional measurement overlay for Blender.

Features
--------
• Distance / Angle / Diameter / Radius measurements
• Interactive vertex picker (click to select in viewport)
• Per-measure color, visibility, prefix/suffix labels
• Copy measurement value to clipboard
• Export all measurements to JSON or CSV
• Addon Preferences (persistent UI settings)
• Text background box for readability
• Adaptive arrow sizing by distance
• Object-mode & Edit-mode support
• Full undo/redo integration
• Clean modular code architecture
"""

bl_info = {
    "name"       : "MRM Metrics Pro",
    "author"     : "Malik + Claude",
    "version"    : (5, 0, 0),
    "blender"    : (4, 2, 0),
    "location"   : "View3D › Sidebar › MRM Metrics",
    "description": "Professional distance / angle / diameter / radius measurement overlay",
    "category"   : "3D View",
    "doc_url"    : "",
    "tracker_url": "",
}

# ──────────────────────────────────────────────────────────────────────────────
#  Imports
# ──────────────────────────────────────────────────────────────────────────────
import bpy
import bmesh
import math
import json
import traceback
from pathlib import Path
from mathutils import Vector, Matrix

import gpu
import blf
from gpu_extras.batch import batch_for_shader
from bpy.props import (
    FloatProperty, FloatVectorProperty, EnumProperty, BoolProperty,
    IntProperty, CollectionProperty, PointerProperty, StringProperty,
)
from bpy.types import (
    Operator, Panel, PropertyGroup, UIList, AddonPreferences,
)
from bpy_extras import view3d_utils

# ──────────────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────────────
ADDON_ID      = __name__
_FONT_ID      = 0
_DRAW_HANDLE  = None
_SHADER       = None

COL_DISTANCE = (0.18, 0.85, 0.34, 1.0)   # vivid green
COL_ANGLE    = (0.25, 0.55, 1.00, 1.0)   # blue
COL_DIAMETER = (1.00, 0.82, 0.10, 1.0)   # amber
COL_RADIUS   = (1.00, 0.35, 0.35, 1.0)   # coral red

METRIC_COLLECTION  = "MRM Metrics"
SUB_COLLECTIONS    = {"DISTANCE": "MRM Distance",
                      "ANGLE"   : "MRM Angle",
                      "DIAMETER": "MRM Diameter",
                      "RADIUS"  : "MRM Radius"}

UNIT_ITEMS = [
    ('AUTO', "Auto",        "Automatic m / cm / mm"),
    ('M',    "Meters",      "Meters"),
    ('CM',   "Centimeters", "Centimeters"),
    ('MM',   "Millimeters", "Millimeters"),
    ('IN',   "Inches",      "Inches"),
    ('BU',   "Blender U",   "Raw Blender units"),
]

ANGLE_UNIT_ITEMS = [
    ('DEG', "Degrees",  "Degrees (°)"),
    ('RAD', "Radians",  "Radians (rad)"),
]

KIND_ITEMS = [
    ('DISTANCE', "Distance", ""),
    ('ANGLE',    "Angle",    ""),
    ('DIAMETER', "Diameter", ""),
    ('RADIUS',   "Radius",   ""),
]


# ──────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def tag_redraw():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def get_preferences():
    return bpy.context.preferences.addons[ADDON_ID].preferences


def safe_project(region, rv3d, co3d):
    try:
        p = view3d_utils.location_3d_to_region_2d(region, rv3d, co3d)
        return (p.x, p.y) if p else None
    except Exception:
        return None


def get_shader():
    global _SHADER
    if _SHADER is None:
        _SHADER = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    return _SHADER


def format_distance(val: float, mode: str) -> str:
    """Convert a Blender-unit value to a display string."""
    if   mode == 'BU': return f"{val:.4f} BU"
    elif mode == 'M':  return f"{val:.4f} m"
    elif mode == 'CM': return f"{val*100:.2f} cm"
    elif mode == 'MM': return f"{val*1000:.1f} mm"
    elif mode == 'IN': return f"{val*39.3701:.3f} in"
    else:  # AUTO
        a = abs(val)
        if   a >= 1.0:    return f"{val:.4f} m"
        elif a >= 0.01:   return f"{val*100:.2f} cm"
        else:             return f"{val*1000:.1f} mm"


def format_angle(val_rad: float, mode: str) -> str:
    if mode == 'RAD':
        return f"{val_rad:.4f} rad"
    return f"{math.degrees(val_rad):.2f}°"


# ──────────────────────────────────────────────────────────────────────────────
#  GPU drawing primitives
# ──────────────────────────────────────────────────────────────────────────────

def _set_shader_uniforms(shader, color, thickness, region):
    shader.bind()
    shader.uniform_float("color",        color)
    shader.uniform_float("lineWidth",    float(max(1.0, thickness)))
    shader.uniform_float("viewportSize", (region.width, region.height))


def draw_line(p1, p2, color, thickness, region):
    if p1 is None or p2 is None:
        return
    try:
        sh = get_shader()
        _set_shader_uniforms(sh, color, thickness, region)
        batch = batch_for_shader(sh, 'LINES',
                                 {"pos": [(p1[0], p1[1], 0.0), (p2[0], p2[1], 0.0)]})
        batch.draw(sh)
    except Exception:
        pass


def draw_polyline(pts, color, thickness, region, closed=False):
    if len(pts) < 2:
        return
    try:
        mode = 'LINE_LOOP' if closed else 'LINE_STRIP'
        sh = get_shader()
        _set_shader_uniforms(sh, color, thickness, region)
        pts3 = [(p[0], p[1], 0.0) for p in pts]
        batch = batch_for_shader(sh, mode, {"pos": pts3})
        batch.draw(sh)
    except Exception:
        pass


def draw_arrow_head(tip, direction_to_tip, color, size_px, thickness, region):
    """Draw arrowhead at 'tip' pointing in direction_to_tip."""
    dx, dy = direction_to_tip
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux
    wing = size_px * 0.45
    a1 = (tip[0] - ux * size_px + px * wing, tip[1] - uy * size_px + py * wing)
    a2 = (tip[0] - ux * size_px - px * wing, tip[1] - uy * size_px - py * wing)
    draw_line(tip, a1, color, thickness, region)
    draw_line(tip, a2, color, thickness, region)


def draw_arrow_2d(p1, p2, color, arrow_size, thickness, region):
    """Full line with arrowheads at both ends."""
    draw_line(p1, p2, color, thickness, region)
    if p1 is None or p2 is None:
        return
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    draw_arrow_head(p1, (-dx, -dy), color, arrow_size, thickness, region)
    draw_arrow_head(p2, ( dx,  dy), color, arrow_size, thickness, region)


def draw_text(pos, text, color, font_size, bg=False):
    """Draw 2-D text label, optionally with dark background rectangle."""
    if pos is None:
        return
    try:
        ui = bpy.context.preferences.system.ui_scale
        size = max(8, int(font_size * ui))
        blf.size(_FONT_ID, size)
        blf.color(_FONT_ID, *color)

        if bg:
            w, h = blf.dimensions(_FONT_ID, text)
            pad = 3 * ui
            verts = [
                (pos[0] - pad,       pos[1] - pad),
                (pos[0] + w + pad,   pos[1] - pad),
                (pos[0] + w + pad,   pos[1] + h + pad),
                (pos[0] - pad,       pos[1] + h + pad),
            ]
            try:
                sh = gpu.shader.from_builtin('UNIFORM_COLOR')
                sh.bind()
                sh.uniform_float("color", (0.0, 0.0, 0.0, 0.55))
                batch = batch_for_shader(sh, 'TRI_FAN',
                                         {"pos": [(v[0], v[1]) for v in verts]})
                gpu.state.blend_set('ALPHA')
                batch.draw(sh)
            except Exception:
                pass

        blf.position(_FONT_ID, pos[0], pos[1], 0)
        blf.draw(_FONT_ID, text)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
#  Collection helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_collections(context):
    scene_col = context.scene.collection
    main = bpy.data.collections.get(METRIC_COLLECTION)
    if main is None:
        main = bpy.data.collections.new(METRIC_COLLECTION)
    if main.name not in scene_col.children:
        try: scene_col.children.link(main)
        except Exception: pass
    for sub_name in SUB_COLLECTIONS.values():
        sub = bpy.data.collections.get(sub_name)
        if sub is None:
            sub = bpy.data.collections.new(sub_name)
        if sub.name not in [c.name for c in main.children]:
            try: main.children.link(sub)
            except Exception: pass
    return main


def link_to_sub(obj, kind, context):
    if obj is None: return
    ensure_collections(context)
    sub = bpy.data.collections.get(SUB_COLLECTIONS.get(kind, ""))
    if sub and obj.name not in [o.name for o in sub.objects]:
        try: sub.objects.link(obj)
        except Exception: pass


def unlink_from_sub_if_empty(obj, kind, context):
    if obj is None: return
    sub = bpy.data.collections.get(SUB_COLLECTIONS.get(kind, ""))
    if sub and obj.name in [o.name for o in sub.objects]:
        coll_map = {
            'DISTANCE': "mrm_distance_measures",
            'ANGLE'   : "mrm_angle_measures",
            'DIAMETER': "mrm_diameter_measures",
            'RADIUS'  : "mrm_radius_measures",
        }
        attr = coll_map.get(kind)
        if attr and len(getattr(obj, attr, [])) == 0:
            try: sub.objects.unlink(obj)
            except Exception: pass


# ──────────────────────────────────────────────────────────────────────────────
#  Mesh helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_selected_vertex_indices(obj):
    """Return selected vertex indices from edit-mesh in history order."""
    if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
        return []
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        hist = [e for e in bm.select_history if isinstance(e, bmesh.types.BMVert) and e.select]
        if hist:
            return [v.index for v in hist]
        return [v.index for v in bm.verts if v.select]
    except Exception:
        return []


def get_world_verts(obj):
    """Return {index: (world_pos, world_normal)} for all vertices."""
    result = {}
    if obj is None or obj.type != 'MESH':
        return result
    try:
        mw  = obj.matrix_world
        mwi = mw.inverted_safe().transposed()
        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            for v in bm.verts:
                result[v.index] = (mw @ v.co, (mwi @ v.normal).normalized())
        else:
            for v in obj.data.vertices:
                result[v.index] = (mw @ v.co, (mwi @ v.normal).normalized())
    except Exception:
        pass
    return result


def find_two_farthest(obj):
    """Return (i1, i2) of the two farthest selected vertices."""
    if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
        return None, None
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        sel = [v for v in bm.verts if v.select]
        if len(sel) < 2: return None, None
        max_d = -1; v1 = v2 = None
        for i, a in enumerate(sel):
            for b in sel[i + 1:]:
                d = (a.co - b.co).length
                if d > max_d:
                    max_d = d; v1 = a; v2 = b
        return (v1.index, v2.index) if v1 else (None, None)
    except Exception:
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
#  Property Groups
# ──────────────────────────────────────────────────────────────────────────────

class MRM_TypeVisualProps(PropertyGroup):
    """Shared visual settings for one measurement type."""
    color: FloatVectorProperty(
        name="Color", subtype='COLOR', size=4,
        default=(1.0, 1.0, 1.0, 1.0), min=0.0, max=1.0,
    )
    thickness : FloatProperty(name="Thickness",     default=2.0, min=1.0, max=20.0)
    normal_off: FloatProperty(name="Normal Offset",  default=0.05)
    text_off  : FloatProperty(name="Text Offset",    default=0.08)
    font_size : IntProperty(  name="Font Size",      default=14, min=8, max=72)
    text_bg   : BoolProperty( name="Text Background",default=True,
                              description="Draw dark box behind measurement text")
    arrow_size: FloatProperty(name="Arrow Size (px)",default=12.0, min=4.0, max=40.0)


class MRM_Measure(PropertyGroup):
    """One stored measurement on a mesh object."""
    name   : StringProperty(default="Measure")
    kind   : EnumProperty(items=KIND_ITEMS, default='DISTANCE')
    v1     : IntProperty(default=-1)
    v2     : IntProperty(default=-1)
    v3     : IntProperty(default=-1)   # angle centre vertex
    show   : BoolProperty(default=True)

    # Per-measure visual overrides
    use_custom_color: BoolProperty(name="Custom Color", default=False,
                                   description="Override type color for this measure")
    custom_color: FloatVectorProperty(name="Color", subtype='COLOR', size=4,
                                      default=(1.0, 1.0, 1.0, 1.0), min=0.0, max=1.0)

    # Label customisation
    prefix : StringProperty(name="Prefix",  default="",  description="Text before value")
    suffix : StringProperty(name="Suffix",  default="",  description="Text after value")
    note   : StringProperty(name="Note",    default="",  description="Extra annotation line")

    # Diameter/Radius extras
    show_circle: BoolProperty(name="Show Circle", default=False,
                              description="Draw circumference circle (diameter / radius)")


class MRM_Settings(PropertyGroup):
    """Scene-level settings."""
    unit_mode : EnumProperty(name="Distance Unit", items=UNIT_ITEMS, default='AUTO')
    angle_unit: EnumProperty(name="Angle Unit",    items=ANGLE_UNIT_ITEMS, default='DEG')

    distance : PointerProperty(type=MRM_TypeVisualProps)
    angle    : PointerProperty(type=MRM_TypeVisualProps)
    diameter : PointerProperty(type=MRM_TypeVisualProps)
    radius   : PointerProperty(type=MRM_TypeVisualProps)

    expand_distance: BoolProperty(default=True)
    expand_angle   : BoolProperty(default=True)
    expand_diameter: BoolProperty(default=True)
    expand_radius  : BoolProperty(default=False)

    show_all: BoolProperty(name="Show All", default=True,
                           description="Global toggle for ALL measurements")


# ──────────────────────────────────────────────────────────────────────────────
#  Addon Preferences
# ──────────────────────────────────────────────────────────────────────────────

class MRM_AddonPreferences(AddonPreferences):
    bl_idname = ADDON_ID

    default_unit: EnumProperty(name="Default Unit", items=UNIT_ITEMS, default='AUTO')
    default_font: IntProperty(name="Default Font Size", default=14, min=8, max=72)
    auto_start  : BoolProperty(name="Auto-enable on Startup", default=False,
                               description="Automatically start measurement overlay when Blender starts")
    export_path : StringProperty(name="Export Directory", default="//",
                                 subtype='DIR_PATH',
                                 description="Default directory for JSON/CSV export")

    def draw(self, context):
        layout = self.layout
        layout.label(text="MRM Metrics Pro — Global Defaults", icon='SETTINGS')
        col = layout.column(align=True)
        col.prop(self, "default_unit")
        col.prop(self, "default_font")
        col.prop(self, "auto_start")
        col.prop(self, "export_path")


# ──────────────────────────────────────────────────────────────────────────────
#  Operators — Core
# ──────────────────────────────────────────────────────────────────────────────

class MRM_OT_ToggleDraw(Operator):
    bl_idname      = "mrm.toggle_draw"
    bl_label       = "Toggle MRM Overlay"
    bl_description = "Start / stop measurement overlay rendering"

    def execute(self, context):
        global _DRAW_HANDLE
        wm = context.window_manager
        if not getattr(wm, "mrm_running", False):
            ensure_collections(context)
            _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
                _draw_callback, (), 'WINDOW', 'POST_PIXEL')
            wm.mrm_running = True
        else:
            if _DRAW_HANDLE is not None:
                try: bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, 'WINDOW')
                except Exception: pass
            _DRAW_HANDLE = None
            wm.mrm_running = False
        tag_redraw()
        return {'FINISHED'}


class MRM_OT_CreateDistance(Operator):
    bl_idname      = "mrm.create_distance"
    bl_label       = "Create Distance"
    bl_description = "Measure distance between exactly 2 selected vertices (Edit Mode)"
    bl_options     = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj  = context.active_object
        inds = get_selected_vertex_indices(obj)
        if len(inds) != 2:
            self.report({'WARNING'}, "Select exactly 2 vertices")
            return {'CANCELLED'}
        ss = context.scene.mrm_settings
        m  = obj.mrm_distance_measures.add()
        m.kind = 'DISTANCE'; m.v1 = inds[0]; m.v2 = inds[1]
        m.name = f"Dist {inds[0]}–{inds[1]}"
        link_to_sub(obj, 'DISTANCE', context)
        tag_redraw()
        return {'FINISHED'}


class MRM_OT_CreateAngle(Operator):
    bl_idname      = "mrm.create_angle"
    bl_label       = "Create Angle"
    bl_description = "Measure angle at middle vertex (select 3 vertices in order: arm–centre–arm)"
    bl_options     = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj  = context.active_object
        inds = get_selected_vertex_indices(obj)
        if len(inds) != 3:
            self.report({'WARNING'}, "Select exactly 3 vertices (arm – centre – arm)")
            return {'CANCELLED'}
        ss = context.scene.mrm_settings
        m  = obj.mrm_angle_measures.add()
        m.kind = 'ANGLE'; m.v1 = inds[0]; m.v2 = inds[1]; m.v3 = inds[2]
        m.name = f"Angle {inds[0]}–{inds[1]}–{inds[2]}"
        link_to_sub(obj, 'ANGLE', context)
        tag_redraw()
        return {'FINISHED'}


class MRM_OT_CreateDiameter(Operator):
    bl_idname      = "mrm.create_diameter"
    bl_label       = "Create Diameter"
    bl_description = "Measure diameter between two farthest selected vertices"
    bl_options     = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.mode != 'EDIT':
            self.report({'WARNING'}, "Enter Edit Mode and select vertices")
            return {'CANCELLED'}
        i1, i2 = find_two_farthest(obj)
        if i1 is None:
            self.report({'WARNING'}, "Select at least 2 vertices")
            return {'CANCELLED'}
        m  = obj.mrm_diameter_measures.add()
        m.kind = 'DIAMETER'; m.v1 = i1; m.v2 = i2
        m.name = f"Ø {i1}–{i2}"
        link_to_sub(obj, 'DIAMETER', context)
        tag_redraw()
        return {'FINISHED'}


class MRM_OT_CreateRadius(Operator):
    bl_idname      = "mrm.create_radius"
    bl_label       = "Create Radius"
    bl_description = "Measure radius from first vertex (centre) to second vertex"
    bl_options     = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj  = context.active_object
        inds = get_selected_vertex_indices(obj)
        if len(inds) != 2:
            self.report({'WARNING'}, "Select exactly 2 vertices (centre → edge)")
            return {'CANCELLED'}
        m  = obj.mrm_radius_measures.add()
        m.kind = 'RADIUS'; m.v1 = inds[0]; m.v2 = inds[1]
        m.name = f"R {inds[0]}–{inds[1]}"
        link_to_sub(obj, 'RADIUS', context)
        tag_redraw()
        return {'FINISHED'}


# ──────────────────────────────────────────────────────────────────────────────
#  Operators — Management
# ──────────────────────────────────────────────────────────────────────────────

class MRM_OT_DeleteMeasure(Operator):
    bl_idname  = "mrm.delete_measure"
    bl_label   = "Delete Measure"
    bl_options = {'REGISTER', 'UNDO'}
    index: IntProperty(); kind: StringProperty()

    def execute(self, context):
        obj = context.active_object
        if obj is None: return {'CANCELLED'}
        coll = _get_coll(obj, self.kind)
        if coll and 0 <= self.index < len(coll):
            coll.remove(self.index)
            unlink_from_sub_if_empty(obj, self.kind, context)
        tag_redraw(); return {'FINISHED'}


class MRM_OT_ClearAll(Operator):
    bl_idname  = "mrm.clear_all"
    bl_label   = "Clear ALL Measures"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        for o in context.scene.objects:
            try:
                for attr in ("mrm_distance_measures", "mrm_angle_measures",
                             "mrm_diameter_measures", "mrm_radius_measures"):
                    getattr(o, attr, None) and getattr(o, attr).clear()
                for k in SUB_COLLECTIONS:
                    unlink_from_sub_if_empty(o, k, context)
            except Exception: pass
        tag_redraw()
        self.report({'INFO'}, "All measurements cleared")
        return {'FINISHED'}


class MRM_OT_ClearKind(Operator):
    bl_idname  = "mrm.clear_kind"
    bl_label   = "Clear Type"
    bl_options = {'REGISTER', 'UNDO'}
    kind: StringProperty()

    def execute(self, context):
        obj = context.active_object
        if obj is None: return {'CANCELLED'}
        coll = _get_coll(obj, self.kind)
        if coll: coll.clear()
        unlink_from_sub_if_empty(obj, self.kind, context)
        tag_redraw(); return {'FINISHED'}


class MRM_OT_ToggleVisibility(Operator):
    bl_idname = "mrm.toggle_visibility"
    bl_label  = "Toggle Visibility"
    index: IntProperty(); kind: StringProperty()

    def execute(self, context):
        obj = context.active_object
        if obj is None: return {'CANCELLED'}
        coll = _get_coll(obj, self.kind)
        if coll and 0 <= self.index < len(coll):
            coll[self.index].show = not coll[self.index].show
        tag_redraw(); return {'FINISHED'}


class MRM_OT_ToggleAllVisibility(Operator):
    bl_idname = "mrm.toggle_all_visibility"
    bl_label  = "Toggle All Visibility"
    kind: StringProperty()

    def execute(self, context):
        obj = context.active_object
        if obj is None: return {'CANCELLED'}
        coll = _get_coll(obj, self.kind)
        if coll:
            state = not all(m.show for m in coll)
            for m in coll: m.show = state
        tag_redraw(); return {'FINISHED'}


class MRM_OT_ToggleCircle(Operator):
    bl_idname = "mrm.toggle_circle"
    bl_label  = "Toggle Circle"
    index: IntProperty(); kind: StringProperty()

    def execute(self, context):
        obj = context.active_object
        if obj is None: return {'CANCELLED'}
        coll = _get_coll(obj, self.kind)
        if coll and 0 <= self.index < len(coll):
            coll[self.index].show_circle = not coll[self.index].show_circle
        tag_redraw(); return {'FINISHED'}


class MRM_OT_CopyValue(Operator):
    """Copy the measured value to the clipboard."""
    bl_idname      = "mrm.copy_value"
    bl_label       = "Copy Value"
    bl_description = "Copy measurement value to clipboard"
    index: IntProperty(); kind: StringProperty()

    def execute(self, context):
        obj = context.active_object
        if obj is None: return {'CANCELLED'}
        coll = _get_coll(obj, self.kind)
        if not coll or not (0 <= self.index < len(coll)):
            return {'CANCELLED'}
        m = coll[self.index]
        ss = context.scene.mrm_settings
        vw = get_world_verts(obj)
        text = _compute_value_text(m, vw, ss)
        if text:
            context.window_manager.clipboard = text
            self.report({'INFO'}, f"Copied: {text}")
        return {'FINISHED'}


class MRM_OT_ExportJSON(Operator):
    """Export all measurements on the active object to a JSON file."""
    bl_idname      = "mrm.export_json"
    bl_label       = "Export JSON"
    bl_description = "Save all measurements to a JSON file"
    filepath: StringProperty(subtype='FILE_PATH')

    def invoke(self, context, event):
        prefs = get_preferences()
        self.filepath = str(Path(bpy.path.abspath(prefs.export_path)) / "mrm_export.json")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'WARNING'}, "No active object"); return {'CANCELLED'}
        data = _collect_export_data(obj, context)
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.report({'INFO'}, f"Exported to {self.filepath}")
        except Exception as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}


class MRM_OT_ExportCSV(Operator):
    """Export measurements to CSV."""
    bl_idname      = "mrm.export_csv"
    bl_label       = "Export CSV"
    bl_description = "Save all measurements to a CSV file"
    filepath: StringProperty(subtype='FILE_PATH')

    def invoke(self, context, event):
        prefs = get_preferences()
        self.filepath = str(Path(bpy.path.abspath(prefs.export_path)) / "mrm_export.csv")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'WARNING'}, "No active object"); return {'CANCELLED'}
        data = _collect_export_data(obj, context)
        try:
            import csv, io
            buf = io.StringIO()
            wr  = csv.writer(buf)
            wr.writerow(["name","kind","value","unit","v1","v2","v3"])
            for row in data.get("measures", []):
                wr.writerow([row.get(k, "") for k in ("name","kind","value","unit","v1","v2","v3")])
            with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
                f.write(buf.getvalue())
            self.report({'INFO'}, f"Exported to {self.filepath}")
        except Exception as e:
            self.report({'ERROR'}, str(e))
        return {'FINISHED'}


# ──────────────────────────────────────────────────────────────────────────────
#  Operator helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _get_coll(obj, kind):
    return {
        'DISTANCE': getattr(obj, "mrm_distance_measures", None),
        'ANGLE'   : getattr(obj, "mrm_angle_measures",    None),
        'DIAMETER': getattr(obj, "mrm_diameter_measures", None),
        'RADIUS'  : getattr(obj, "mrm_radius_measures",   None),
    }.get(kind)


def _compute_value_text(m, verts_world, ss):
    """Return a plain value string for a measure (no color / drawing)."""
    try:
        if m.kind == 'DISTANCE':
            v1, v2 = verts_world.get(m.v1), verts_world.get(m.v2)
            if v1 and v2:
                return format_distance((v2[0] - v1[0]).length, ss.unit_mode)
        elif m.kind == 'ANGLE':
            v1, v2, v3 = verts_world.get(m.v1), verts_world.get(m.v2), verts_world.get(m.v3)
            if v1 and v2 and v3:
                a = (v1[0] - v2[0]).normalized().angle((v3[0] - v2[0]).normalized())
                return format_angle(a, ss.angle_unit)
        elif m.kind in ('DIAMETER', 'RADIUS'):
            v1, v2 = verts_world.get(m.v1), verts_world.get(m.v2)
            if v1 and v2:
                d = (v2[0] - v1[0]).length
                val = d if m.kind == 'DIAMETER' else d / 2
                prefix = "Ø " if m.kind == 'DIAMETER' else "R "
                return prefix + format_distance(val, ss.unit_mode)
    except Exception:
        pass
    return ""


def _collect_export_data(obj, context):
    ss  = context.scene.mrm_settings
    vw  = get_world_verts(obj)
    out = {"object": obj.name, "unit": ss.unit_mode, "measures": []}
    for kind, attr in [("DISTANCE","mrm_distance_measures"),
                       ("ANGLE","mrm_angle_measures"),
                       ("DIAMETER","mrm_diameter_measures"),
                       ("RADIUS","mrm_radius_measures")]:
        for m in getattr(obj, attr, []):
            val = _compute_value_text(m, vw, ss)
            out["measures"].append({
                "name": m.name, "kind": kind,
                "value": val, "unit": ss.unit_mode,
                "v1": m.v1, "v2": m.v2, "v3": m.v3,
                "prefix": m.prefix, "suffix": m.suffix, "note": m.note,
            })
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Draw callback
# ──────────────────────────────────────────────────────────────────────────────

def _draw_circle_3d(center, radius, normal, off_n, color, thickness, region, rv3d, segments=48):
    """Project a 3-D circle into 2-D and draw it."""
    if abs(normal.z) > 0.9:
        tang = Vector((1, 0, 0)).cross(normal)
    else:
        tang = Vector((0, 0, 1)).cross(normal)
    tang = tang.normalized()
    bita = normal.cross(tang).normalized()
    pts = []
    for i in range(segments + 1):
        a = (i / segments) * math.tau
        p3 = center + (tang * math.cos(a) + bita * math.sin(a)) * radius + normal * off_n
        p2 = safe_project(region, rv3d, p3)
        if p2: pts.append(p2)
    if len(pts) >= 3:
        draw_polyline(pts, color, thickness, region, closed=True)


def _draw_callback():
    try:
        ctx = bpy.context
        if not ctx or not ctx.area or ctx.area.type != 'VIEW_3D':
            return
        ss = ctx.scene.mrm_settings
        if not getattr(ctx.window_manager, "mrm_running", False):
            return
        if not ss.show_all:
            return

        region = ctx.region
        rv3d   = ctx.space_data.region_3d
        prefs  = get_preferences()
        gpu.state.blend_set('ALPHA')

        for obj in ctx.visible_objects:
            if obj.type != 'MESH':
                continue
            vw = get_world_verts(obj)
            if not vw:
                continue

            # ── DISTANCE ──────────────────────────────────────────────────────
            vs = ss.distance
            for m in obj.mrm_distance_measures:
                if not m.show: continue
                w1, w2 = vw.get(m.v1), vw.get(m.v2)
                if not w1 or not w2: continue
                col = tuple(m.custom_color) if m.use_custom_color else tuple(vs.color)
                p1w, n1 = w1; p2w, n2 = w2
                off = vs.normal_off
                o1 = p1w + n1 * off; o2 = p2w + n2 * off
                q1 = safe_project(region, rv3d, p1w)
                q2 = safe_project(region, rv3d, p2w)
                r1 = safe_project(region, rv3d, o1)
                r2 = safe_project(region, rv3d, o2)
                draw_line(q1, r1, col, vs.thickness, region)
                draw_line(q2, r2, col, vs.thickness, region)
                draw_arrow_2d(r1, r2, col, vs.arrow_size, vs.thickness, region)
                dist = (p2w - p1w).length
                lbl  = m.prefix + format_distance(dist, ss.unit_mode) + m.suffix
                if m.note: lbl += f"\n{m.note}"
                avg_n = (n1 + n2).normalized()
                t3 = (o1 + o2) * 0.5 + avg_n * vs.text_off
                draw_text(safe_project(region, rv3d, t3), lbl, col, vs.font_size, vs.text_bg)

            # ── ANGLE ─────────────────────────────────────────────────────────
            va = ss.angle
            for m in obj.mrm_angle_measures:
                if not m.show: continue
                w1, w2, w3 = vw.get(m.v1), vw.get(m.v2), vw.get(m.v3)
                if not w1 or not w2 or not w3: continue
                col = tuple(m.custom_color) if m.use_custom_color else tuple(va.color)
                p1w, n1 = w1; p2w, n2 = w2; p3w, n3 = w3
                off = va.normal_off
                o1 = p1w + n1*off; o2 = p2w + n2*off; o3 = p3w + n3*off
                for src, dst in [(p1w,o1),(p2w,o2),(p3w,o3)]:
                    draw_line(safe_project(region,rv3d,src),
                              safe_project(region,rv3d,dst), col, va.thickness, region)
                q1=safe_project(region,rv3d,o1)
                q2=safe_project(region,rv3d,o2)
                q3=safe_project(region,rv3d,o3)
                draw_line(q1, q2, col, va.thickness, region)
                draw_line(q3, q2, col, va.thickness, region)
                try:
                    vec1 = (o1-o2).normalized(); vec2 = (o3-o2).normalized()
                    angle = vec1.angle(vec2)
                    R = max((o1-o2).length, (o3-o2).length) * 0.3
                    axis = vec1.cross(vec2)
                    if axis.length < 1e-6: axis = Vector((0,0,1))
                    axis = axis.normalized()
                    perp = axis.cross(vec1).normalized()
                    arc = []
                    for i in range(17):
                        t = i / 16
                        th = t * angle
                        p3d = o2 + (vec1*math.cos(th) + perp*math.sin(th)) * R
                        p2d = safe_project(region, rv3d, p3d)
                        if p2d: arc.append(p2d)
                    draw_polyline(arc, col, va.thickness, region)
                    lbl = m.prefix + format_angle(angle, ss.angle_unit) + m.suffix
                    mid_n = (vec1+vec2).normalized() if (vec1+vec2).length > 1e-6 else Vector((0,0,1))
                    t3 = o2 + mid_n*(R + va.text_off)
                    draw_text(safe_project(region,rv3d,t3), lbl, col, va.font_size, va.text_bg)
                except Exception:
                    pass

            # ── DIAMETER ──────────────────────────────────────────────────────
            vd = ss.diameter
            for m in obj.mrm_diameter_measures:
                if not m.show: continue
                w1, w2 = vw.get(m.v1), vw.get(m.v2)
                if not w1 or not w2: continue
                col = tuple(m.custom_color) if m.use_custom_color else tuple(vd.color)
                p1w, n1 = w1; p2w, n2 = w2
                off = vd.normal_off
                o1 = p1w + n1*off; o2 = p2w + n2*off
                draw_line(safe_project(region,rv3d,p1w), safe_project(region,rv3d,o1), col, vd.thickness, region)
                draw_line(safe_project(region,rv3d,p2w), safe_project(region,rv3d,o2), col, vd.thickness, region)
                draw_arrow_2d(safe_project(region,rv3d,o1), safe_project(region,rv3d,o2), col, vd.arrow_size, vd.thickness, region)
                diam = (p2w - p1w).length
                lbl  = m.prefix + "Ø " + format_distance(diam, ss.unit_mode) + m.suffix
                avg_n = (n1+n2)
                avg_n = avg_n.normalized() if avg_n.length > 1e-6 else Vector((0,0,1))
                t3 = (o1+o2)*0.5 + avg_n*vd.text_off
                draw_text(safe_project(region,rv3d,t3), lbl, col, vd.font_size, vd.text_bg)
                if m.show_circle:
                    centre = (p1w+p2w)*0.5
                    circ_n = avg_n
                    _draw_circle_3d(centre, diam*0.5, circ_n, off, col, vd.thickness, region, rv3d)

            # ── RADIUS ────────────────────────────────────────────────────────
            vr = ss.radius
            for m in obj.mrm_radius_measures:
                if not m.show: continue
                w1, w2 = vw.get(m.v1), vw.get(m.v2)
                if not w1 or not w2: continue
                col = tuple(m.custom_color) if m.use_custom_color else tuple(vr.color)
                p1w, n1 = w1; p2w, n2 = w2
                off = vr.normal_off
                o1 = p1w + n1*off; o2 = p2w + n2*off
                draw_line(safe_project(region,rv3d,p1w), safe_project(region,rv3d,o1), col, vr.thickness, region)
                draw_line(safe_project(region,rv3d,p2w), safe_project(region,rv3d,o2), col, vr.thickness, region)
                q1 = safe_project(region,rv3d,o1)
                q2 = safe_project(region,rv3d,o2)
                draw_line(q1, q2, col, vr.thickness, region)
                draw_arrow_head(q2, (q2[0]-q1[0], q2[1]-q1[1]), col, vr.arrow_size, vr.thickness, region)
                rad = (p2w - p1w).length
                lbl = m.prefix + "R " + format_distance(rad, ss.unit_mode) + m.suffix
                avg_n = (n1+n2)
                avg_n = avg_n.normalized() if avg_n.length > 1e-6 else Vector((0,0,1))
                t3 = (o1+o2)*0.5 + avg_n*vr.text_off
                draw_text(safe_project(region,rv3d,t3), lbl, col, vr.font_size, vr.text_bg)
                if m.show_circle:
                    circ_n = (n1+n2)
                    circ_n = circ_n.normalized() if circ_n.length > 1e-6 else Vector((0,0,1))
                    _draw_circle_3d(p1w, rad, circ_n, off, col, vr.thickness, region, rv3d)

        gpu.state.blend_set('NONE')
    except Exception:
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
#  UIList
# ──────────────────────────────────────────────────────────────────────────────

class MRM_UL_Measures(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type not in {'DEFAULT', 'COMPACT'}:
            return
        row = layout.row(align=True)
        # visibility eye
        row.operator("mrm.toggle_visibility", text="",
                     icon='HIDE_OFF' if item.show else 'HIDE_ON',
                     emboss=False).index = index
        # Hack: pass kind via a second op with the same index — we set it later in panel
        row.prop(item, "name", text="", emboss=False,
                 icon='DRIVER_DISTANCE' if item.kind=='DISTANCE' else
                      'DRIVER_ROTATIONAL_DIFFERENCE' if item.kind=='ANGLE' else
                      'MESH_CIRCLE')
        # copy
        cp = row.operator("mrm.copy_value", text="", icon='COPYDOWN', emboss=False)
        cp.index = index; cp.kind = item.kind
        # circle toggle for diam/radius
        if item.kind in ('DIAMETER','RADIUS'):
            ci = row.operator("mrm.toggle_circle", text="",
                              icon='ANTIALIASED' if item.show_circle else 'MESH_CIRCLE',
                              emboss=False)
            ci.index = index; ci.kind = item.kind
        # delete
        dl = row.operator("mrm.delete_measure", text="", icon='X', emboss=False)
        dl.index = index; dl.kind = item.kind


# ──────────────────────────────────────────────────────────────────────────────
#  Panel
# ──────────────────────────────────────────────────────────────────────────────

class MRM_PT_Main(Panel):
    bl_label       = "MRM Metrics Pro"
    bl_idname      = "MRM_PT_Main"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "MRM Metrics"

    def draw_header(self, context):
        self.layout.label(icon='DRIVER_DISTANCE')

    def draw(self, context):
        layout = self.layout
        ss  = context.scene.mrm_settings
        wm  = context.window_manager
        obj = context.active_object
        running = getattr(wm, "mrm_running", False)

        # ── Master toggle ──────────────────────────────────────────────────────
        box = layout.box()
        row = box.row(align=True)
        icon = 'RADIOBUT_ON' if running else 'RADIOBUT_OFF'
        row.operator("mrm.toggle_draw",
                     text="● Live  ON" if running else "○ Live  OFF",
                     icon=icon,
                     depress=running)
        row.prop(ss, "show_all", text="", icon='HIDE_OFF' if ss.show_all else 'HIDE_ON', toggle=True)

        # ── Global settings ───────────────────────────────────────────────────
        col = box.column(align=True)
        row2 = col.row(align=True)
        row2.prop(ss, "unit_mode",  text="Dist")
        row2.prop(ss, "angle_unit", text="Ang")

        # ── Export row ────────────────────────────────────────────────────────
        row3 = box.row(align=True)
        row3.operator("mrm.export_json", text="JSON", icon='FILE_TEXT')
        row3.operator("mrm.export_csv",  text="CSV",  icon='FILE')
        row3.operator("mrm.clear_all",   text="",     icon='TRASH')

        layout.separator(factor=0.5)

        # ── Per-type sections ─────────────────────────────────────────────────
        for kind, label, attr, ss_attr, idx_attr in [
            ('DISTANCE', "Distance",  "mrm_distance_measures", "distance", "mrm_active_distance_index"),
            ('ANGLE',    "Angle",     "mrm_angle_measures",    "angle",    "mrm_active_angle_index"),
            ('DIAMETER', "Diameter",  "mrm_diameter_measures", "diameter", "mrm_active_diameter_index"),
            ('RADIUS',   "Radius",    "mrm_radius_measures",   "radius",   "mrm_active_radius_index"),
        ]:
            self._draw_type_section(
                layout, context, kind, label, attr, ss_attr, idx_attr, obj, ss)

    # ── Helper ────────────────────────────────────────────────────────────────
    def _draw_type_section(self, layout, context, kind, label, attr, ss_attr, idx_attr, obj, ss):
        expand_attr = f"expand_{ss_attr}"
        vs   = getattr(ss, ss_attr)
        box  = layout.box()
        hrow = box.row()
        hrow.prop(ss, expand_attr, text=label, emboss=False,
                  icon='TRIA_DOWN' if getattr(ss, expand_attr) else 'TRIA_RIGHT')
        if not getattr(ss, expand_attr):
            return

        # Create button
        create_map = {
            'DISTANCE': ("mrm.create_distance", 'ARROW_LEFTRIGHT'),
            'ANGLE'   : ("mrm.create_angle",    'DRIVER_ROTATIONAL_DIFFERENCE'),
            'DIAMETER': ("mrm.create_diameter",  'MESH_CIRCLE'),
            'RADIUS'  : ("mrm.create_radius",    'SPHERECURVE'),
        }
        op_id, op_icon = create_map[kind]
        box.operator(op_id, text=f"Add {label}", icon=op_icon)

        # Visual settings (collapsible via sub-box)
        sub = box.column(align=True)
        sub.prop(vs, "color",      text="Color")
        sub.prop(vs, "thickness",  text="Thickness")
        sub.prop(vs, "arrow_size", text="Arrow Size")
        sub.prop(vs, "normal_off", text="Normal Offset")
        sub.prop(vs, "text_off",   text="Text Offset")
        sub.prop(vs, "font_size",  text="Font Size")
        sub.prop(vs, "text_bg",    text="Text Background")

        box.separator(factor=0.4)

        if obj is None:
            box.label(text="No active object", icon='ERROR')
            return

        # Header row with count & toggle-all
        coll = getattr(obj, attr, [])
        hdr  = box.row()
        hdr.label(text=f"{label}s: {len(coll)}")
        tog = hdr.operator("mrm.toggle_all_visibility", text="",
                           icon='HIDE_OFF' if all(m.show for m in coll) else 'HIDE_ON',
                           emboss=False)
        tog.kind = kind
        clr = hdr.operator("mrm.clear_kind", text="", icon='TRASH', emboss=False)
        clr.kind = kind

        # UIList
        row = box.row()
        col_l = row.column()
        col_l.template_list("MRM_UL_Measures", kind, obj, attr,
                            context.scene, idx_attr, rows=3)

        # Selected measure detail panel
        idx = getattr(context.scene, idx_attr, -1)
        if obj and 0 <= idx < len(coll):
            m = coll[idx]
            det = box.box()
            det.label(text="Selected Measure", icon='INFO')
            det.prop(m, "name",   text="Name")
            det.prop(m, "prefix", text="Prefix")
            det.prop(m, "suffix", text="Suffix")
            det.prop(m, "note",   text="Note")
            det.prop(m, "use_custom_color", text="Custom Color")
            if m.use_custom_color:
                det.prop(m, "custom_color", text="")


# ──────────────────────────────────────────────────────────────────────────────
#  Registration
# ──────────────────────────────────────────────────────────────────────────────

classes = [
    # Property Groups
    MRM_TypeVisualProps,
    MRM_Measure,
    MRM_Settings,
    # Preferences
    MRM_AddonPreferences,
    # Operators
    MRM_OT_ToggleDraw,
    MRM_OT_CreateDistance,
    MRM_OT_CreateAngle,
    MRM_OT_CreateDiameter,
    MRM_OT_CreateRadius,
    MRM_OT_DeleteMeasure,
    MRM_OT_ClearAll,
    MRM_OT_ClearKind,
    MRM_OT_ToggleVisibility,
    MRM_OT_ToggleAllVisibility,
    MRM_OT_ToggleCircle,
    MRM_OT_CopyValue,
    MRM_OT_ExportJSON,
    MRM_OT_ExportCSV,
    # UIList
    MRM_UL_Measures,
    # Panel
    MRM_PT_Main,
]


def _apply_default_colors():
    """Apply design-spec colors to all scenes after registration."""
    try:
        for scene in bpy.data.scenes:
            ss = scene.mrm_settings
            ss.distance.color  = COL_DISTANCE
            ss.angle.color     = COL_ANGLE
            ss.diameter.color  = COL_DIAMETER
            ss.radius.color    = COL_RADIUS
    except Exception:
        pass


@bpy.app.handlers.persistent
def _load_post_handler(dummy):
    """Restore overlay state after file load if auto_start is set."""
    try:
        prefs = bpy.context.preferences.addons[ADDON_ID].preferences
        if prefs.auto_start:
            if not getattr(bpy.context.window_manager, "mrm_running", False):
                bpy.ops.mrm.toggle_draw()
    except Exception:
        pass


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Per-object measure collections
    bpy.types.Object.mrm_distance_measures = CollectionProperty(type=MRM_Measure)
    bpy.types.Object.mrm_angle_measures    = CollectionProperty(type=MRM_Measure)
    bpy.types.Object.mrm_diameter_measures = CollectionProperty(type=MRM_Measure)
    bpy.types.Object.mrm_radius_measures   = CollectionProperty(type=MRM_Measure)

    # Scene-level settings and list indices
    bpy.types.Scene.mrm_settings               = PointerProperty(type=MRM_Settings)
    bpy.types.Scene.mrm_active_distance_index  = IntProperty(default=-1)
    bpy.types.Scene.mrm_active_angle_index     = IntProperty(default=-1)
    bpy.types.Scene.mrm_active_diameter_index  = IntProperty(default=-1)
    bpy.types.Scene.mrm_active_radius_index    = IntProperty(default=-1)

    # Global draw-running flag
    bpy.types.WindowManager.mrm_running = BoolProperty(default=False)

    # Handlers
    if _load_post_handler not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_load_post_handler)

    # Deferred color initialisation
    bpy.app.timers.register(_apply_default_colors, first_interval=0.1)

    print(f"✔  MRM Metrics Pro v{'.'.join(str(x) for x in bl_info['version'])} registered")


def unregister():
    global _DRAW_HANDLE

    # Remove overlay
    if _DRAW_HANDLE is not None:
        try: bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, 'WINDOW')
        except Exception: pass
    _DRAW_HANDLE = None

    # Remove handlers
    if _load_post_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_load_post_handler)

    for cls in reversed(classes):
        try: bpy.utils.unregister_class(cls)
        except Exception: pass

    for attr in ("mrm_distance_measures", "mrm_angle_measures",
                 "mrm_diameter_measures", "mrm_radius_measures"):
        try: delattr(bpy.types.Object, attr)
        except Exception: pass

    for attr in ("mrm_settings", "mrm_active_distance_index",
                 "mrm_active_angle_index", "mrm_active_diameter_index",
                 "mrm_active_radius_index"):
        try: delattr(bpy.types.Scene, attr)
        except Exception: pass

    try: del bpy.types.WindowManager.mrm_running
    except Exception: pass

    print("✖  MRM Metrics Pro unregistered")


if __name__ == "__main__":
    register()
