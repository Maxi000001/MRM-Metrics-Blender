# -*- coding: utf-8 -*-
bl_info = {
    "name": "MRM Metrics",
    "author": "Malik + Assistant",
    "version": (4, 5, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > MRM Metrics",
    "description": "Distance / Angle / Diameter measurements with improved UI and normal offset for all types",
    "category": "3D View",
}

import bpy
import bmesh
import math
import traceback
from mathutils import Vector
import gpu
import blf
from gpu_extras.batch import batch_for_shader
from bpy.props import (
    FloatProperty, FloatVectorProperty, EnumProperty, BoolProperty,
    IntProperty, CollectionProperty, PointerProperty, StringProperty
)
from bpy.types import (
    Operator, Panel, PropertyGroup, UIList
)
from bpy_extras import view3d_utils

# ---------- Globals ----------
_SHADER = None
_FONT_ID = 0
_DRAW_HANDLE = None

# default colour tuples - ИСПРАВЛЕНЫ ЦВЕТА ПО УМОЛЧАНИЮ
DEFAULT_COL_DISTANCE = (0.0, 1.0, 0.0, 1.0)   # green
DEFAULT_COL_ANGLE    = (0.0, 0.3, 1.0, 1.0)   # blue
DEFAULT_COL_DIAMETER = (1.0, 1.0, 0.0, 1.0)   # yellow

METRIC_COLLECTION_NAME = "Metric Collection"
METRIC_SUBCOLS = {
    'DISTANCE': "Distance",
    'ANGLE': "Angle",
    'DIAMETER': "Diameter",
}

# ---------- helpers ----------
def tag_redraw_all_view3d():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

def safe_project(region, rv3d, coord):
    try:
        p = view3d_utils.location_3d_to_region_2d(region, rv3d, coord)
        if p is None:
            return None
        return (p.x, p.y)
    except Exception:
        return None

def get_shader():
    global _SHADER
    if _SHADER is None:
        _SHADER = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    return _SHADER

def draw_line_2d(p1, p2, color, width):
    if p1 is None or p2 is None:
        return
    coords = [(p1[0], p1[1], 0.0), (p2[0], p2[1], 0.0)]
    try:
        shader = get_shader()
        shader.bind()
        shader.uniform_float("color", color)
        shader.uniform_float("lineWidth", float(max(1.0, width)))
        region = bpy.context.region
        shader.uniform_float("viewportSize", (region.width, region.height))
        batch = batch_for_shader(shader, 'LINES', {"pos": coords})
        batch.draw(shader)
    except Exception:
        pass

def draw_arrow_2d(p1, p2, color, size_px=12, width=1.0):
    if p1 is None or p2 is None:
        return
    draw_line_2d(p1, p2, color, width)
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    ux = dx / dist; uy = dy / dist
    px = -uy; py = ux
    a1 = (p1[0] + ux * size_px + px * (size_px*0.45), p1[1] + uy * size_px + py * (size_px*0.45))
    a2 = (p1[0] + ux * size_px - px * (size_px*0.45), p1[1] + uy * size_px - py * (size_px*0.45))
    b1 = (p2[0] - ux * size_px + px * (size_px*0.45), p2[1] - uy * size_px + py * (size_px*0.45))
    b2 = (p2[0] - ux * size_px - px * (size_px*0.45), p2[1] - uy * size_px - py * (size_px*0.45))
    draw_line_2d(p1, a1, color, width)
    draw_line_2d(p1, a2, color, width)
    draw_line_2d(p2, b1, color, width)
    draw_line_2d(p2, b2, color, width)

def draw_text_2d(pos, text, color, font_size):
    if pos is None:
        return
    ui_scale = bpy.context.preferences.system.ui_scale
    size = max(8, int(font_size * ui_scale))
    blf.position(_FONT_ID, pos[0], pos[1], 0)
    blf.size(_FONT_ID, size)
    blf.color(_FONT_ID, color[0], color[1], color[2], color[3])
    blf.draw(_FONT_ID, text)

def format_units(val, mode):
    if mode == 'BU':
        return f"{val:.4f} BU"
    if mode == 'M':
        return f"{val:.4f} m"
    if mode == 'CM':
        return f"{val*100:.2f} cm"
    if mode == 'MM':
        return f"{val*1000:.1f} mm"
    if mode == 'AUTO':
        if abs(val) >= 1.0:
            return f"{val:.4f} m"
        elif abs(val) >= 0.01:
            return f"{val*100:.2f} cm"
        else:
            return f"{val*1000:.1f} mm"
    return f"{val:.4f}"

# ---------- Metric collection helpers ----------
def ensure_metric_collections(context):
    scene_col = context.scene.collection
    main = bpy.data.collections.get(METRIC_COLLECTION_NAME)
    if main is None:
        main = bpy.data.collections.new(METRIC_COLLECTION_NAME)
        try:
            scene_col.children.link(main)
        except Exception:
            pass
    else:
        if main.name not in [c.name for c in scene_col.children]:
            try:
                scene_col.children.link(main)
            except Exception:
                pass

    for kind, subname in METRIC_SUBCOLS.items():
        sub = bpy.data.collections.get(subname)
        if sub is None:
            sub = bpy.data.collections.new(subname)
            try:
                main.children.link(sub)
            except Exception:
                pass
        else:
            if sub.name not in [c.name for c in main.children]:
                try:
                    main.children.link(sub)
                except Exception:
                    pass
    return main

def link_object_to_metric_subcollection(obj, kind, context):
    try:
        if obj is None:
            return
        ensure_metric_collections(context)
        subname = METRIC_SUBCOLS.get(kind)
        if subname is None:
            return
        sub = bpy.data.collections.get(subname)
        if sub is None:
            return
        if obj.name not in [o.name for o in sub.objects]:
            try:
                sub.objects.link(obj)
            except Exception:
                pass
    except Exception:
        traceback.print_exc()

def unlink_object_from_metric_subcollection_if_empty(obj, kind, context):
    try:
        if obj is None:
            return
        has = False
        if kind == 'DISTANCE':
            has = len(getattr(obj, "mrm_distance_measures", [])) > 0
        elif kind == 'ANGLE':
            has = len(getattr(obj, "mrm_angle_measures", [])) > 0
        elif kind == 'DIAMETER':
            has = len(getattr(obj, "mrm_diameter_measures", [])) > 0
        subname = METRIC_SUBCOLS.get(kind)
        sub = bpy.data.collections.get(subname)
        if sub and not has:
            try:
                if obj.name in [o.name for o in sub.objects]:
                    sub.objects.unlink(obj)
            except Exception:
                pass
    except Exception:
        traceback.print_exc()

# ---------- Data classes ----------
class MRM_Measure(PropertyGroup):
    name: StringProperty(name="Name", default="Measure")
    kind: EnumProperty(name="Type", items=[
        ('DISTANCE','Distance',''),
        ('ANGLE','Angle',''),
        ('DIAMETER','Diameter','')
    ], default='DISTANCE')
    v1: IntProperty(default=-1)
    v2: IntProperty(default=-1)
    v3: IntProperty(default=-1)  # for angle
    # Visual params (stored per-measure so they persist & can be individual if needed)
    color: FloatVectorProperty(size=4, default=(0.8,0.8,0.8,1.0))
    thickness: FloatProperty(default=2.0, min=1.0, max=20.0)
    normal_offset: FloatProperty(default=0.05)
    text_offset: FloatProperty(default=0.05)
    font_size: IntProperty(default=14, min=8, max=64)
    show: BoolProperty(default=True)
    # Diameter specific
    show_circle: BoolProperty(name="Show Circle", default=False, description="Show circle for diameter measurement")

class MRM_DistanceSettings(PropertyGroup):
    color: FloatVectorProperty(
        name="Color", subtype='COLOR', size=4,
        default=DEFAULT_COL_DISTANCE, min=0.0, max=1.0
    )
    thickness: FloatProperty(name="Thickness", default=2.0, min=1.0, max=20.0)
    normal_offset: FloatProperty(name="Normal offset", default=0.05)
    text_offset: FloatProperty(name="Text offset", default=0.05)
    font_size: IntProperty(name="Font size", default=14, min=8, max=64)

class MRM_AngleSettings(PropertyGroup):
    color: FloatVectorProperty(
        name="Color", subtype='COLOR', size=4,
        default=DEFAULT_COL_ANGLE, min=0.0, max=1.0
    )
    thickness: FloatProperty(name="Thickness", default=2.0, min=1.0, max=20.0)
    normal_offset: FloatProperty(name="Normal offset", default=0.05)
    text_offset: FloatProperty(name="Text offset", default=0.05)
    font_size: IntProperty(name="Font size", default=14, min=8, max=64)

class MRM_DiameterSettings(PropertyGroup):
    color: FloatVectorProperty(
        name="Color", subtype='COLOR', size=4,
        default=DEFAULT_COL_DIAMETER, min=0.0, max=1.0
    )
    thickness: FloatProperty(name="Thickness", default=2.0, min=1.0, max=20.0)
    normal_offset: FloatProperty(name="Normal offset", default=0.05)
    text_offset: FloatProperty(name="Text offset", default=0.05)
    font_size: IntProperty(name="Font size", default=14, min=8, max=64)

class MRM_Settings(PropertyGroup):
    unit_mode: EnumProperty(
        name="Units",
        items=[
            ('AUTO','Auto','Auto choose m/cm/mm'),
            ('M','Meters','Meters'),
            ('CM','Centimeters','Centimeters'),
            ('MM','Millimeters','Millimeters'),
            ('BU','Blender Units','Blender units (raw)'),
        ],
        default='AUTO'
    )
    distance: PointerProperty(type=MRM_DistanceSettings)
    angle: PointerProperty(type=MRM_AngleSettings)
    diameter: PointerProperty(type=MRM_DiameterSettings)
    expand_distance: BoolProperty(name="Expand Distance", default=True)
    expand_angle: BoolProperty(name="Expand Angle", default=True)
    expand_diameter: BoolProperty(name="Expand Diameter", default=True)

# ---------- Utility: propagate type settings to existing measures ----------
def propagate_type_settings(kind, ss):
    """
    Apply per-type settings to all existing measures of that type across all objects.
    This ensures all measures have identical visuals after changing panel settings.
    """
    try:
        for o in bpy.data.objects:
            try:
                if kind == 'DISTANCE':
                    coll = getattr(o, "mrm_distance_measures", None)
                    if coll:
                        for m in coll:
                            m.color = ss.distance.color
                            m.thickness = ss.distance.thickness
                            m.normal_offset = ss.distance.normal_offset
                            m.text_offset = ss.distance.text_offset
                            m.font_size = ss.distance.font_size
                elif kind == 'ANGLE':
                    coll = getattr(o, "mrm_angle_measures", None)
                    if coll:
                        for m in coll:
                            m.color = ss.angle.color
                            m.thickness = ss.angle.thickness
                            m.normal_offset = ss.angle.normal_offset
                            m.text_offset = ss.angle.text_offset
                            m.font_size = ss.angle.font_size
                elif kind == 'DIAMETER':
                    coll = getattr(o, "mrm_diameter_measures", None)
                    if coll:
                        for m in coll:
                            m.color = ss.diameter.color
                            m.thickness = ss.diameter.thickness
                            m.normal_offset = ss.diameter.normal_offset
                            m.text_offset = ss.diameter.text_offset
                            m.font_size = ss.diameter.font_size
            except Exception:
                pass
        tag_redraw_all_view3d()
    except Exception:
        traceback.print_exc()

# ---------- selected vertices helper ----------
def get_selected_vertices_indices(obj):
    if obj is None or obj.type != 'MESH':
        return []
    if obj.mode != 'EDIT':
        return []
    try:
        bm = bmesh.from_edit_mesh(obj.data)
    except Exception:
        return []
    bm.verts.ensure_lookup_table()
    try:
        hist = [e for e in bm.select_history if isinstance(e, bmesh.types.BMVert)]
    except Exception:
        hist = []
    if hist:
        sel_hist = [v.index for v in hist if v.select]
        if sel_hist:
            return sel_hist
    sel = [v.index for v in bm.verts if v.select]
    return sel

# ---------- Diameter detection ----------
def find_farthest_vertices(obj):
    """Find the two farthest apart selected vertices"""
    if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
        return None, None
    
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        
        selected_verts = [v for v in bm.verts if v.select]
        
        if len(selected_verts) < 2:
            return None, None
        
        # Find the two farthest vertices
        max_dist = 0
        v1 = v2 = None
        
        for i, vert_a in enumerate(selected_verts):
            for vert_b in selected_verts[i+1:]:
                dist = (vert_a.co - vert_b.co).length
                if dist > max_dist:
                    max_dist = dist
                    v1 = vert_a
                    v2 = vert_b
        
        if v1 and v2:
            return v1.index, v2.index
        else:
            return None, None
        
    except Exception as e:
        print(f"Error in diameter detection: {e}")
        return None, None

# ---------- Operators ----------
class MRM_OT_create_distance(Operator):
    bl_idname = "mrm.create_distance"
    bl_label = "Create Distance"
    bl_description = "Create distance measurement between two selected vertices (Edit Mode)"
    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'WARNING'}, "No active object")
            return {'CANCELLED'}
        inds = get_selected_vertices_indices(obj)
        if len(inds) != 2:
            self.report({'WARNING'}, "Please select exactly 2 vertices in Edit Mode.")
            return {'CANCELLED'}
        ss = context.scene.mrm_settings
        i1, i2 = inds[0], inds[1]
        meas = obj.mrm_distance_measures.add()
        meas.kind = 'DISTANCE'; meas.v1 = i1; meas.v2 = i2
        # inherit current type settings
        meas.color = ss.distance.color
        meas.thickness = ss.distance.thickness
        meas.normal_offset = ss.distance.normal_offset
        meas.text_offset = ss.distance.text_offset
        meas.font_size = ss.distance.font_size
        meas.name = f"Dist {i1}-{i2}"
        link_object_to_metric_subcollection(obj, 'DISTANCE', context)
        tag_redraw_all_view3d()
        return {'FINISHED'}

class MRM_OT_create_angle(Operator):
    bl_idname = "mrm.create_angle"
    bl_label = "Create Angle"
    bl_description = "Create angle measurement (select 3 vertices, center is middle one) in Edit Mode"
    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'WARNING'}, "No active object")
            return {'CANCELLED'}
        inds = get_selected_vertices_indices(obj)
        if len(inds) != 3:
            self.report({'WARNING'}, "Please select exactly 3 vertices (center must be one of them).")
            return {'CANCELLED'}
        ss = context.scene.mrm_settings
        i1, i2, i3 = inds[0], inds[1], inds[2]
        meas = obj.mrm_angle_measures.add()
        meas.kind = 'ANGLE'; meas.v1 = i1; meas.v2 = i2; meas.v3 = i3
        meas.color = ss.angle.color
        meas.thickness = ss.angle.thickness
        meas.normal_offset = ss.angle.normal_offset
        meas.text_offset = ss.angle.text_offset
        meas.font_size = ss.angle.font_size
        meas.name = f"Angle {i1}-{i2}-{i3}"
        link_object_to_metric_subcollection(obj, 'ANGLE', context)
        tag_redraw_all_view3d()
        return {'FINISHED'}

class MRM_OT_create_diameter(Operator):
    bl_idname = "mrm.create_diameter"
    bl_label = "Create Diameter"
    bl_description = "Create diameter measurement between two farthest selected vertices (Edit Mode)"
    
    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'WARNING'}, "No active object")
            return {'CANCELLED'}
        
        # Find the two farthest vertices
        v1_idx, v2_idx = find_farthest_vertices(obj)
        
        if v1_idx is None or v2_idx is None:
            self.report({'WARNING'}, "Please select at least 2 vertices to measure diameter")
            return {'CANCELLED'}
        
        ss = context.scene.mrm_settings
        meas = obj.mrm_diameter_measures.add()
        meas.kind = 'DIAMETER'
        meas.v1 = v1_idx
        meas.v2 = v2_idx
        meas.color = ss.diameter.color
        meas.thickness = ss.diameter.thickness
        meas.normal_offset = ss.diameter.normal_offset
        meas.text_offset = ss.diameter.text_offset
        meas.font_size = ss.diameter.font_size
        meas.show_circle = False  # Circle disabled by default
        meas.name = f"Diameter {v1_idx}-{v2_idx}"
        link_object_to_metric_subcollection(obj, 'DIAMETER', context)
        tag_redraw_all_view3d()
        return {'FINISHED'}

class MRM_OT_clear_all(Operator):
    bl_idname = "mrm.clear_all_measures"
    bl_label = "Clear All Measures"
    def execute(self, context):
        removed = 0
        for o in context.scene.objects:
            try:
                removed += len(o.mrm_distance_measures)
                removed += len(o.mrm_angle_measures)
                removed += len(o.mrm_diameter_measures)
                o.mrm_distance_measures.clear()
                o.mrm_angle_measures.clear()
                o.mrm_diameter_measures.clear()
                for k in METRIC_SUBCOLS.keys():
                    unlink_object_from_metric_subcollection_if_empty(o, k, context)
            except Exception:
                pass
        self.report({'INFO'}, f"Cleared {removed} measures across objects")
        tag_redraw_all_view3d()
        return {'FINISHED'}

class MRM_OT_clear_type(Operator):
    bl_idname = "mrm.clear_type_measures"
    bl_label = "Clear Measures (type)"
    kind: StringProperty()
    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'WARNING'}, "No active object")
            return {'CANCELLED'}
        removed = 0
        try:
            if self.kind == 'DISTANCE':
                removed = len(obj.mrm_distance_measures); obj.mrm_distance_measures.clear()
            elif self.kind == 'ANGLE':
                removed = len(obj.mrm_angle_measures); obj.mrm_angle_measures.clear()
            elif self.kind == 'DIAMETER':
                removed = len(obj.mrm_diameter_measures); obj.mrm_diameter_measures.clear()
            unlink_object_from_metric_subcollection_if_empty(obj, self.kind, context)
        except Exception:
            pass
        self.report({'INFO'}, f"Cleared {removed} {self.kind} measures on active object")
        tag_redraw_all_view3d()
        return {'FINISHED'}

class MRM_OT_delete_measure(Operator):
    bl_idname = "mrm.delete_measure"
    bl_label = "Delete Selected Measure"
    index: IntProperty()
    kind: StringProperty()
    def execute(self, context):
        obj = context.active_object
        if obj is None:
            return {'CANCELLED'}
        try:
            if self.kind == 'DISTANCE':
                coll = obj.mrm_distance_measures
            elif self.kind == 'ANGLE':
                coll = obj.mrm_angle_measures
            elif self.kind == 'DIAMETER':
                coll = obj.mrm_diameter_measures
            if 0 <= self.index < len(coll):
                coll.remove(self.index)
            else:
                self.report({'WARNING'}, "Index out of range")
                return {'CANCELLED'}
            unlink_object_from_metric_subcollection_if_empty(obj, self.kind, context)
        except Exception:
            self.report({'WARNING'}, "Failed to delete")
            return {'CANCELLED'}
        tag_redraw_all_view3d()
        return {'FINISHED'}

class MRM_OT_toggle_measure_visibility(Operator):
    bl_idname = "mrm.toggle_measure_visibility"
    bl_label = "Toggle Measure Visibility"
    bl_description = "Show/hide this specific measure"
    index: IntProperty()
    kind: StringProperty()
    
    def execute(self, context):
        obj = context.active_object
        if obj is None:
            return {'CANCELLED'}
        try:
            if self.kind == 'DISTANCE':
                coll = obj.mrm_distance_measures
            elif self.kind == 'ANGLE':
                coll = obj.mrm_angle_measures
            elif self.kind == 'DIAMETER':
                coll = obj.mrm_diameter_measures
            else:
                return {'CANCELLED'}
            
            if 0 <= self.index < len(coll):
                meas = coll[self.index]
                meas.show = not meas.show
        except Exception:
            pass
        tag_redraw_all_view3d()
        return {'FINISHED'}

class MRM_OT_toggle_all_visibility(Operator):
    bl_idname = "mrm.toggle_all_visibility"
    bl_label = "Toggle All Visibility"
    bl_description = "Show/hide all measurements of this type"
    kind: StringProperty()
    
    def execute(self, context):
        obj = context.active_object
        if obj is None:
            return {'CANCELLED'}
        
        try:
            if self.kind == 'DISTANCE':
                coll = obj.mrm_distance_measures
            elif self.kind == 'ANGLE':
                coll = obj.mrm_angle_measures
            elif self.kind == 'DIAMETER':
                coll = obj.mrm_diameter_measures
            else:
                return {'CANCELLED'}
            
            # Check if all are visible or all are hidden
            all_visible = all(m.show for m in coll)
            new_state = not all_visible
            
            for m in coll:
                m.show = new_state
                
        except Exception:
            pass
        
        tag_redraw_all_view3d()
        return {'FINISHED'}

class MRM_OT_toggle_diameter_circle(Operator):
    bl_idname = "mrm.toggle_diameter_circle"
    bl_label = "Toggle Diameter Circle"
    bl_description = "Show/hide circle for diameter measurement"
    index: IntProperty()
    
    def execute(self, context):
        obj = context.active_object
        if obj is None:
            return {'CANCELLED'}
        try:
            if 0 <= self.index < len(obj.mrm_diameter_measures):
                meas = obj.mrm_diameter_measures[self.index]
                meas.show_circle = not meas.show_circle
        except Exception:
            pass
        tag_redraw_all_view3d()
        return {'FINISHED'}

class MRM_OT_delete_all_in_list(Operator):
    bl_idname = "mrm.delete_all_in_list"
    bl_label = "Delete All In List"
    kind: StringProperty()
    def execute(self, context):
        obj = context.active_object
        if obj is None:
            self.report({'WARNING'}, "No active object")
            return {'CANCELLED'}
        try:
            if self.kind == 'DISTANCE':
                obj.mrm_distance_measures.clear()
            elif self.kind == 'ANGLE':
                obj.mrm_angle_measures.clear()
            elif self.kind == 'DIAMETER':
                obj.mrm_diameter_measures.clear()
            unlink_object_from_metric_subcollection_if_empty(obj, self.kind, context)
        except Exception:
            pass
        tag_redraw_all_view3d()
        return {'FINISHED'}

# Toggle draw handler
class MRM_OT_toggle_draw(Operator):
    bl_idname = "mrm.toggle_draw"
    bl_label = "Toggle MRM Draw"
    def execute(self, context):
        global _DRAW_HANDLE
        wm = context.window_manager
        if not hasattr(wm, "mrm_draw_running"):
            wm.mrm_draw_running = False
        if not wm.mrm_draw_running:
            ensure_metric_collections(context)
            _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(_draw_callback, (), 'WINDOW', 'POST_PIXEL')
            wm.mrm_draw_running = True
        else:
            try:
                if _DRAW_HANDLE is not None:
                    bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, 'WINDOW')
            except Exception:
                pass
            _DRAW_HANDLE = None
            wm.mrm_draw_running = False
        tag_redraw_all_view3d()
        return {'FINISHED'}

# ---------- UIList ----------
class MRM_UL_measures(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            
            # Кнопка видимости для каждого измерения
            icon = 'HIDE_OFF' if item.show else 'HIDE_ON'
            vis_op = row.operator("mrm.toggle_measure_visibility", text="", icon=icon, emboss=False)
            vis_op.index = index
            vis_op.kind = item.kind
            
            row.prop(item, "name", text="", emboss=False)
            ops = row.row(align=True)
            
            # For diameter, add circle toggle button
            if item.kind == 'DIAMETER':
                circle_icon = 'MESH_CIRCLE' if getattr(item, "show_circle", False) else 'MESH_CIRCLE'
                circle_op = ops.operator("mrm.toggle_diameter_circle", text="", icon=circle_icon, emboss=False)
                circle_op.index = index
            
            op = ops.operator("mrm.delete_measure", text="", icon='X', emboss=False)
            op.kind = item.kind
            op.index = index
            
        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.label(text="")

# ---------- Draw callback ----------
def _draw_callback():
    try:
        context = bpy.context
        area = context.area
        if area is None or area.type != 'VIEW_3D':
            return
        region = context.region
        rv3d = context.space_data.region_3d
        ss = context.scene.mrm_settings

        gpu.state.blend_set('ALPHA')

        for obj in context.visible_objects:
            if obj.type != 'MESH':
                continue

            verts_world = {}
            try:
                if obj.mode == 'EDIT':
                    bm = bmesh.from_edit_mesh(obj.data)
                    bm.verts.ensure_lookup_table()
                    for v in bm.verts:
                        verts_world[v.index] = (obj.matrix_world @ v.co, (obj.matrix_world.inverted_safe().transposed() @ v.normal).normalized())
                else:
                    for v in obj.data.vertices:
                        verts_world[v.index] = (obj.matrix_world @ v.co, (obj.matrix_world.inverted_safe().transposed() @ v.normal).normalized())
            except Exception:
                continue

            # DISTANCE
            try:
                col = tuple(ss.distance.color)
                thickness = ss.distance.thickness
                off_default = ss.distance.normal_offset
                text_off_default = ss.distance.text_offset
                font_default = ss.distance.font_size
                for meas in obj.mrm_distance_measures:
                    if not getattr(meas, "show", True):
                        continue
                    v1 = verts_world.get(meas.v1); v2 = verts_world.get(meas.v2)
                    if not v1 or not v2:
                        continue
                    p1w, n1 = v1; p2w, n2 = v2
                    p1_off = p1w + n1 * off_default
                    p2_off = p2w + n2 * off_default
                    p1_2d = safe_project(region, rv3d, p1w)
                    p2_2d = safe_project(region, rv3d, p2w)
                    p1o_2d = safe_project(region, rv3d, p1_off)
                    p2o_2d = safe_project(region, rv3d, p2_off)
                    draw_line_2d(p1_2d, p1o_2d, col, thickness)
                    draw_line_2d(p2_2d, p2o_2d, col, thickness)
                    draw_arrow_2d(p1o_2d, p2o_2d, col, size_px=12, width=thickness)
                    mid3 = (p1_off + p2_off) * 0.5
                    avg_n = (n1 + n2)
                    if avg_n.length != 0:
                        avg_n = avg_n.normalized()
                    text3 = mid3 + avg_n * text_off_default
                    text2 = safe_project(region, rv3d, text3)
                    dist = (p2w - p1w).length
                    draw_text_2d(text2, format_units(dist, ss.unit_mode), col, font_default)
            except Exception:
                traceback.print_exc()

            # ANGLE - with normal offset
            try:
                col = tuple(ss.angle.color)
                thickness = ss.angle.thickness
                off_default = ss.angle.normal_offset
                text_off_default = ss.angle.text_offset
                font_default = ss.angle.font_size
                for meas in obj.mrm_angle_measures:
                    if not getattr(meas, "show", True):
                        continue
                    v1 = verts_world.get(meas.v1); v2 = verts_world.get(meas.v2); v3 = verts_world.get(meas.v3)
                    if not v1 or not v2 or not v3:
                        continue
                    p1w, n1 = v1; p2w, n2 = v2; p3w, n3 = v3
                    
                    # Apply normal offset to all three points
                    p1_off = p1w + n1 * off_default
                    p2_off = p2w + n2 * off_default
                    p3_off = p3w + n3 * off_default
                    
                    # Draw connection lines from original to offset points
                    p1_2d = safe_project(region, rv3d, p1w)
                    p2_2d = safe_project(region, rv3d, p2w)
                    p3_2d = safe_project(region, rv3d, p3w)
                    p1o_2d = safe_project(region, rv3d, p1_off)
                    p2o_2d = safe_project(region, rv3d, p2_off)
                    p3o_2d = safe_project(region, rv3d, p3_off)
                    
                    draw_line_2d(p1_2d, p1o_2d, col, thickness)
                    draw_line_2d(p2_2d, p2o_2d, col, thickness)
                    draw_line_2d(p3_2d, p3o_2d, col, thickness)
                    
                    # Draw angle lines between offset points
                    draw_line_2d(p1o_2d, p2o_2d, col, thickness)
                    draw_line_2d(p3o_2d, p2o_2d, col, thickness)
                    
                    try:
                        vec1 = (p1_off - p2_off).normalized()
                        vec2 = (p3_off - p2_off).normalized()
                        angle = vec1.angle(vec2)
                        radius = max((p1_off - p2_off).length, (p3_off - p2_off).length) * 0.35
                        axis = vec1.cross(vec2)
                        if axis.length == 0:
                            axis = Vector((0,0,1))
                        axis = axis.normalized()
                        perp = axis.cross(vec1).normalized()
                        pts2d = []
                        samples = 10
                        for i in range(samples+1):
                            t = i / samples
                            theta = t * angle
                            p3d = p2_off + (vec1 * math.cos(theta) + perp * math.sin(theta)) * radius
                            p2d = safe_project(region, rv3d, p3d)
                            if p2d:
                                pts2d.append((p2d[0], p2d[1], 0.0))
                        if len(pts2d) >= 2:
                            shader = get_shader()
                            batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts2d})
                            shader.bind()
                            shader.uniform_float("color", col)
                            shader.uniform_float("lineWidth", float(max(1.0, thickness)))
                            shader.uniform_float("viewportSize", (region.width, region.height))
                            batch.draw(shader)
                        try:
                            text3 = p2_off + ((vec1 + vec2).normalized() * (radius + text_off_default))
                        except Exception:
                            text3 = p2_off + Vector((0,0,1)) * (radius + text_off_default)
                        text2 = safe_project(region, rv3d, text3)
                        draw_text_2d(text2, f"{math.degrees(angle):.1f}°", col, font_default)
                    except Exception:
                        pass
            except Exception:
                traceback.print_exc()

            # DIAMETER - with normal offset and 3D circle
            try:
                col = tuple(ss.diameter.color)
                thickness = ss.diameter.thickness
                off_default = ss.diameter.normal_offset
                text_off_default = ss.diameter.text_offset
                font_default = ss.diameter.font_size
                for meas in obj.mrm_diameter_measures:
                    if not getattr(meas, "show", True):
                        continue
                    v1 = verts_world.get(meas.v1); v2 = verts_world.get(meas.v2)
                    if not v1 or not v2:
                        continue
                    p1w, n1 = v1; p2w, n2 = v2
                    
                    # Apply normal offset to both points
                    p1_off = p1w + n1 * off_default
                    p2_off = p2w + n2 * off_default
                    
                    # Draw connection lines from original to offset points
                    p1_2d = safe_project(region, rv3d, p1w)
                    p2_2d = safe_project(region, rv3d, p2w)
                    p1o_2d = safe_project(region, rv3d, p1_off)
                    p2o_2d = safe_project(region, rv3d, p2_off)
                    
                    draw_line_2d(p1_2d, p1o_2d, col, thickness)
                    draw_line_2d(p2_2d, p2o_2d, col, thickness)
                    
                    # Draw diameter line between offset points
                    draw_line_2d(p1o_2d, p2o_2d, col, thickness)
                    
                    # Draw arrows at both ends
                    draw_arrow_2d(p1o_2d, p2o_2d, col, size_px=12, width=thickness)
                    
                    # Calculate diameter value
                    diameter = (p2w - p1w).length
                    
                    # Calculate midpoint for text
                    mid3 = (p1_off + p2_off) * 0.5
                    
                    # Calculate normal for text offset (average of both vertex normals)
                    avg_n = (n1 + n2)
                    if avg_n.length != 0:
                        avg_n = avg_n.normalized()
                    
                    # Position text with offset
                    text3 = mid3 + avg_n * text_off_default
                    text2 = safe_project(region, rv3d, text3)
                    
                    # Draw diameter symbol and value
                    draw_text_2d(text2, f"Ø {format_units(diameter, ss.unit_mode)}", col, font_default)
                    
                    # Draw 3D circle for diameter if enabled
                    if getattr(meas, "show_circle", False):
                        # Calculate circle center and radius
                        circle_center = (p1w + p2w) * 0.5
                        circle_radius = diameter * 0.5
                        
                        # Calculate circle normal (average of both vertex normals)
                        circle_normal = (n1 + n2)
                        if circle_normal.length == 0:
                            circle_normal = Vector((0, 0, 1))
                        else:
                            circle_normal = circle_normal.normalized()
                        
                        # Generate circle points in 3D space
                        segments = 32
                        circle_points = []
                        
                        # Find two perpendicular vectors in the circle plane
                        if abs(circle_normal.z) > 0.9:
                            tangent = Vector((1, 0, 0)).cross(circle_normal)
                        else:
                            tangent = Vector((0, 0, 1)).cross(circle_normal)
                        tangent = tangent.normalized()
                        bitangent = circle_normal.cross(tangent).normalized()
                        
                        for i in range(segments + 1):
                            angle = (i / segments) * 2 * math.pi
                            x = math.cos(angle) * circle_radius
                            y = math.sin(angle) * circle_radius
                            point = circle_center + tangent * x + bitangent * y
                            circle_points.append(point)
                        
                        # Project circle points to 2D and draw
                        circle_points_2d = []
                        for point in circle_points:
                            # Apply normal offset to circle points
                            point_off = point + circle_normal * off_default
                            p2d = safe_project(region, rv3d, point_off)
                            if p2d:
                                circle_points_2d.append((p2d[0], p2d[1], 0.0))
                        
                        # Draw the circle as LINE_LOOP
                        if len(circle_points_2d) >= 3:
                            shader = get_shader()
                            batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": circle_points_2d})
                            shader.bind()
                            shader.uniform_float("color", col)
                            shader.uniform_float("lineWidth", float(max(1.0, thickness)))
                            shader.uniform_float("viewportSize", (region.width, region.height))
                            batch.draw(shader)
            except Exception:
                traceback.print_exc()

        gpu.state.blend_set('NONE')
    except Exception:
        traceback.print_exc()
        try:
            bpy.context.window_manager.mrm_draw_running = False
        except Exception:
            pass

# ---------- UI Panel ----------
class MRM_PT_panel(Panel):
    bl_label = "MRM Metrics"
    bl_idname = "MRM_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MRM Metrics'

    def draw(self, context):
        layout = self.layout
        ss = context.scene.mrm_settings
        wm = context.window_manager
        obj = context.active_object

        # Top block
        box = layout.box()
        row = box.row(align=True)
        icon = 'PLAY' if not getattr(wm, "mrm_draw_running", False) else 'PAUSE'
        txt = "Show" if not getattr(wm, "mrm_draw_running", False) else "Hide"
        row.operator("mrm.toggle_draw", text=f"{txt} Measurements", icon=icon)
        
        row = box.row(align=True)
        row.prop(ss, "unit_mode", text="Units")
        row.operator("mrm.clear_all_measures", text="Clear All", icon='TRASH')

        layout.separator()

        # DISTANCE block - with large create button
        box = layout.box()
        header = box.row()
        header.prop(ss, "expand_distance", text="📏 Distance", emboss=False, icon='TRIA_DOWN' if ss.expand_distance else 'TRIA_RIGHT')
        
        if ss.expand_distance:
            # Large create button
            create_row = box.row()
            create_row.operator("mrm.create_distance", text="Create Distance", icon='ARROW_LEFTRIGHT')
            
            inner = box.column(align=True)
            inner.prop(ss.distance, "color", text="Color")
            inner.prop(ss.distance, "thickness", text="Thickness")
            inner.prop(ss.distance, "normal_offset", text="Normal offset")
            inner.prop(ss.distance, "text_offset", text="Text offset")
            inner.prop(ss.distance, "font_size", text="Font size")
            inner.separator()
            
            # Distance measures header with toggle all button
            if obj is None:
                inner.label(text="No active object")
            else:
                row = inner.row()
                row.label(text="Distance measures:")
                
                # Check if all distance measures are visible
                all_visible = True
                if obj.mrm_distance_measures:
                    all_visible = all(m.show for m in obj.mrm_distance_measures)
                
                # Toggle all visibility button
                op = row.operator("mrm.toggle_all_visibility", text="", 
                                 icon='HIDE_OFF' if all_visible else 'HIDE_ON', 
                                 emboss=False)
                op.kind = 'DISTANCE'
                
                row = inner.row()
                col = row.column()
                col.template_list("MRM_UL_measures", "", obj, "mrm_distance_measures", context.scene, "mrm_active_distance_index", rows=3)
                col2 = row.column(align=True)
                idx = context.scene.mrm_active_distance_index
                idx = idx if (obj and 0 <= idx < len(obj.mrm_distance_measures)) else -1
                col2.operator("mrm.delete_measure", text="", icon='X').kind = 'DISTANCE'
                col2.operator("mrm.delete_all_in_list", text="", icon='TRASH').kind = 'DISTANCE'

        # ANGLE block - with large create button
        box = layout.box()
        header = box.row()
        header.prop(ss, "expand_angle", text="🔺 Angle", emboss=False, icon='TRIA_DOWN' if ss.expand_angle else 'TRIA_RIGHT')
        
        if ss.expand_angle:
            # Large create button
            create_row = box.row()
            create_row.operator("mrm.create_angle", text="Create Angle", icon='DRIVER_ROTATIONAL_DIFFERENCE')
            
            inner = box.column(align=True)
            inner.prop(ss.angle, "color", text="Color")
            inner.prop(ss.angle, "thickness", text="Thickness")
            inner.prop(ss.angle, "normal_offset", text="Normal offset")
            inner.prop(ss.angle, "text_offset", text="Text offset")
            inner.prop(ss.angle, "font_size", text="Font size")
            inner.separator()
            
            # Angle measures header with toggle all button
            if obj is None:
                inner.label(text="No active object")
            else:
                row = inner.row()
                row.label(text="Angle measures:")
                
                # Check if all angle measures are visible
                all_visible = True
                if obj.mrm_angle_measures:
                    all_visible = all(m.show for m in obj.mrm_angle_measures)
                
                # Toggle all visibility button
                op = row.operator("mrm.toggle_all_visibility", text="", 
                                 icon='HIDE_OFF' if all_visible else 'HIDE_ON', 
                                 emboss=False)
                op.kind = 'ANGLE'
                
                row = inner.row()
                col = row.column()
                col.template_list("MRM_UL_measures", "", obj, "mrm_angle_measures", context.scene, "mrm_active_angle_index", rows=3)
                col2 = row.column(align=True)
                idx = context.scene.mrm_active_angle_index
                idx = idx if (obj and 0 <= idx < len(obj.mrm_angle_measures)) else -1
                col2.operator("mrm.delete_measure", text="", icon='X').kind = 'ANGLE'
                col2.operator("mrm.delete_all_in_list", text="", icon='TRASH').kind = 'ANGLE'

        # DIAMETER block - with large create button
        box = layout.box()
        header = box.row()
        header.prop(ss, "expand_diameter", text="⌀ Diameter", emboss=False, icon='TRIA_DOWN' if ss.expand_diameter else 'TRIA_RIGHT')
        
        if ss.expand_diameter:
            # Large create button
            create_row = box.row()
            create_row.operator("mrm.create_diameter", text="Create Diameter", icon='MESH_CIRCLE')
            
            inner = box.column(align=True)
            inner.prop(ss.diameter, "color", text="Color")
            inner.prop(ss.diameter, "thickness", text="Thickness")
            inner.prop(ss.diameter, "normal_offset", text="Normal offset")
            inner.prop(ss.diameter, "text_offset", text="Text offset")
            inner.prop(ss.diameter, "font_size", text="Font size")
            inner.separator()
            
            # Diameter measures header with toggle all button
            if obj is None:
                inner.label(text="No active object")
            else:
                row = inner.row()
                row.label(text="Diameter measures:")
                
                # Check if all diameter measures are visible
                all_visible = True
                if obj.mrm_diameter_measures:
                    all_visible = all(m.show for m in obj.mrm_diameter_measures)
                
                # Toggle all visibility button
                op = row.operator("mrm.toggle_all_visibility", text="", 
                                 icon='HIDE_OFF' if all_visible else 'HIDE_ON', 
                                 emboss=False)
                op.kind = 'DIAMETER'
                
                row = inner.row()
                col = row.column()
                col.template_list("MRM_UL_measures", "", obj, "mrm_diameter_measures", context.scene, "mrm_active_diameter_index", rows=3)
                col2 = row.column(align=True)
                idx = context.scene.mrm_active_diameter_index
                idx = idx if (obj and 0 <= idx < len(obj.mrm_diameter_measures)) else -1
                col2.operator("mrm.delete_measure", text="", icon='X').kind = 'DIAMETER'
                col2.operator("mrm.delete_all_in_list", text="", icon='TRASH').kind = 'DIAMETER'

# ---------- Registration ----------
classes = [
    MRM_Measure,
    MRM_DistanceSettings,
    MRM_AngleSettings,
    MRM_DiameterSettings,
    MRM_Settings,
    MRM_OT_create_distance,
    MRM_OT_create_angle,
    MRM_OT_create_diameter,
    MRM_OT_clear_all,
    MRM_OT_clear_type,
    MRM_OT_delete_measure,
    MRM_OT_toggle_measure_visibility,
    MRM_OT_toggle_all_visibility,
    MRM_OT_toggle_diameter_circle,
    MRM_OT_delete_all_in_list,
    MRM_OT_toggle_draw,
    MRM_UL_measures,
    MRM_PT_panel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # per-object measure collections
    bpy.types.Object.mrm_distance_measures = CollectionProperty(type=MRM_Measure)
    bpy.types.Object.mrm_angle_measures = CollectionProperty(type=MRM_Measure)
    bpy.types.Object.mrm_diameter_measures = CollectionProperty(type=MRM_Measure)
    
    # scene-level settings and indices
    bpy.types.Scene.mrm_settings = PointerProperty(type=MRM_Settings)
    bpy.types.Scene.mrm_active_distance_index = IntProperty(default=-1)
    bpy.types.Scene.mrm_active_angle_index = IntProperty(default=-1)
    bpy.types.Scene.mrm_active_diameter_index = IntProperty(default=-1)
    
    bpy.types.WindowManager.mrm_draw_running = BoolProperty(default=False)

    print("MRM Metrics v4.5 improved UI + Diameter + Normal Offset registered")

def unregister():
    # remove draw handler if active
    try:
        if getattr(bpy.context.window_manager, "mrm_draw_running", False):
            bpy.ops.mrm.toggle_draw()
    except Exception:
        pass
    
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    
    # remove props
    try:
        del bpy.types.Object.mrm_distance_measures
        del bpy.types.Object.mrm_angle_measures
        del bpy.types.Object.mrm_diameter_measures
    except Exception:
        pass
    
    try:
        del bpy.types.Scene.mrm_settings
        del bpy.types.Scene.mrm_active_distance_index
        del bpy.types.Scene.mrm_active_angle_index
        del bpy.types.Scene.mrm_active_diameter_index
    except Exception:
        pass
    
    try:
        del bpy.types.WindowManager.mrm_draw_running
    except Exception:
        pass

    print("MRM Metrics v4.5 improved UI + Diameter + Normal Offset unregistered")

if __name__ == "__main__":
    register()