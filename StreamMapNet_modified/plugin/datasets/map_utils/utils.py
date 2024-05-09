from shapely.geometry import LineString, box, Polygon, LinearRing
from shapely.geometry.base import BaseGeometry
from shapely import ops
import numpy as np
from scipy.spatial import distance
from typing import List, Optional, Tuple
from numpy.typing import NDArray

def split_collections(geom: BaseGeometry) -> List[Optional[BaseGeometry]]:
    ''' Split Multi-geoms to list and check is valid or is empty.
        
    Args:
        geom (BaseGeometry): geoms to be split or validate.
    
    Returns:
        geometries (List): list of geometries.
    '''
    assert geom.geom_type in ['MultiLineString', 'LineString', 'MultiPolygon', 
        'Polygon', 'GeometryCollection'], f"got geom type {geom.geom_type}"
    if 'Multi' in geom.geom_type:
        outs = []
        for g in geom.geoms:
            if g.is_valid and not g.is_empty:
                outs.append(g)
        return outs
    else:
        if geom.is_valid and not geom.is_empty:
            return [geom,]
        else:
            return []

def get_drivable_area_contour(drivable_areas: List[Polygon], 
                              roi_size: Tuple) -> List[LineString]:
    ''' Extract drivable area contours to get list of boundaries.

    Args:
        drivable_areas (list): list of drivable areas.
        roi_size (tuple): bev range size
    
    Returns:
        boundaries (List): list of boundaries.
    '''
    max_x = roi_size[0] / 2
    max_y = roi_size[1] / 2

    # a bit smaller than roi to avoid unexpected boundaries on edges
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    
    exteriors = []
    interiors = []
    
    for poly in drivable_areas:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)
    
    results = []
    for ext in exteriors:
        # NOTE: we make sure all exteriors are clock-wise
        # such that each boundary's right-hand-side is drivable area
        # and left-hand-side is walk way
        
        if ext.is_ccw:
            ext = LinearRing(list(ext.coords)[::-1])
        lines = ext.intersection(local_patch)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        assert lines.geom_type in ['MultiLineString', 'LineString']
        
        results.extend(split_collections(lines))

    for inter in interiors:
        # NOTE: we make sure all interiors are counter-clock-wise
        if not inter.is_ccw:
            inter = LinearRing(list(inter.coords)[::-1])
        lines = inter.intersection(local_patch)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        assert lines.geom_type in ['MultiLineString', 'LineString']
        
        results.extend(split_collections(lines))

    return results

def get_ped_crossing_contour(polygon: Polygon, 
                             local_patch: box) -> Optional[LineString]:
    ''' Extract ped crossing contours to get a closed polyline.
    Different from `get_drivable_area_contour`, this function ensures a closed polyline.

    Args:
        polygon (Polygon): ped crossing polygon to be extracted.
        local_patch (tuple): local patch params
    
    Returns:
        line (LineString): a closed line
    '''

    ext = polygon.exterior
    if not ext.is_ccw:
        ext = LinearRing(list(ext.coords)[::-1])
    lines = ext.intersection(local_patch)
    if lines.type != 'LineString':
        # remove points in intersection results
        lines = [l for l in lines.geoms if l.geom_type != 'Point']
        lines = ops.linemerge(lines)
        
        # same instance but not connected.
        if lines.type != 'LineString':
            ls = []
            for l in lines.geoms:
                ls.append(np.array(l.coords))
            
            lines = np.concatenate(ls, axis=0)
            lines = LineString(lines)

        start = list(lines.coords[0])
        end = list(lines.coords[-1])
        if not np.allclose(start, end, atol=1e-3):
            new_line = list(lines.coords)
            new_line.append(start)
            lines = LineString(new_line) # make ped cross closed

    if not lines.is_empty:
        return lines
    
    return None

def remove_repeated_lines(lines: List[LineString]) -> List[LineString]:
    ''' Remove repeated dividers since each divider in argoverse2 is mentioned twice
    by both left lane and right lane.

    Args:
        lines (List): list of dividers

    Returns:
        lines (List): list of left dividers
    '''

    new_lines = []
    for line in lines:
        repeated = False
        for l in new_lines:
            length = min(line.length, l.length)
            
            # hand-crafted rule to check overlap
            if line.buffer(0.3).intersection(l.buffer(0.3)).area \
                    > 0.2 * length:
                repeated = True
                break
        
        if not repeated:
            new_lines.append(line)
    
    return new_lines

def remove_boundary_dividers(dividers: List[LineString], 
                             boundaries: List[LineString]) -> List[LineString]:
    ''' Some dividers overlaps with boundaries in argoverse2 dataset so
    we need to remove these dividers.

    Args:
        dividers (list): list of dividers
        boundaries (list): list of boundaries

    Returns:
        left_dividers (list): list of left dividers
    '''

    for idx in range(len(dividers))[::-1]:
        divider = dividers[idx]
        
        for bound in boundaries:
            length = min(divider.length, bound.length)

            # hand-crafted rule to check overlap
            if divider.buffer(0.3).intersection(bound.buffer(0.3)).area \
                    > 0.2 * length:
                # the divider overlaps boundary
                dividers.pop(idx)
                break

    return dividers

def connect_lines(lines: List[LineString]) -> List[LineString]:
    ''' Some dividers are split into multiple small parts
    so we need to connect these lines.

    Args:
        dividers (list): list of dividers
        boundaries (list): list of boundaries

    Returns:
        left_dividers (list): list of left dividers
    '''

    new_lines = []
    eps = 0.1 # threshold to identify continuous lines
    while len(lines) > 1:
        line1 = lines[0]
        merged_flag = False
        for i, line2 in enumerate(lines[1:]):
            # hand-crafted rule
            begin1 = list(line1.coords)[0]
            end1 = list(line1.coords)[-1]
            begin2 = list(line2.coords)[0]
            end2 = list(line2.coords)[-1]

            dist_matrix = distance.cdist([begin1, end1], [begin2, end2])
            if dist_matrix[0, 0] < eps:
                coords = list(line2.coords)[::-1] + list(line1.coords)
            elif dist_matrix[0, 1] < eps:
                coords = list(line2.coords) + list(line1.coords)
            elif dist_matrix[1, 0] < eps:
                coords = list(line1.coords) + list(line2.coords)
            elif dist_matrix[1, 1] < eps:
                coords = list(line1.coords) + list(line2.coords)[::-1]
            else: continue

            new_line = LineString(coords)
            lines.pop(i + 1)
            lines[0] = new_line
            merged_flag = True
            break
        
        if merged_flag: continue

        new_lines.append(line1)
        lines.pop(0)

    if len(lines) == 1:
        new_lines.append(lines[0])

    return new_lines

def transform_from(xyz: NDArray, 
                   translation: NDArray, 
                   rotation: NDArray) -> NDArray:
    ''' Transform points between different coordinate system.

    Args:
        xyz (array): original point coordinates
        translation (array): translation
        rotation (array): rotation matrix

    Returns:
        left_dividers (list): list of left dividers
    '''
    
    new_xyz = xyz @ rotation.T + translation
    return new_xyz
