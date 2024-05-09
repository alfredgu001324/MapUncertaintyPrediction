from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path
from shapely.geometry import LineString, box, Polygon
from shapely import ops
import numpy as np
from .utils import split_collections, get_drivable_area_contour, \
        get_ped_crossing_contour, remove_repeated_lines, transform_from, \
        connect_lines, remove_boundary_dividers
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union

class AV2MapExtractor(object):
    """Argoverse 2 map ground-truth extractor.

    Args:
        roi_size (tuple or list): bev range
        id2map (dict): log id to map json path
    """
    def __init__(self, roi_size: Union[Tuple, List], id2map: Dict) -> None:
        self.roi_size = roi_size
        self.id2map = {}

        for log_id, path in id2map.items():
            self.id2map[log_id] = ArgoverseStaticMap.from_json(Path(path))
        
    def get_map_geom(self, 
                     log_id: str, 
                     e2g_translation: NDArray, 
                     e2g_rotation: NDArray, 
                     polygon_ped=True) -> Dict[str, List[Union[LineString, Polygon]]]:
        ''' Extract geometries given `log_id` and ego pose.
        
        Args:
            log_id (str): log id
            e2g_translation (array): ego2global translation, shape (3,)
            e2g_rotation (array): ego2global rotation matrix, shape (3, 3)
            polygon_ped: if True, organize each ped crossing as closed polylines. \
                Otherwise organize each ped crossing as two parallel polylines. \
                Default: True
        
        Returns:
            geometries (Dict): extracted geometries by category.
        '''

        avm = self.id2map[log_id]
        
        g2e_translation = e2g_rotation.T.dot(-e2g_translation)
        g2e_rotation = e2g_rotation.T

        roi_x, roi_y = self.roi_size[:2]
        local_patch = box(-roi_x / 2, -roi_y / 2, roi_x / 2, roi_y / 2)

        all_dividers = []
        # for every lane segment, extract its right/left boundaries as road dividers
        for _, ls in avm.vector_lane_segments.items():
            # right divider
            right_xyz = ls.right_lane_boundary.xyz
            right_mark_type = ls.right_mark_type
            right_ego_xyz = transform_from(right_xyz, g2e_translation, g2e_rotation)

            right_line = LineString(right_ego_xyz)
            right_line_local = right_line.intersection(local_patch)

            if not right_line_local.is_empty and not right_mark_type in ['NONE', 'UNKNOWN']:
                all_dividers += split_collections(right_line_local)
                
            # left divider
            left_xyz = ls.left_lane_boundary.xyz
            left_mark_type = ls.left_mark_type
            left_ego_xyz = transform_from(left_xyz, g2e_translation, g2e_rotation)

            left_line = LineString(left_ego_xyz)
            left_line_local = left_line.intersection(local_patch)

            if not left_line_local.is_empty and not left_mark_type in ['NONE', 'UNKNOWN']:
                all_dividers += split_collections(left_line_local)
        
        # remove repeated dividers since each divider in argoverse2 is mentioned twice
        # by both left lane and right lane
        all_dividers = remove_repeated_lines(all_dividers)
        
        ped_crossings = [] 
        for _, pc in avm.vector_pedestrian_crossings.items():
            edge1_xyz = pc.edge1.xyz
            edge2_xyz = pc.edge2.xyz
            ego1_xyz = transform_from(edge1_xyz, g2e_translation, g2e_rotation)
            ego2_xyz = transform_from(edge2_xyz, g2e_translation, g2e_rotation)

            # if True, organize each ped crossing as closed polylines. 
            if polygon_ped:
                vertices = np.concatenate([ego1_xyz, ego2_xyz[::-1, :]])
                p = Polygon(vertices)
                line = get_ped_crossing_contour(p, local_patch)
                if line is not None:
                    ped_crossings.append(line)

            # Otherwise organize each ped crossing as two parallel polylines.
            else:
                line1 = LineString(ego1_xyz)
                line2 = LineString(ego2_xyz)
                line1_local = line1.intersection(local_patch)
                line2_local = line2.intersection(local_patch)

                # take the whole ped cross if all two edges are in roi range
                if not line1_local.is_empty and not line2_local.is_empty:
                    ped_crossings.append(line1_local)
                    ped_crossings.append(line2_local)

        drivable_areas = []
        for _, da in avm.vector_drivable_areas.items():
            polygon_xyz = da.xyz
            ego_xyz = transform_from(polygon_xyz, g2e_translation, g2e_rotation)
            polygon = Polygon(ego_xyz)
            polygon_local = polygon.intersection(local_patch)

            drivable_areas.append(polygon_local)

        # union all drivable areas polygon
        drivable_areas = ops.unary_union(drivable_areas)
        drivable_areas = split_collections(drivable_areas)

        # boundaries are defined as the contour of drivable areas
        boundaries = get_drivable_area_contour(drivable_areas, self.roi_size)

        # some dividers overlaps with boundaries in argoverse2 dataset
        # we need to remove these dividers
        all_dividers = remove_boundary_dividers(all_dividers, boundaries)

        # some dividers are split into multiple small parts
        # we connect these lines
        all_dividers = connect_lines(all_dividers)

        return dict(
            divider=all_dividers, # List[LineString]
            ped_crossing=ped_crossings, # List[LineString]
            boundary=boundaries, # List[LineString]
            drivable_area=drivable_areas, # List[Polygon],
        )
