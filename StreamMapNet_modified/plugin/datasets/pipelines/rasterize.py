import numpy as np
from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString, Polygon
from shapely import affinity
import cv2
from PIL import Image, ImageDraw
from numpy.typing import NDArray
from typing import List, Tuple, Union, Dict

@PIPELINES.register_module(force=True)
class RasterizeMap(object):
    """Generate rasterized semantic map and put into 
    `semantic_mask` key.

    Args:
        roi_size (tuple or list): bev range
        canvas_size (tuple or list): bev feature size
        thickness (int): thickness of rasterized lines
        coords_dim (int): dimension of point coordinates
    """

    def __init__(self, 
                 roi_size: Union[Tuple, List], 
                 canvas_size: Union[Tuple, List], 
                 thickness: int, 
                 coords_dim: int):

        self.roi_size = roi_size
        self.canvas_size = canvas_size
        self.scale_x = self.canvas_size[0] / self.roi_size[0]
        self.scale_y = self.canvas_size[1] / self.roi_size[1]
        self.thickness = thickness
        self.coords_dim = coords_dim
    
    def line_ego_to_mask(self, 
                         line_ego: LineString, 
                         mask: NDArray, 
                         color: int=1, 
                         thickness: int=3) -> None:
        ''' Rasterize a single line to mask.
        
        Args:
            line_ego (LineString): line
            mask (array): semantic mask to paint on
            color (int): positive label, default: 1
            thickness (int): thickness of rasterized lines, default: 3
        '''

        trans_x = self.canvas_size[0] / 2
        trans_y = self.canvas_size[1] / 2
        line_ego = affinity.scale(line_ego, self.scale_x, self.scale_y, origin=(0, 0))
        line_ego = affinity.affine_transform(line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
        
        coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
        coords = coords.reshape((-1, 2))
        assert len(coords) >= 2
        
        cv2.polylines(mask, np.int32([coords]), False, color=color, thickness=thickness)
    
    def polygons_ego_to_mask(self, 
                             polygons: List[Polygon], 
                             color: int=1) -> NDArray:
        ''' Rasterize a polygon to mask.
        
        Args:
            polygons (list): list of polygons
            color (int): positive label, default: 1
        
        Returns:
            mask (array): mask with rasterize polygons
        '''

        mask = Image.new("L", size=(self.canvas_size[0], self.canvas_size[1]), color=0) 
        # Image lib api expect size as (w, h)
        trans_x = self.canvas_size[0] / 2
        trans_y = self.canvas_size[1] / 2
        for polygon in polygons:
            polygon = affinity.scale(polygon, self.scale_x, self.scale_y, origin=(0, 0))
            polygon = affinity.affine_transform(polygon, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            ext = np.array(polygon.exterior.coords)[:, :2]
            vert_list = [(x, y) for x, y in ext]

            ImageDraw.Draw(mask).polygon(vert_list, outline=1, fill=color)

        return np.array(mask, np.uint8)
    
    def get_semantic_mask(self, map_geoms: Dict) -> NDArray:
        ''' Rasterize all map geometries to semantic mask.
        
        Args:
            map_geoms (dict): map geoms by class
        
        Returns:
            semantic_mask (array): semantic mask
        '''

        num_classes = len(map_geoms)
        semantic_mask = np.zeros((num_classes, self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)

        for label, geom_list in map_geoms.items():
            if len(geom_list) == 0:
                continue
            if geom_list[0].geom_type == 'LineString':
                for line in geom_list:
                    self.line_ego_to_mask(line, semantic_mask[label], color=1,
                        thickness=self.thickness)
            elif geom_list[0].geom_type == 'Polygon':
                # drivable area 
                polygons = []
                for polygon in geom_list:
                    polygons.append(polygon)
                semantic_mask[label] = self.polygons_ego_to_mask(polygons, color=1)
            else:
                raise ValueError('map geoms must be either LineString or Polygon!')
        
        return np.ascontiguousarray(semantic_mask)
    
    def __call__(self, input_dict: Dict) -> Dict:
        map_geoms = input_dict['map_geoms'] # {0: List[ped_crossing: LineString], 1: ...}

        semantic_mask = self.get_semantic_mask(map_geoms)
        input_dict['semantic_mask'] = semantic_mask # (num_class, canvas_size[1], canvas_size[0])
        return input_dict
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(roi_size={self.roi_size}, '
        repr_str += f'canvas_size={self.canvas_size}), '
        repr_str += f'thickness={self.thickness}), ' 
        repr_str += f'coords_dim={self.coords_dim})'

        return repr_str
