from .base_dataset import BaseMapDataset
from .map_utils.av2map_extractor import AV2MapExtractor
from mmdet.datasets import DATASETS
import numpy as np
from .visualize.renderer import Renderer
from time import time
import mmcv
from IPython import embed

@DATASETS.register_module()
class AV2Dataset(BaseMapDataset):
    """Argoverse2 map dataset class.

    Args:
        ann_file (str): annotation file path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        eval_config (Config): evaluation config
        meta (dict): meta information
        pipeline (Config): data processing pipeline config,
        interval (int): annotation load interval
        work_dir (str): path to work dir
        test_mode (bool): whether in test mode
    """

    def __init__(self, **kwargs,):
        super().__init__(**kwargs)
        self.map_extractor = AV2MapExtractor(self.roi_size, self.id2map)
        self.renderer = Renderer(self.cat2id, self.roi_size, 'av2')
    
    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        if self.seq_split_num == -1:
            self.flag = np.arange(len(self.samples))
            return
        
        res = []
        assert self.seq_split_num == 1

        all_log_ids = []
        for s in self.samples:
            if s['scene_name'] not in all_log_ids:
                all_log_ids.append(s['scene_name'])
        
        for idx in range(len(self.samples)):
            res.append(all_log_ids.index(self.samples[idx]['scene_name']))

        self.flag = np.array(res, dtype=np.int64)
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        
        start_time = time()
        ann = mmcv.load(ann_file)
        self.id2map = ann['id2map']
        samples = ann['samples']
        samples = samples[::self.interval]
        
        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
        self.samples = samples

    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        log_id = sample['log_id']
        
        map_geoms = self.map_extractor.get_map_geom(log_id, sample['e2g_translation'], 
                sample['e2g_rotation'])

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v
        
        ego2img_rts = []
        for c in sample['cams'].values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)

        input_dict = {
            'token': sample['token'],
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            # extrinsics are 4x4 tranform matrix, NOTE: **ego2cam**
            'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            'ego2global_translation': sample['e2g_translation'], 
            'ego2global_rotation': sample['e2g_rotation'],
            'scene_name': sample['scene_name']
        }

        return input_dict