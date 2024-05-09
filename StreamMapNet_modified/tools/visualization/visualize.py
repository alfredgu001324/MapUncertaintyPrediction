import argparse
import mmcv
from mmcv import Config
import os
from mmdet3d.datasets import build_dataset, build_dataloader
from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('idx', type=int,
        help='which scene to visualize')
    parser.add_argument('--result', 
        default=None,
        help='prediction result to visualize'
        'If submission file is not provided, only gt will be visualized')
    parser.add_argument('--thr', 
        type=float,
        default=0.4,
        help='score threshold to filter predictions')
    parser.add_argument(
        '--out-dir', 
        default='demo',
        help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args

def import_plugin(cfg):
    '''
        import modules from plguin/xx, registry will be update
    '''

    import sys
    sys.path.append(os.path.abspath('.'))    
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            
            def import_path(plugin_dir):
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            plugin_dirs = cfg.plugin_dir
            if not isinstance(plugin_dirs, list):
                plugin_dirs = [plugin_dirs,]
            for plugin_dir in plugin_dirs:
                import_path(plugin_dir)

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    # build the dataset
    dataset = build_dataset(cfg.eval_config)

    # ann_file = mmcv.load('datasets/nuScenes/nuscenes_map_infos_val.pkl')
    scene_name2idx = {}
    for idx, sample in enumerate(dataset.samples):
        scene = sample['scene_name']
        if scene not in scene_name2idx:
            scene_name2idx[scene] = []
        scene_name2idx[scene].append(idx)

    scene_name = sorted(list(scene_name2idx.keys()))[args.idx]
    print(scene_name)
    scene_dir = os.path.join(args.out_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    start_idx = scene_name2idx[scene_name][0]
    results = mmcv.load(args.result)
    for idx in mmcv.track_iter_progress(scene_name2idx[scene_name]):
        
        out_dir = os.path.join(scene_dir, str(idx - start_idx + 1))
        gt_dir = os.path.join(out_dir, 'gt')
        pred_dir = os.path.join(out_dir, 'pred')

        if args.result is not None:
            os.makedirs(pred_dir, exist_ok=True)
            dataset.show_result(
                    submission=results, 
                    idx=idx, 
                    score_thr=args.thr, 
                    out_dir=pred_dir
                )
            
        os.makedirs(gt_dir, exist_ok=True)
        dataset.show_gt(idx, gt_dir)


if __name__ == '__main__':
    main()