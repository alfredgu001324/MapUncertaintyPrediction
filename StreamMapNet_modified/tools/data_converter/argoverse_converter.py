from functools import partial
from multiprocessing import Pool
import multiprocessing
from random import sample
import time
import mmcv
import logging
from pathlib import Path
from os import path as osp
import os
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from tqdm import tqdm
import argparse
from IPython import embed

CAM_NAMES = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left',
    # 'stereo_front_left', 'stereo_front_right',
    ]

FAIL_LOGS = [
    '01bb304d-7bd8-35f8-bbef-7086b688e35e',
    '453e5558-6363-38e3-bf9b-42b5ba0a6f1d',
    '75e8adad-50a6-3245-8726-5e612db3d165',
    '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
    'af170aac-8465-3d7b-82c5-64147e94af7d',
    '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
]

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        help='specify the root path of dataset')
    parser.add_argument(
        '--newsplit',
        action='store_true')
    parser.add_argument(
        '--nproc',
        type=int,
        default=64,
        required=False,
        help='workers to process data')
    args = parser.parse_args()
    return args

def create_av2_infos_mp(root_path,
                        info_prefix,
                        log_ids,
                        split,
                        dest_path=None,
                        num_multithread=64, 
                        newsplit=False):
    """Create info file of av2 dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        dest_path (str): Path to store generated file, default to root_path
        split (str): Split of the data.
            Default: 'train'
    """
    
    if dest_path is None:
        dest_path = root_path

    for i in FAIL_LOGS:
        if i in log_ids:
            log_ids.remove(i)
    # dataloader by original split
    train_loader = AV2SensorDataLoader(Path(osp.join(root_path, 'train')), 
        Path(osp.join(root_path, 'train')))
    val_loader = AV2SensorDataLoader(Path(osp.join(root_path, 'val')), 
        Path(osp.join(root_path, 'val')))
    test_loader = AV2SensorDataLoader(Path(osp.join(root_path, 'test')), 
        Path(osp.join(root_path, 'test')))
    loaders = [train_loader, val_loader, test_loader]

    print('collecting samples...')
    start_time = time.time()
    print('num cpu:', multiprocessing.cpu_count())
    print(f'using {num_multithread} threads')

    # ignore warning from av2.utils.synchronization_database
    sdb_logger = logging.getLogger('av2.utils.synchronization_database')
    prev_level = sdb_logger.level
    sdb_logger.setLevel(logging.CRITICAL)

    pool = Pool(num_multithread)
    fn = partial(get_data_from_logid, loaders=loaders, data_root=root_path)
    
    rt = pool.map_async(fn, log_ids)
    pool.close()
    pool.join()
    results = rt.get()

    samples = []
    discarded = 0
    sample_idx = 0
    for _samples, _discarded in results:
        for i in range(len(_samples)):
            _samples[i]['sample_idx'] = sample_idx
            sample_idx += 1
        samples.extend(_samples)
        discarded += _discarded
    
    sdb_logger.setLevel(prev_level)
    print(f'{len(samples)} available samples, {discarded} samples discarded')

    id2map = {}
    for log_id in log_ids:
        for i in range(3):
            if log_id in loaders[i]._sdb.get_valid_logs():
                loader = loaders[i]
        
        map_path_dir = osp.join(loader._data_dir, log_id, 'map')
        map_fname = str(list(Path(map_path_dir).glob("log_map_archive_*.json"))[0])
        map_fname = osp.join(map_path_dir, map_fname)
        id2map[log_id] = map_fname

    print('collected in {:.1f}s'.format(time.time() - start_time))
    infos = dict(samples=samples, id2map=id2map)

    if newsplit:
        info_path = osp.join(dest_path,
                                    '{}_map_infos_{}_newsplit.pkl'.format(info_prefix, split))
    else:
        info_path = osp.join(dest_path,
                                    '{}_map_infos_{}.pkl'.format(info_prefix, split))
    print(f'saving results to {info_path}')
    mmcv.dump(infos, info_path)

def get_data_from_logid(log_id, loaders, data_root):
    samples = []
    discarded = 0

    # find corresponding loader
    for i in range(3):
        if log_id in loaders[i]._sdb.get_valid_logs():
            loader = loaders[i]
    
    # use lidar timestamps to query all sensors.
    # the frequency is 10Hz
    cam_timestamps = loader._sdb.per_log_lidar_timestamps_index[log_id]
    prev = -1
    for ts in cam_timestamps:
        cam_ring_fpath = [loader.get_closest_img_fpath(
                log_id, cam_name, ts
            ) for cam_name in CAM_NAMES]
        lidar_fpath = loader.get_closest_lidar_fpath(log_id, ts)

        # if bad sensor synchronization, discard the sample
        if None in cam_ring_fpath or lidar_fpath is None:
            discarded += 1
            continue

        cams = {}
        for i, cam_name in enumerate(CAM_NAMES):
            pinhole_cam = loader.get_log_pinhole_camera(log_id, cam_name)
            cams[cam_name] = dict(
                img_fpath=str(cam_ring_fpath[i]),
                intrinsics=pinhole_cam.intrinsics.K,
                extrinsics=pinhole_cam.extrinsics,
            )
        
        city_SE3_ego = loader.get_city_SE3_ego(log_id, int(ts))
        e2g_translation = city_SE3_ego.translation
        e2g_rotation = city_SE3_ego.rotation
        
        samples.append(dict(
            e2g_translation=e2g_translation,
            e2g_rotation=e2g_rotation,
            cams=cams, 
            lidar_fpath=str(lidar_fpath),
            prev=prev,
            # map_fpath=map_fname,
            token=str(ts),
            log_id=log_id,
            scene_name=log_id))
        
        prev = str(ts)

    return samples, discarded


if __name__ == '__main__':
    args = parse_args()
    with open('tools/data_converter/av2_train_split.txt') as f:
        train_split = [s.strip() for s in f.readlines()]
    with open('tools/data_converter/av2_val_split.txt') as f:
        val_split = [s.strip() for s in f.readlines()]
    
    test_split = None
    if not args.newsplit:
        train_split = os.listdir(osp.join(args.data_root, 'train'))
        val_split = os.listdir(osp.join(args.data_root, 'val'))
        test_split = os.listdir(osp.join(args.data_root, 'test'))

    create_av2_infos_mp(
        root_path=args.data_root,
        split='train',
        log_ids=train_split,
        info_prefix='av2',
        dest_path=args.data_root,
        newsplit=args.newsplit)
    
    create_av2_infos_mp(
        root_path=args.data_root,
        split='val',
        log_ids=val_split,
        info_prefix='av2',
        dest_path=args.data_root,
        newsplit=args.newsplit)

    if test_split:
        create_av2_infos_mp(
            root_path=args.data_root,
            split='test',
            log_ids=test_split,
            info_prefix='av2',
            dest_path=args.data_root,)