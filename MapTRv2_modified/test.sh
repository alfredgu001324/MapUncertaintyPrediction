export PYTHONPATH="${PYTHONPATH}:/MapUncertaintyPrediction/MapTRv2_modified"
python tools/test.py /MapUncertaintyPrediction/MapTRv2_modified/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py /MapUncertaintyPrediction/MapTRv2_modified/work_dirs/maptrv2_nusc_r50_24ep/YOURCHECKPOINT.pth --eval chamfer

# python tools/test.py /MapUncertaintyPrediction/MapTRv2_modified/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_w_centerline.py /MapUncertaintyPrediction/MapTRv2_modified/work_dirs/maptrv2_nusc_r50_24ep_w_centerline/YOUR_CHECKPOINT.pth --eval chamfer
