export PYTHONPATH="${PYTHONPATH}:/MapUncertaintyPrediction/MapTRv2_modified"
python tools/train.py /MapUncertaintyPrediction/MapTRv2_modified/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py --deterministic --no-validate
