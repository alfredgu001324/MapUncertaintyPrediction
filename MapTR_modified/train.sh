export PYTHONPATH="${PYTHONPATH}:/MapUncertaintyPrediction/MapTR_modified"
python tools/train.py /MapUncertaintyPrediction/MapTR_modified/projects/configs/maptr/maptr_tiny_r50_24e.py --deterministic --no-validate
