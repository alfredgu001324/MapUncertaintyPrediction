export PYTHONPATH="${PYTHONPATH}:/MapUncertaintyPrediction/StreamMapNet_modified"
python tools/train.py plugin/configs/nusc_newsplit_480_60x30_24e.py --no-validate --deterministic 
