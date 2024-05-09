export PYTHONPATH="${PYTHONPATH}:/MapUncertaintyPrediction/StreamMapNet_modified"
python tools/test.py plugin/configs/nusc_newsplit_480_60x30_24e.py work_dirs/nusc_newsplit_480_60x30_24e/YOURCHECKPOINT.pth --eval
