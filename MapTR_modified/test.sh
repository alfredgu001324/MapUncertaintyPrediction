export PYTHONPATH="${PYTHONPATH}:/MapUncertaintyPrediction/MapTR_modified"

python tools/test.py /MapUncertaintyPrediction/MapTR_modified/projects/configs/maptr/maptr_tiny_r50_24e.py /MapUncertaintyPrediction/MapTR_modified/work_dirs/maptr_tiny_r50_24e/YOURCHECKPOINT.pth --eval chamfer
