## Map Estimation Environment Setup

**Create and Activate Conda Environment**
```
conda create -n map python=3.8 -y
conda activate map
```

**Install PyTorch and Related Packages**
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**Additional Requirements**
```
pip install -r requirements.txt
```

**Setup MMDetection3D**
```
cd /path/to/MapTR_v2_modified/mmdetection3d
python setup.py develop
# Resolve CUDA_HOME not found error if it occurs
# conda install -c conda-forge cudatoolkit-dev
```

**Install Geometric Kernel Attention Module**
```
cd /path/to/MapTR_v2_modified/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install
```

## Trajectory Prediction Environment Setup

**Create and Activate Conda Environment**
```
conda create -n predict python=3.8 -y
conda activate predict
```

**Install Trajdata**
```
pip install trajdata==1.3.1
pip install "trajdata[nusc]"
```

