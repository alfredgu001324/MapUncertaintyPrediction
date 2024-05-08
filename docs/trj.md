## Trajectory Prediction Models Training

### HiVT 

**Setup**

Please follow [HiVT](https://github.com/ZikangZhou/HiVT) setup guide to set up environment.

**Training**

```
cd HiVT_modified

python train.py \
  --root ../trj_data/{maptr,maptrv2,maptrv2_cent,stream} \
  --method {base,unc} \
  --embed_dim 128
```

For training MapTRv2 Centerline, add an `--centerline` argument. 

**Testing**

```
cd HiVT_modified

python eval.py \
  --root ../trj_data/{maptr,maptrv2,maptrv2_cent,stream} \
  --split {mini_val,val} \
  --method {base,unc} \
  --batch_size 32 \
  --ckpt_path /path/to/your_checkpoint.ckpt
```

For evaluating MapTRv2 Centerline, add an `--centerline` argument. 
