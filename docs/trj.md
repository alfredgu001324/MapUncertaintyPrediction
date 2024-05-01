## Trajectory Prediction Models Training

### HiVT 

**Setup**

Please follow [HiVT](https://github.com/ZikangZhou/HiVT) setup guide to set up environment.

**Training**

```
python train.py --root ./trj_data/{maptr, maptrv2, maptrv2_cent, stream} --embed_dim 128
```
**Testing**

```
python eval.py --root ./trj_data/{maptr, maptrv2, maptrv2_cent, stream} --batch_size 32 --ckpt_path /path/to/your_checkpoint.ckpt

```
