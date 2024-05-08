python eval.py \
  --root ../trj_data/{maptr,maptrv2,maptrv2_cent,stream} \
  --split {mini_val,val} \
  --method {base,unc} \
  --batch_size 32 \
  --ckpt_path /path/to/your_checkpoint.ckpt
