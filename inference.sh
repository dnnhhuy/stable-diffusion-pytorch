python3 inference.py --model_path ./weights/model/v1-5-pruned-emaonly.ckpt \
--tokenizer_dir ./weights/tokenizer \
--device mps \
--img_size 512 \
--prompt "A cat" \
--n_samples 1000 \
--batch_size 2 \
--sampler ddim \
--do_cfg