python3 inference.py --model_path ./weights/model/v1-5-pruned-emaonly.ckpt \
--tokenizer_dir ./weights/tokenizer \
--device mps \
--img_size 512 \
--prompt "a cat in the snow, high quality, high detail, ultra realistic, 8K, Canon EOS, f/1.4, ISO 400, shutter speed 1/250" \
--n_samples 1