python3 inference.py --model_path ./weights/model/v1-5-pruned-emaonly.ckpt \
--tokenizer_dir ./weights/tokenizer \
--device mps \
--img_size 512 \
--prompt "a sks cat walking on yellow leaves in the autumn forest, big round eyes, high resolution, 4K, highly detailed" \
--n_samples 1 \
--lora_ckpt "./weights/model/stable_diffusion_lora_epoch_75.ckpt"