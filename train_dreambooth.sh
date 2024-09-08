#!/bin/bash

python3 train_lora_dreambooth.py \
--device mps \
--model_path ./weights/model/v1-5-pruned-emaonly.ckpt \
--tokenizer_dir ./weights/tokenizer \
--data_dir ./data/dreambooth \
--img_size 512 \
--batch_size 1 \
--lr 5e-6 \
--use_lora false \
--gradient_accumulation_steps 8 \
--gradient_checkpointing false