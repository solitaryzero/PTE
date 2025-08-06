CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ./scripts/Llama3.2-3B/accelerate_config.yaml \
    ./src/math/train_latent_deepspeed.py \
    --base_model ./models/Llama-3.2-3B \
    --data_path ./data/gsm8k \
    --save_model_path ./models/Llama-3.2-3B_latent/gsm8k_multihot \
    --save_result_path ./output/Llama-3.2-3B_latent/gsm8k_multihot \
    --bf16 \
    --do_eval \
    --learning_rate 1e-5 \
    --train_batch_size 1 \
    --epoch 6 \
    --logging_steps 200 \
    --int_precision 20 \
    --frac_precision 10 \
    --seed 42