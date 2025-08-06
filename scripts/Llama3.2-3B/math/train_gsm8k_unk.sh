CUDA_VISIBLE_DEVICES=3,5 accelerate launch --config_file ./scripts/Llama3.2-3B/accelerate_config.yaml \
    ./src/math/train_gsm8k_deepspeed.py \
    --base_model ./models/Llama-3.2-3B \
    --data_path ./data/gsm8k \
    --save_model_path ./models/Llama-3.2-3B_tuned/gsm8k_unk_s1 \
    --save_result_path ./output/Llama-3.2-3B_tuned/gsm8k_unk_s1 \
    --bf16 \
    --learning_rate 1e-5 \
    --train_batch_size 1 \
    --epoch 3 \
    --logging_steps 200 \
    --replace_numbers 'yes' \
    --use_unk \
    --max_new_tokens 2048 \
    --seed 42

echo "=== Stage 2 ==="

CUDA_VISIBLE_DEVICES=3,5 accelerate launch --config_file ./scripts/Llama3.2-3B/accelerate_config.yaml \
    ./src/math/train_gsm8k_deepspeed.py \
    --base_model ./models/Llama-3.2-3B \
    --data_path ./data/gsm8k \
    --second_stage \
    --previous_model ./models/Llama-3.2-3B_tuned/gsm8k_unk_s1 \
    --save_model_path ./models/Llama-3.2-3B_tuned/gsm8k_unk_s2 \
    --save_result_path ./output/Llama-3.2-3B_tuned/gsm8k_unk_s2 \
    --bf16 \
    --do_eval \
    --learning_rate 1e-5 \
    --train_batch_size 1 \
    --epoch 3 \
    --logging_steps 200 \
    --max_new_tokens 2048 \
    --seed 42