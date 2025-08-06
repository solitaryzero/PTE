CUDA_VISIBLE_DEVICES=1 python ./src/math/train_gsm8k.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/gsm8k \
    --save_model_path ./models/Qwen2.5-1.5B_tuned/gsm8k_var_s1 \
    --save_result_path ./output/Qwen2.5-1.5B_tuned/gsm8k_var_s1 \
    --bf16 \
    --learning_rate 1e-5 \
    --train_batch_size 4 \
    --epoch 1 \
    --logging_steps 2000 \
    --replace_numbers 'yes' \
    --max_new_tokens 2048 \
    --seed 42

echo "=== Stage 2 ==="

CUDA_VISIBLE_DEVICES=1 python ./src/math/train_gsm8k.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/gsm8k \
    --second_stage \
    --previous_model ./models/Qwen2.5-1.5B_tuned/gsm8k_var_s1 \
    --save_model_path ./models/Qwen2.5-1.5B_tuned/gsm8k_var_s2 \
    --save_result_path ./output/Qwen2.5-1.5B_tuned/gsm8k_var_s2 \
    --bf16 \
    --do_eval \
    --learning_rate 1e-5 \
    --train_batch_size 4 \
    --epoch 1 \
    --logging_steps 2000 \
    --max_new_tokens 2048 \
    --seed 42