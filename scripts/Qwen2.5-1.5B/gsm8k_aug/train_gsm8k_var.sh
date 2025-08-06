CUDA_VISIBLE_DEVICES=6 python ./src/math/train_gsm8k_aug.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/gsm8k \
    --aug_data_path ./data/gsm8k_aug \
    --save_model_path ./models/Qwen2.5-1.5B_tuned/gsm8k_aug_var_s1 \
    --save_result_path ./output/Qwen2.5-1.5B_tuned/gsm8k_aug_var_s1 \
    --bf16 \
    --learning_rate 1e-5 \
    --train_batch_size 4 \
    --epoch 1 \
    --logging_steps 200 \
    --replace_numbers 'yes' \
    --max_new_tokens 2048 \
    --report_to none \
    --seed 42

echo "=== Stage 2 ==="

CUDA_VISIBLE_DEVICES=6 python ./src/math/train_gsm8k_aug.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/gsm8k \
    --aug_data_path ./data/gsm8k_aug \
    --second_stage \
    --previous_model ./models/Qwen2.5-1.5B_tuned/gsm8k_aug_var_s1 \
    --save_model_path ./models/Qwen2.5-1.5B_tuned/gsm8k_aug_var_s2 \
    --save_result_path ./output/Qwen2.5-1.5B_tuned/gsm8k_aug_var_s2 \
    --bf16 \
    --do_eval \
    --learning_rate 1e-5 \
    --train_batch_size 4 \
    --epoch 1 \
    --logging_steps 200 \
    --max_new_tokens 2048 \
    --report_to none \
    --seed 42