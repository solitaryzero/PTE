CUDA_VISIBLE_DEVICES=6 python ./src/math/eval_gsm8k.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/gsm8k \
    --tuned_model_path ./models/Qwen2.5-1.5B_tuned/gsm8k_aug_var_s1 \
    --save_result_path ./output/Qwen2.5-1.5B_tuned/gsm8k_aug_var_s1 \
    --bf16 \
    --seed 42