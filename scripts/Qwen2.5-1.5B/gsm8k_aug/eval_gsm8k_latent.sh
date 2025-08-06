CUDA_VISIBLE_DEVICES=7 python ./src/math/eval_latent.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/gsm8k \
    --tuned_model_path ./models/Qwen2.5-1.5B_latent/gsm8k_aug_multihot \
    --save_result_path ./output/Qwen2.5-1.5B_latent/gsm8k_aug_multihot \
    --bf16 \
    --int_precision 10 \
    --frac_precision 5 \
    --seed 42