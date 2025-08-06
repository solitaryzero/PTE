LATENT_TYPE="multihot"

CUDA_VISIBLE_DEVICES=7 python ./src/stepwise/eval_stepwise.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/gsm8k_aug \
    --tuned_model_path ./models/Qwen2.5-1.5B_stepwise/gsm8k_aug_${LATENT_TYPE} \
    --save_result_path ./output/Qwen2.5-1.5B_stepwise/gsm8k_aug_${LATENT_TYPE} \
    --bf16 \
    --max_new_tokens 2048 \
    --latent_type ${LATENT_TYPE} \
    --seed 42