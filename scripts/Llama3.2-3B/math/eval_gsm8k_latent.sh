CUDA_VISIBLE_DEVICES=7 python ./src/math/eval_latent.py \
    --base_model ./models/Llama-3.2-3B \
    --data_path ./data/gsm8k \
    --tuned_model_path ./models/Llama-3.2-3B_latent/gsm8k_multihot \
    --save_result_path ./output/Llama-3.2-3B_latent/gsm8k_multihot \
    --bf16 \
    --int_precision 20 \
    --frac_precision 10 \
    --seed 42