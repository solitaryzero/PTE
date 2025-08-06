LATENT_TYPE="multihot"

CUDA_VISIBLE_DEVICES=7 python ./src/stepwise/train_stepwise.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/gsm8k_aug \
    --latent_projector_path ./models/Qwen2.5-1.5B_probe \
    --save_model_path ./models/Qwen2.5-1.5B_stepwise/gsm8k_aug_${LATENT_TYPE} \
    --save_result_path ./output/Qwen2.5-1.5B_stepwise/gsm8k_aug_${LATENT_TYPE} \
    --bf16 \
    --do_eval \
    --learning_rate 1e-5 \
    --grad_clip 1 \
    --train_batch_size 1 \
    --epoch 1 \
    --logging_steps 100 \
    --max_new_tokens 2048 \
    --latent_type ${LATENT_TYPE} \
    --int_precision 10 \
    --frac_precision 5 \
    --intermediate_dim 512 \
    --max_latent_tokens 20 \
    --latent_step_n_tokens 2 \
    --seed 42