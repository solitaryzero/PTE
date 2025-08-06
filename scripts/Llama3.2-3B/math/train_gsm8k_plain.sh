CUDA_VISIBLE_DEVICES=3,5 accelerate launch --config_file ./scripts/Llama3.2-3B/accelerate_config.yaml \
    ./src/math/train_gsm8k_deepspeed.py \
    --base_model ./models/Llama-3.2-3B \
    --data_path ./data/gsm8k \
    --save_model_path ./models/Llama-3.2-3B_tuned/gsm8k_plain \
    --save_result_path ./output/Llama-3.2-3B_tuned/gsm8k_plain \
    --bf16 \
    --do_eval \
    --learning_rate 1e-5 \
    --train_batch_size 1 \
    --epoch 6 \
    --logging_steps 200 \
    --max_new_tokens 2048 \
    --seed 42