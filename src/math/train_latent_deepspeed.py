import os
import json
import argparse
from tqdm import tqdm
import time
import random
from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from accelerate import Accelerator
import wandb

from data import load_gsm8k, process_gsm8k_latent, encode_number_multihot_fn, get_dummy_latent
from utils import build_dataloader, get_optimizer, get_constant_scheduler
from latent_llama_model import LatentLlamaTokenizer, LatentLlamaForCausalLM
from latent_qwen_model import LatentQwen2Tokenizer, LatentQwen2ForCausalLM


def run(
    args,
    train_dataset,
    test_dataset,
    save_model_path,
):  
    wandb.login()
    latent_dim = (args.int_precision + args.frac_precision)*10

    if ('Llama' in args.base_model):
        model = LatentLlamaForCausalLM.from_pretrained(args.base_model, latent_dim=latent_dim).cuda()
        tokenizer = LatentLlamaTokenizer.from_pretrained(args.base_model)
    elif ('Qwen2.5' in args.base_model):
        model = LatentQwen2ForCausalLM.from_pretrained(args.base_model, latent_dim=latent_dim).cuda()
        tokenizer = LatentQwen2Tokenizer.from_pretrained(args.base_model)
    else:
        raise NotImplementedError('Unsupported model type') 
        
    if (tokenizer.pad_token is None):
        pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = pad_token_id
    else:
        pad_token = tokenizer.pad_token
        pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = pad_token_id

    latent_token = tokenizer.latent_token
    latent_token_id = tokenizer.latent_token_id

    latent_encoder = partial(
        encode_number_multihot_fn, 
        int_precision=args.int_precision,
        frac_precision=args.frac_precision,
    )
    dummy_latent = get_dummy_latent(
        int_precision=args.int_precision,
        frac_precision=args.frac_precision,
    )

    # Train
    dataset = process_gsm8k_latent(
        train_dataset,
        latent_token=latent_token,
        latent_encoding_func=latent_encoder,
    )
    def tokenize_function(element):
        full_tokens = tokenizer(
            element['prompt']+tokenizer.eos_token,
            truncation=True,
            max_length=4096,
            add_special_tokens=True,
        )
        query_tokens = tokenizer(
            element['query'],
            truncation=True,
            max_length=4096,
            add_special_tokens=True,
        )

        full_tokens['query_ids'] = query_tokens['input_ids']

        return full_tokens

    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=['query', 'prompt'],
    )
    train_dataloader = build_dataloader(
        dataset=tokenized_dataset,
        batch_size=args.train_batch_size,
        pad_token_id=pad_token_id,
        latent=True,
        latent_token=latent_token,
        latent_token_id=latent_token_id,
        dummy_latent=get_dummy_latent(args.int_precision, args.frac_precision),
    )

    optimizer = get_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_constant_scheduler(optimizer)

    # Accelerate
    accelerator = Accelerator(
        log_with='wandb',
    )
    accelerator.init_trackers(
        project_name="PTE_latent_gsm8k",
        config={
            "base_model": "Llama3.2-3B",
            "int_precision": args.int_precision, 
            "frac_precision": args.frac_precision, 
            "learning_rate": args.learning_rate,
            "epochs": args.epoch,
        },
    )

    train_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, model, optimizer, scheduler
    )

    start_time = time.time()

    accumulated_loss = 0.0
    accumulated_steps = 0
    for epoch in tqdm(range(args.epoch), desc='Epoch'):
        for step, batch in tqdm(enumerate(train_dataloader), desc='Step'):
            with accelerator.accumulate(model):
                outputs = model(
                    return_dict=True,
                    **batch,
                )
                loss = outputs.loss

                accumulated_loss += loss.item()
                accumulated_steps += 1
                if (accumulated_steps % args.logging_steps == 0):
                    avg_loss = accumulated_loss / args.logging_steps
                    print(f'Step {accumulated_steps} avg loss: {avg_loss}')
                    accelerator.log({
                        "epoch": epoch,
                        "loss": avg_loss,
                    }, step=accumulated_steps)
                    accumulated_loss = 0.0
                
                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    tokenizer.save_pretrained(save_model_path)
    model.save_pretrained(save_model_path)

    accelerator.end_training()

    time_elapsed = time.time()-start_time
    run_time = time_elapsed

    h = int(run_time//3600)
    m = int((run_time-(h*3600))//60)
    s = run_time-(h*3600)-(m*60)
    print(f'[Training] Run time : {h}h{m}m{s}s')

    # Eval
    if (args.do_eval):
        test_dataset = process_gsm8k_latent(
            test_dataset,
            latent_token=latent_token,
            latent_encoding_func=latent_encoder,
        )

        if (args.fp16):
            model = model.half()
        elif (args.bf16):
            model = model.bfloat16()

        correct, total = 0, 0
        generation_config = GenerationConfig(
            num_beams=1,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        all_predictions = []
        for entry in tqdm(test_dataset, desc='Eval'):
            prompt = entry['query']
            inputs = tokenizer(
                prompt, 
                truncation=True,
                max_length=2048,
                add_special_tokens=True,
            )
            input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0).cuda()
            attention_mask = torch.tensor(inputs["attention_mask"]).unsqueeze(0).cuda()
            with torch.no_grad():
                generation_outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    max_new_tokens=args.max_new_tokens,
                )

            decoded = tokenizer.decode(generation_outputs.sequences[0], skip_special_tokens=True)
            golden = entry['prompt'].strip().split('The answer is ')[-1].strip('.')
            golden = float(golden.replace(',', '').replace(' ', ''))
            try:
                prediction = decoded.strip().split('The answer is ')[-1].strip('.')
                prediction = float(prediction.replace(',', '').replace(' ', ''))
            except:
                prediction = None
            
            # print(prompt)
            # print(decoded)
            # print(f'No. tokens: {len(generation_outputs.sequences[0])}')
            # print(golden)
            # print(prediction)
            # print(golden == prediction)
            # print('=====')
            # input()

            js = {
                'query': entry['query'],
                'output': decoded,
                'golden': golden,
                'prediction': prediction,
            }
            all_predictions.append(js)

            if (golden == prediction):
                correct += 1
            total += 1

        result = {
            'correct': correct,
            'total': total,
            'accuracy': correct/total,
        }

        return model, result, all_predictions
    else:
        return model, None, None

def main(args):
    train_dataset = load_gsm8k(split='train', data_path=args.data_path, shuffle=True)
    test_dataset = load_gsm8k(split='test', data_path=args.data_path, shuffle=True)

    save_model_path = os.path.join(args.save_model_path)
    if not(os.path.exists(save_model_path)):
        os.makedirs(save_model_path)
    model, result, predictions = run(args, train_dataset, test_dataset, save_model_path)

    if (result is not None):
        print('Accuracy:')
        print(result)

        if not(os.path.exists(args.save_result_path)):
            os.makedirs(args.save_result_path)
        out_path = os.path.join(args.save_result_path, f'scores.json')
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(result, fout)

        out_path = os.path.join(args.save_result_path, f'predictions.json')
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(predictions, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path args
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_model_path', type=str)
    parser.add_argument('--save_result_path', type=str, required=True)
    parser.add_argument('--accelerator_config', type=str, default=None)

    # Model args
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')

    # Training args
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=2000)

    # Stage 2 args
    parser.add_argument('--second_stage', action='store_true')
    parser.add_argument('--previous_model', type=str)

    # Misc
    parser.add_argument('--int_precision', type=int, default=20)
    parser.add_argument('--frac_precision', type=int, default=10)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)