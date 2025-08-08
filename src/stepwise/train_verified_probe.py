import os
import re
import json
import argparse
from tqdm import tqdm
import time
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteria
from accelerate import Accelerator

from data import load_gsm8k_aug, process_gsm8k_aug, convert_to_probe_dataset
from utils import build_probe_dataloader, get_probe_optimizer, get_constant_scheduler
from latent_encoders import latent_encoder_factory


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = set(stop_token_ids)
    
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() in self.stop_token_ids


def extract_first_float(s):
    match = re.search(r'-?\d+\.?\d*', s)
    if match:
        return match.group()
    else:
        return '0'


def run(
    args,
    train_dataset,
    test_dataset,
    save_model_path,
):  
    llm = AutoModelForCausalLM.from_pretrained(args.base_model).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        
    if (tokenizer.pad_token is None):
        pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.eos_token_id
        llm.generation_config.pad_token_id = pad_token_id
    else:
        pad_token = tokenizer.pad_token
        pad_token_id = tokenizer.pad_token_id
        llm.generation_config.pad_token_id = pad_token_id

    latent_encoder = latent_encoder_factory(
        latent_type=args.latent_type,
        int_precision=args.int_precision,
        frac_precision=args.frac_precision,
        intermediate_dim=args.intermediate_dim,
    )
    model = latent_encoder.get_projection_module(input_dimension=llm.config.hidden_size)

    def tokenize_function(element):
        full_prompt = element['prompt']
        full_tokens = tokenizer(
            full_prompt,
            truncation=True,
            max_length=4096,
            add_special_tokens=True,
        )

        return full_tokens

    stop_words = [">>", "\n"]
    stop_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in stop_words]
    token_stopping_criteria = StopOnTokens(stop_ids)

    if (args.do_train):
        # Train
        train_dataset = process_gsm8k_aug(
            train_dataset,
            sanity_check=True,
            # highlight_equal_sign=True,
        )

        train_dataset = convert_to_probe_dataset(train_dataset)

        tokenized_dataset = train_dataset.map(
            tokenize_function,
            remove_columns=['prompt'],
        )

        train_dataloader = build_probe_dataloader(
            dataset=tokenized_dataset,
            batch_size=args.train_batch_size,
            latent_encoder=latent_encoder,
            pad_token_id=pad_token_id,
            tokenizer=tokenizer,
            verified=True,
        )

        optimizer = get_probe_optimizer(
            model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = get_constant_scheduler(optimizer)

        start_time = time.time()

        accumulated_loss = 0.0
        accumulated_steps = 0
        model = model.to(llm.device)

        for epoch in tqdm(range(args.epoch), desc='Epoch'):
            for step, batch in tqdm(enumerate(train_dataloader), desc='Step'):
                generation_outputs = llm.generate(
                    input_ids=batch['input_ids'].to(llm.device),
                    attention_mask=batch['attention_mask'].to(llm.device),
                    stopping_criteria=[token_stopping_criteria],
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    max_new_tokens=args.max_new_tokens,
                )
                decoded_outputs = tokenizer.decode(generation_outputs.sequences[0], skip_special_tokens=True)
                prediction = decoded_outputs.split('=')[-1].strip()
                # prediction = prediction.split(' ')[0].split('>')[0].rstrip('.')
                prediction = extract_first_float(prediction)

                try:
                    prediction = float(prediction)
                except:
                    print('=== Bad Generation ==')
                    print(decoded_outputs.split('=')[-1])
                    continue

                golden = batch['answer'][0].item()

                if (prediction != golden): # wrong calculation, ignore
                    continue

                last_hidden_states = generation_outputs.hidden_states[0][-1][:, -1, :].detach()

                predictions = model(
                    last_hidden_states,
                )
                predictions = predictions.to(torch.float32)
                labels = batch['labels'].to(llm.device).to(torch.float32)
                loss = latent_encoder.latent_loss(predictions, labels)
                loss = torch.mean(loss)

                accumulated_loss += loss.item()
                accumulated_steps += 1
                if (accumulated_steps % args.logging_steps == 0):
                    avg_loss = accumulated_loss / args.logging_steps
                    print(f'Step {accumulated_steps} avg loss: {avg_loss}')
                    accumulated_loss = 0.0
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model_file_name = os.path.join(save_model_path, f'{args.latent_type}_prober.bin')
        torch.save(model, model_file_name)

        time_elapsed = time.time()-start_time
        run_time = time_elapsed

        h = int(run_time//3600)
        m = int((run_time-(h*3600))//60)
        s = run_time-(h*3600)-(m*60)
        print(f'[Training] Run time : {h}h{m}m{s}s')
    else:
        model_file_name = os.path.join(save_model_path, f'{args.latent_type}_prober.bin')
        model = torch.load(model_file_name, map_location='cpu', weights_only=False)
        model = model.to(llm.device)

    # Eval
    if (args.do_eval):
        test_dataset = process_gsm8k_aug(
            test_dataset,
            sanity_check=True,
            # highlight_equal_sign=True,
        )

        test_dataset = convert_to_probe_dataset(test_dataset)

        tokenized_dataset = test_dataset.map(
            tokenize_function,
            remove_columns=['prompt'],
        )

        test_dataloader = build_probe_dataloader(
            dataset=tokenized_dataset,
            batch_size=args.train_batch_size,
            latent_encoder=latent_encoder,
            pad_token_id=pad_token_id,
            tokenizer=tokenizer,
            verified=True,
        )

        if (args.fp16):
            llm = llm.half()
        elif (args.bf16):
            llm = llm.bfloat16()

        model.eval()
        correct, total = 0, 0

        all_predictions = []
        for step, batch in tqdm(enumerate(test_dataloader), desc='Eval'):
            with torch.no_grad():
                generation_outputs = llm.generate(
                    input_ids=batch['input_ids'].to(llm.device),
                    attention_mask=batch['attention_mask'].to(llm.device),
                    stopping_criteria=[token_stopping_criteria],
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    max_new_tokens=args.max_new_tokens,
                )
                decoded_outputs = tokenizer.decode(generation_outputs.sequences[0], skip_special_tokens=True)
                prediction = decoded_outputs.split('=')[-1].strip()
                # prediction = prediction.split(' ')[0].split('>')[0].rstrip('.')
                prediction = extract_first_float(prediction)

                try:
                    prediction = float(prediction)
                except:
                    print('=== Bad Generation ==')
                    print(decoded_outputs.split('=')[-1])
                    continue

                golden = batch['answer'][0].item()

                if (prediction != golden): # wrong calculation, ignore
                    continue

                last_hidden_states = generation_outputs.hidden_states[0][-1][:, -1, :].detach()

                predictions = model(
                    last_hidden_states,
                )
                predictions = predictions.to(torch.float32)
                labels = batch['labels'].to(llm.device).to(torch.float32)
                metrics = latent_encoder.compute_metrics(
                    predictions=predictions,
                    labels=labels,
                )

                batch_correct, batch_total = metrics['correct'], metrics['total']
                correct += batch_correct
                total += batch_total

        result = {
            'correct': correct,
            'total': total,
            'accuracy': correct/total,
        }

        return model, result, all_predictions
    else:
        return model, None, None

def main(args):
    # train_dataset = load_gsm8k_aug(split='train', data_path=args.data_path, shuffle=False).select(range(200))
    train_dataset = load_gsm8k_aug(split='train', data_path=args.data_path, shuffle=True)
    test_dataset = load_gsm8k_aug(split='test', data_path=args.data_path, shuffle=False)

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

    # Latent args
    parser.add_argument('--latent_type', type=str, required=True)
    parser.add_argument('--intermediate_dim', type=int, default=512)
    parser.add_argument('--int_precision', type=int, default=20)
    parser.add_argument('--frac_precision', type=int, default=10)

    # Training args
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=2000)

    # Misc
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)