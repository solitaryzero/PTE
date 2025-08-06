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

from data import load_gsm8k_aug, process_gsm8k_aug
from utils import build_dataloader, get_optimizer, get_constant_scheduler
from model import LSVModel, LatentLlamaTokenizer, LatentQwen2Tokenizer
from latent_encoders import latent_encoder_factory


def run(
    args,
    test_dataset,
    tuned_model_file_name,
):  
    model = torch.load(tuned_model_file_name, map_location='cpu', weights_only=False)
    model = model.to('cuda')

    if ('Llama' in args.base_model):
        tokenizer = LatentLlamaTokenizer.from_pretrained(args.base_model)
    elif ('Qwen2.5' in args.base_model):
        tokenizer = LatentQwen2Tokenizer.from_pretrained(args.base_model)
    else:
        raise NotImplementedError('Unsupported model type') 
        
    if (tokenizer.pad_token is None):
        pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.eos_token_id
        model.llm.generation_config.pad_token_id = pad_token_id
    else:
        pad_token = tokenizer.pad_token
        pad_token_id = tokenizer.pad_token_id
        model.llm.generation_config.pad_token_id = pad_token_id

    model.tokenizer = tokenizer

    # Eval
    test_dataset = process_gsm8k_aug(
        test_dataset,
        sanity_check=False,
    )

    if (args.fp16):
        model = model.half()
    elif (args.bf16):
        model = model.bfloat16()

    correct, total = 0, 0
    total_cot_tokens = 0
    generation_config = GenerationConfig(
        num_beams=1,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    all_predictions = []
    for entry in tqdm(test_dataset, desc='Eval'):
        model_inputs = {
            'question': [entry['query']],
        }

        with torch.no_grad():
            latent_generation_outputs = model.generate(
                model_inputs,
                llm_generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=args.max_new_tokens,
            )
            generation_outputs, num_latents = latent_generation_outputs['prediction'], latent_generation_outputs['num_latents']

        decoded = tokenizer.decode(generation_outputs.sequences[0], skip_special_tokens=True)
        golden = entry['answer'].strip().split('The answer is ')[-1].strip('.')
        golden = float(golden.replace(',', '').replace(' ', ''))
        try:
            prediction = decoded.strip().split('The answer is ')[-1].strip('.')
            prediction = float(prediction.replace(',', '').replace(' ', ''))
        except:
            prediction = None

        # print(num_latents)
        # input()
        total_cot_tokens += num_latents[0].item()
        
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
        'avg_tokens': total_cot_tokens/total,
    }

    return model, result, all_predictions

def main(args):
    test_dataset = load_gsm8k_aug(split='test', data_path=args.data_path, shuffle=False)

    tuned_model_file_name = os.path.join(args.tuned_model_path, f'{args.latent_type}_lsv.bin')
    model, result, predictions = run(args, test_dataset, tuned_model_file_name)

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
    parser.add_argument('--tuned_model_path', type=str)
    parser.add_argument('--save_result_path', type=str, required=True)

    # Model args
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')

    # Latent args
    parser.add_argument('--latent_type', type=str, required=True)

    # Misc
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)