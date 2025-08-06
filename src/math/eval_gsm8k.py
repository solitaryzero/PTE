import os
import json
import argparse
from tqdm import tqdm
import gc
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, GenerationConfig
from accelerate import Accelerator

from data import load_gsm8k, process_gsm8k

def run(
    args,
    test_dataset,
):  
    model = AutoModelForCausalLM.from_pretrained(args.tuned_model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if (tokenizer.pad_token is None):
        pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = pad_token_id
    else:
        pad_token = tokenizer.pad_token
        pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = pad_token_id

    test_dataset = process_gsm8k(test_dataset)

    if (args.fp16):
        model = model.half()
    elif (args.bf16):
        model = model.bfloat16()

    correct, total = 0, 0
    total_cot_tokens = 0
    generation_config = GenerationConfig(
        num_beams=1,
        do_sample=False,
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

        cot = decoded.strip().split('The answer is ')[0]
        cot = cot.split('step by step. ')[-1].strip()
        cot_tokens = tokenizer.tokenize(cot)
        total_cot_tokens += len(cot_tokens)
        
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
    test_dataset = load_gsm8k(split='test', data_path=args.data_path)

    model, result, predictions = run(args, test_dataset)

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
    parser.add_argument('--tuned_model_path', type=str, required=True)
    parser.add_argument('--save_result_path', type=str, required=True)

    # Model args
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')

    # Misc
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)