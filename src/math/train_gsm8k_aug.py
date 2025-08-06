import os
import json
import argparse
from tqdm import tqdm
import gc
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, GenerationConfig

from data import load_gsm8k, process_gsm8k, load_gsm8k_aug, process_gsm8k_aug

def run(
    args,
    train_dataset,
    test_dataset,
    save_model_path,
):  
    if (args.second_stage):
        model = AutoModelForCausalLM.from_pretrained(args.previous_model).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if (tokenizer.pad_token is None):
        pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = pad_token_id
    else:
        pad_token = tokenizer.pad_token
        pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = pad_token_id

    if args.use_unk:
        unk_token = '<|fim_pad|>'
        unk_token_id = tokenizer._convert_token_to_id(unk_token)
    else:
        unk_token = None
        unk_token_id = None

    # Train
    dataset = process_gsm8k_aug(
        train_dataset,
        replace_numbers=args.replace_numbers,
        unk_token=unk_token,
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

    epoch = args.epoch

    train_params = TrainingArguments(
        output_dir=save_model_path,
        num_train_epochs=epoch,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_torch",
        save_strategy='epoch',
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.grad_clip,
        max_steps=-1,
        warmup_ratio=0,
        group_by_length=False,
        remove_unused_columns=False,
        lr_scheduler_type="constant",
        report_to=args.report_to,
    )
    tokenizer.save_pretrained(save_model_path)

    # Train
    def collate(elements):
        nonlocal unk_token, unk_token_id, pad_token_id
        tokenlist = [e["input_ids"] for e in elements]
        query_lengths = [len(e['query_ids']) for e in elements]
        tokens_maxlen = max([len(t) for t in tokenlist])

        input_ids, labels, attention_masks = [], [], []
        for tokens, query_len in zip(tokenlist, query_lengths):
            pad_len = tokens_maxlen - len(tokens)
            input_ids.append(tokens + [pad_token_id]*pad_len)
            attention_masks.append([1]*len(tokens) + [0]*pad_len)

            label = [-100]*query_len + tokens[query_len:] + [-100]*pad_len
            if (unk_token is not None):
                for i, t in enumerate(input_ids):
                    if (t == unk_token_id):
                        label[i] = -100
            try:
                assert len(label) == tokens_maxlen
            except AssertionError as e:
                nonlocal tokenizer
                print(tokens_maxlen)
                print(len(label))
                print(tokenizer.convert_ids_to_tokens(tokens))
                print(tokenizer.convert_ids_to_tokens(label))
                quit()

            labels.append(label)

        batch = {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_masks),
        }
        return batch

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=collate,
        args=train_params,
        train_dataset=tokenized_dataset,
        eval_dataset=None,
        compute_metrics=None,
    )

    trainer.train()
    model.save_pretrained(save_model_path)

    # Eval
    if (args.do_eval):
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
    else:
        return model, None, None

def main(args):
    train_dataset = load_gsm8k_aug(split='train', data_path=args.aug_data_path, shuffle=True)
    test_dataset = load_gsm8k(split='test', data_path=args.data_path)

    save_model_path = os.path.join(args.save_model_path)
    if not(os.path.exists(save_model_path)):
        try:
            os.makedirs(save_model_path)
        except:
            pass
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
    parser.add_argument('--aug_data_path', type=str, required=True)
    parser.add_argument('--save_model_path', type=str)
    parser.add_argument('--save_result_path', type=str, required=True)

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
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    # Stage 2 args
    parser.add_argument('--second_stage', action='store_true')
    parser.add_argument('--previous_model', type=str)

    # Misc
    parser.add_argument('--replace_numbers', type=str, choices=['yes', 'no', 'mixed'], default='no')
    parser.add_argument('--use_unk', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report_to', type=str, default='wandb')

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)