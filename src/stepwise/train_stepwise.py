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
    train_dataset,
    test_dataset,
    save_model_path,
):  
    llm = AutoModelForCausalLM.from_pretrained(args.base_model).cuda()
    if ('Llama' in args.base_model):
        tokenizer = LatentLlamaTokenizer.from_pretrained(args.base_model)
    elif ('Qwen2.5' in args.base_model):
        tokenizer = LatentQwen2Tokenizer.from_pretrained(args.base_model)
    else:
        raise NotImplementedError('Unsupported model type') 
        
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

    projector_file_name = os.path.join(args.latent_projector_path, f'{args.latent_type}_prober.bin')
    latent_projector = torch.load(projector_file_name, map_location='cpu', weights_only=False)
    latent_projector = latent_projector.to(llm.device)

    model = LSVModel(
        llm=llm,
        tokenizer=tokenizer,
        latent_encoder=latent_encoder,
        latent_projector=latent_projector,
        latent_step_n_tokens=args.latent_step_n_tokens,
    )

    # Train
    # if (args.fp16):
    #     model = model.half()
    # elif (args.bf16):
    #     model = model.bfloat16()

    train_dataset = process_gsm8k_aug(
        train_dataset,
        sanity_check=True,
    )

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
    )

    optimizer = get_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_constant_scheduler(optimizer)

    # Accelerate
    accelerator = Accelerator()
    # accelerator.init_trackers(
    #     project_name="LSV_gsm8k_aug",
    #     config={
    #         "base_model": args.base_model.split('/')[-1],
    #         "int_precision": args.int_precision, 
    #         "frac_precision": args.frac_precision, 
    #         "learning_rate": args.learning_rate,
    #         "epochs": args.epoch,
    #     },
    # )

    train_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, model, optimizer, scheduler
    )

    start_time = time.time()

    accumulated_loss = {
        'loss': 0.0,
        'ce_loss': 0.0,
        'latent_loss': 0.0,
    }
    accumulated_steps = 0
    for epoch in tqdm(range(args.epoch), desc='Epoch'):
        for step, batch in tqdm(enumerate(train_dataloader), desc='Step'):
            with accelerator.accumulate(model):
                outputs = model(
                    batch,
                )
                loss = outputs['loss']
                ce_loss, latent_loss = outputs['ce_loss'], outputs['latent_loss']

                accumulated_loss['loss'] += loss.item()
                accumulated_loss['ce_loss'] += ce_loss.item()
                accumulated_loss['latent_loss'] += latent_loss.item()
                accumulated_steps += 1
                if (accumulated_steps % args.logging_steps == 0):
                    avg_loss = accumulated_loss['loss'] / args.logging_steps
                    avg_ce_loss = accumulated_loss['ce_loss'] / args.logging_steps
                    avg_latent_loss = accumulated_loss['latent_loss'] / args.logging_steps
                    print(f'Step {accumulated_steps} avg loss: {avg_loss}; CE: {avg_ce_loss}; Latent: {avg_latent_loss}')
                    # accelerator.log({
                    #     "epoch": epoch,
                    #     "loss": avg_loss,
                    #     'ce_loss': avg_ce_loss,
                    #     'latent_loss': avg_latent_loss
                    # }, step=accumulated_steps)
                    for key in accumulated_loss:
                        accumulated_loss[key] = 0.0
                
                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    tokenizer.save_pretrained(save_model_path)
    # model.save_pretrained(save_model_path)
    model_file_name = os.path.join(save_model_path, f'{args.latent_type}_lsv.bin')
    torch.save(model, model_file_name)

    accelerator.end_training()

    time_elapsed = time.time()-start_time
    run_time = time_elapsed

    h = int(run_time//3600)
    m = int((run_time-(h*3600))//60)
    s = run_time-(h*3600)-(m*60)
    print(f'[Training] Run time : {h}h{m}m{s}s')

    # Eval
    if (args.do_eval):
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
    # train_dataset = load_gsm8k_aug(split='train', data_path=args.data_path, shuffle=False).select(range(20))
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
    parser.add_argument('--latent_projector_path', type=str, required=True)
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
    parser.add_argument('--max_latent_tokens', type=int, default=20)
    parser.add_argument('--latent_step_n_tokens', type=int, default=2)

    # Training args
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