import re
from tqdm import tqdm

import datasets
from datasets import load_dataset, load_from_disk, Dataset


def load_gsm8k_aug(
    split,
    data_path="zen-E/GSM8k-Aug",
    shuffle=False,
    seed=42
):
    if (data_path == "zen-E/GSM8k-Aug"):
        ds = load_dataset(data_path, split=split)
    else:
        ds = load_from_disk(data_path)[split]

    if (shuffle):
        ds = ds.shuffle(seed=seed)

    return ds


def process_gsm8k_aug(
    examples,
    sanity_check=False,
    highlight_equal_sign=False,
):
    queries, steps, step_results, answers = [], [], [], []
    iterator = tqdm(examples.iter(batch_size=1), desc='Preprocess')
    for entry in iterator:
        if (entry['cot'][0] == ''): # Sanity check
            continue

        query = f"Q: {entry['question'][0]}\nA: Let\'s think step by step. \n"
        cot, answer = entry['cot'][0], entry['answer'][0]

        sanity_flag = True
        step_list = cot.split('>>')[:-1]
        step_list = [s.strip().strip('<') for s in step_list]
        if (highlight_equal_sign):
            step_list = [s.replace('=', ' =') for s in step_list]

        intermediate_results = []
        for step in step_list:
            if (step.count('=') != 1):
                sanity_flag = False
                break

            r = step.split('=')[-1]
            r = r.strip('>').strip()

            if (sanity_check):
                try:
                    r = float(r)
                    assert ('inf' not in str(r))
                except:
                    sanity_flag = False
                    break

            intermediate_results.append(r)

        if not(sanity_flag):
            continue
        
        answer = f'\nThe answer is {answer.strip()}.'
        queries.append(query)
        answers.append(answer)
        steps.append(step_list)
        step_results.append(intermediate_results)

    processed_dataset = Dataset.from_dict({
        'query': queries,
        'steps': steps,
        'step_results': step_results,
        'answer': answers,
    })

    return processed_dataset


def convert_to_probe_dataset(
    examples,
):
    prompts, answers = [], []
    iterator = tqdm(examples.iter(batch_size=1), desc='Convert')
    for entry in iterator:
        steps, step_results = entry['steps'][0], entry['step_results'][0]
        assert (len(steps) == len(step_results))
        for i, step in enumerate(steps):
            res = step_results[i]

            query = entry['query'][0]
            prev_steps = ' '.join([('<<' + s + '>>') for s in steps[:i-1]])
            current_step = '<<' + '='.join(step.split('=')[:-1]) + '='
            if (len(prev_steps) > 0):
                current_step = ' ' + current_step
            prompt = query + prev_steps + current_step
            prompts.append(prompt)
            answers.append(res)

    processed_dataset = Dataset.from_dict({
        'prompt': prompts,
        'answer': answers,
    })

    return processed_dataset
        

if __name__ == '__main__':
    pass