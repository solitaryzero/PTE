import re
from tqdm import tqdm

import datasets
from datasets import load_dataset, load_from_disk, Dataset


def load_gsm8k(
    split,
    data_path="openai/gsm8k",
    shuffle=False,
    seed=42
):
    if (data_path == "openai/gsm8k"):
        ds = load_dataset(data_path, "main", split=split)
    else:
        ds = load_from_disk(data_path)[split]

    if (shuffle):
        ds = ds.shuffle(seed=seed)

    return ds


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


def replace_numbers_with_variables(text, unk_token=None):
    number_to_var = {}
    counter = 1
    
    def replace_match(match):
        nonlocal counter
        nonlocal unk_token
        number = match.group()

        suffix = ''
        if (number.endswith(',')):
            suffix = ','
            number = number[:-1]

        if (number.endswith('..')): # Looped numbers like 3.33...
            num_str = number.rstrip('.')
        elif (number.endswith('.')):
            suffix = '.'
            num_str = number[:-1]
        else:
            num_str = number

        num_str = num_str.replace(',', '')
        if (len(num_str.replace('.', '')) == 0): # Unexpected '.,'
            return match.group()

        # Sanity check
        try:
            if '.' in num_str:
                num = float(num_str)
            else:
                num = int(num_str)
        except ValueError:
            print(number)
            print(text)
            return match.group()
        
        if (unk_token is not None):
            return unk_token

        if num in number_to_var:
            return number_to_var[num] + suffix
        else:
            var_name = f'y_{counter}'
            number_to_var[num] = var_name
            counter += 1
            return var_name + suffix
    
    pattern = r'([0-9.,]{2,})|([0-9]+)'
    result = re.sub(pattern, replace_match, text)

    return result


def process_gsm8k(
    examples,
    replace_numbers='no',
    unk_token=None,
):
    pattern_brackets = r'<<[^>]+>>'

    queries, prompts = [], []
    iterator = tqdm(examples.iter(batch_size=1), desc='Preprocess')
    for entry in iterator:
        query = f"Q: {entry['question'][0]}\nA: Let\'s think step by step. "

        raw_answer = entry['answer'][0]
        t = raw_answer.split('####')
        answer = t[0] + f'The answer is {t[1].strip()}.'
        answer = re.sub(pattern_brackets, '', answer) # remove brackets
        prompt = query + answer
        
        if (replace_numbers == 'yes'):
            replaced_prompt = replace_numbers_with_variables(prompt, unk_token)
            replaced_query = replace_numbers_with_variables(query, unk_token)
            queries.append(replaced_query)
            prompts.append(replaced_prompt)
        elif (replace_numbers == 'mixed'):
            replaced_prompt = replace_numbers_with_variables(prompt, unk_token)
            replaced_query = replace_numbers_with_variables(query, unk_token)
            queries.append(replaced_query)
            prompts.append(replaced_prompt)

            queries.append(query)
            prompts.append(prompt)
        else:
            assert (replace_numbers == 'no')
            queries.append(query)
            prompts.append(prompt)

    processed_dataset = Dataset.from_dict({
        'query': queries,
        'prompt': prompts,
    })

    return processed_dataset


def process_gsm8k_aug(
    examples,
    replace_numbers='no',
    unk_token=None,
):

    queries, prompts = [], []
    iterator = tqdm(examples.iter(batch_size=1), desc='Preprocess')
    for entry in iterator:
        query = f"Q: {entry['question'][0]}\nA: Let\'s think step by step. "

        cot, answer = entry['cot'][0], entry['answer'][0]
        answer = cot + '\n' + f'The answer is {answer.strip()}.'
        prompt = query + answer
        
        if (replace_numbers == 'yes'):
            replaced_prompt = replace_numbers_with_variables(prompt, unk_token)
            replaced_query = replace_numbers_with_variables(query, unk_token)
            queries.append(replaced_query)
            prompts.append(replaced_prompt)
        elif (replace_numbers == 'mixed'):
            replaced_prompt = replace_numbers_with_variables(prompt, unk_token)
            replaced_query = replace_numbers_with_variables(query, unk_token)
            queries.append(replaced_query)
            prompts.append(replaced_prompt)

            queries.append(query)
            prompts.append(prompt)
        else:
            assert (replace_numbers == 'no')
            queries.append(query)
            prompts.append(prompt)
            

    processed_dataset = Dataset.from_dict({
        'query': queries,
        'prompt': prompts,
    })

    return processed_dataset


def replace_numbers_with_latents(text, latent_token, latent_encoder):
    latent_embeddings = []

    def replace_match(match):
        nonlocal latent_token
        nonlocal latent_encoder
        nonlocal latent_embeddings

        number = match.group()

        suffix = ''
        if (number.endswith(',')):
            suffix = ','
            number = number[:-1]

        if (number.endswith('..')): # Looped numbers like 3.33...
            num_str = number.rstrip('.')
        elif (number.endswith('.')):
            suffix = '.'
            num_str = number[:-1]
        else:
            num_str = number

        num_str = num_str.replace(',', '')
        if (len(num_str.replace('.', '')) == 0): # Unexpected '.,'
            return match.group()

        # Sanity check
        try:
            if '.' in num_str:
                num = float(num_str)
            else:
                num = int(num_str)
        except ValueError:
            print(number)
            print(text)
            return match.group()
        
        latent_embed = latent_encoder.encode(num)
        latent_embeddings.append(latent_embed)

        return latent_token + suffix
    
    pattern = r'([0-9.,]{2,})|([0-9]+)'
    result = re.sub(pattern, replace_match, text)

    return result, latent_embeddings


def process_gsm8k_latent(
    examples,
    latent_token,
    latent_encoder,
):
    pattern_brackets = r'<<[^>]+>>'

    queries, prompts, latent_embeddings = [], [], []
    iterator = tqdm(range(len(examples['question'])), desc='Preprocess')
    for i in iterator:
        query = f"Q: {examples['question'][i]}\nA: Let\'s think step by step.\n"
        queries.append(query)

        raw_answer = examples['answer'][i]
        t = raw_answer.split('####')
        thinking = t[0]
        thinking = re.sub(pattern_brackets, '', thinking) # remove brackets
        thinking, latent_embeds = replace_numbers_with_latents(
            thinking, 
            latent_token=latent_token, 
            latent_encoder=latent_encoder,
        )
        
        answer = thinking + f'The answer is {t[1].strip()}.'
        prompt = query + answer
        prompts.append(prompt)
        latent_embeddings.append(latent_embeds)

    processed_dataset = Dataset.from_dict({
        'query': queries,
        'prompt': prompts,
        'latent_embeds': latent_embeddings,
    })

    return processed_dataset


def process_gsm8k_aug_latent(
    examples,
    latent_token,
    latent_encoder,
    only_intermediate_result=False,
):

    queries, prompts, latent_embeddings = [], [], []
    iterator = tqdm(examples.iter(batch_size=1), desc='Preprocess')
    for entry in iterator:
        if (entry['cot'][0] == ''): # Sanity check
            continue

        query = f"Q: {entry['question'][0]}\nA: Let\'s think step by step.\n"
        queries.append(query)

        thinking, answer = entry['cot'][0], entry['answer'][0]

        if (only_intermediate_result):
            steps = thinking.split('>>')
            intermediates = []
            for step in steps:
                r = step.split('=')[-1]
                r = r.strip('>').strip()
                intermediates.append(r)

            thinking = ' '.join(intermediates)

        thinking, latent_embeds = replace_numbers_with_latents(
            thinking, 
            latent_token=latent_token, 
            latent_encoder=latent_encoder,
        )
        
        answer = thinking + f'\nThe answer is {answer.strip()}.'
        prompt = query + answer
        prompts.append(prompt)

        latent_embeddings.append(latent_embeds)

    processed_dataset = Dataset.from_dict({
        'query': queries,
        'prompt': prompts,
        'latent_embeds': latent_embeddings,
    })

    return processed_dataset
        

if __name__ == '__main__':
    text = "There are 3 apples and 4/5 oranges. The temperature is 25.5 degrees. 3 apples cost 5 dollars."
    output = replace_numbers_with_variables(text)
    print(output)