import datasets
import os

input_dataset_name = 'Asap7772/llama_3.1_binary_train_small_processed'
output_dataset_name = 'Asap7772/llama_3.1_binary_train_small_processed_hhstyle'
token="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"

def format_prompt_hh(prompt):
    return f'Human: {prompt}\n\nAssistant:'

ds_dict = datasets.load_dataset(input_dataset_name)
def map_fn(examples):
    # map chosen rejected to prompt, selected, rejected, source
    examples['prompt'] = [format_prompt_hh(x[0]['content']) for x in examples['chosen']]
    examples['selected'] = [x[1]['content'] for x in examples['chosen']]
    examples['rejected'] = [x[1]['content'] for x in examples['rejected']]
    examples['source'] = ['llama31' for _ in examples['chosen']]
    return examples

ds_dict = ds_dict.map(map_fn, batched=True, num_proc=os.cpu_count(), remove_columns=['chosen'])
ds_dict.push_to_hub(output_dataset_name, token=token)