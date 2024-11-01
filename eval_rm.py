from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
import tqdm
from peft import PeftModel, PeftConfig
from torch.cuda.amp import autocast
import gc

model_checkpoint = '/iris/u/asap7772/autocrit/checkpoints_linearlr/meta-llama_Meta-Llama-3.1-8B-Instruct_0e9e39f2_Asap7772_llama_3.1_binary_train_small_processed_hhstyle_1e-05'
dataset_name = 'Asap7772/llama_3.1_binary_train_processed_hhstyle'
peft=False
batch_size = 128

if not peft:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, 
        num_labels=1, 
        trust_remote_code=True, 
        device_map='auto',
        cache_dir='/iris/u/asap7772/.cache/',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
else:
    config = PeftConfig.from_pretrained(
        model_checkpoint, 
        cache_dir='/iris/u/asap7772/.cache/'
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        num_labels=1, 
        trust_remote_code=True, 
        device_map='auto',
        cache_dir='/iris/u/asap7772/.cache/'
    )
    model = PeftModel.from_pretrained(
        model,
        model_checkpoint, 
        is_trainable=True
    )
    model.print_trainable_parameters()
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
tokenizer.padding_side = "right"
tokenizer.truncation_side = "left"

model.eval()
for param in model.parameters():
    param.requires_grad = False

def tokenize(prompt, selected, rejected, tokenizer):
    return {
        "selected_input_ids": tokenizer(prompt + selected + tokenizer.eos_token, truncation=True, max_length=1052).input_ids,
        "rejected_input_ids": tokenizer(prompt + rejected + tokenizer.eos_token, truncation=True, max_length=1052).input_ids,
    }

def collate_fn(batch):
    input_ids = sum([[x["rejected_input_ids"], x["selected_input_ids"]] for x in batch], [])
    return tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")

dataset = load_dataset(dataset_name)
if "chosen" in dataset["train"].column_names:
    dataset = dataset.rename_column("chosen", "selected")
if "replies" in dataset["train"].column_names:
    dataset = dataset.map(lambda x: {"selected": x["replies"][0], "rejected": x["replies"][1]}, remove_columns=["replies"])

def to_vicuna_format(sample):
    prompt = sample["prompt"].strip()
    prompt = prompt.replace("\n\nHuman: ", "</s>USER: ") \
                    .replace("\n\nAssistant: ", " ASSISTANT: ") \
                    .replace("\n\nAssistant:", " ASSISTANT:")
    if prompt.startswith("Human: "):
        prompt = prompt.replace("Human: ", "USER: ")
    if prompt.startswith("</s>"):
        prompt = prompt[4:]

    selected = " " + sample["selected"].strip()
    rejected = " " + sample["rejected"].strip()

    return {"prompt": prompt, "selected": selected, "rejected": rejected}

def to_oa_format(sample):
    prompt = sample["prompt"].strip()
    prompt = prompt.replace("\n\nHuman: ", "</s><|prompter|>") \
                    .replace("\n\nAssistant: ", "</s><|assistant|>") \
                    .replace("\n\nAssistant:", "</s><|assistant|>")
    if prompt.startswith("Human: "):
        prompt = prompt.replace("Human: ", "<|prompter|>")

    selected = sample["selected"].strip()
    rejected = sample["rejected"].strip()

    return {"prompt": prompt, "selected": selected, "rejected": rejected}

dataset = dataset.map(to_vicuna_format)
tokenized = dataset.map(tokenize, input_columns=["prompt", "selected", "rejected"], fn_kwargs=dict(tokenizer=tokenizer), desc="Tokenizing")

train_dataloader = torch.utils.data.DataLoader(tokenized["train"], shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = torch.utils.data.DataLoader(tokenized["test"], shuffle=False, batch_size=batch_size, collate_fn=collate_fn)


with torch.no_grad():
    for split in ['test']:
        dataloader = train_dataloader if split == 'train' else test_dataloader

        all_accuracies = []
        for batch in tqdm.tqdm(dataloader, desc=f"Evaluating on {dataset_name}", leave=True):
            with torch.no_grad():
                scores = model(**batch)[0]
            delta_scores = scores.reshape(-1, 2).diff().view(-1)
            accuracy = (delta_scores > 0).float().mean()
            
            chosen_scores = scores.reshape(-1, 2)[:, 1]
            rejected_scores = scores.reshape(-1, 2)[:, 0]
            
            tqdm.tqdm.write(f"Chosen logits: {chosen_scores.mean().item():.2f}")
            tqdm.tqdm.write(f"Rejected logits: {rejected_scores.mean().item():.2f}")
            tqdm.tqdm.write(f"Accuracy: {accuracy:.2f}")
            all_accuracies.append(accuracy)
            
            # Cleanup to free memory
            del scores, delta_scores, chosen_scores, rejected_scores
            gc.collect()
            torch.cuda.empty_cache()
        
        print(f"Accuracy: {sum(all_accuracies) / len(all_accuracies):.2f}")