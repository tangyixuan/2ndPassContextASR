import json
from datasets import load_from_disk
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from jiwer import wer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import sys
import argparse

# load saved prompt templates
template_path = './prompt_templates.json'

with open(template_path, 'r') as fn:
    prompt_templates = json.load(fn) 
    
def parse_args():
    parser = argparse.ArgumentParser(description="Hyper-parameters for prompt tuning on 2nd pass resocring.")

    parser.add_argument(
        "--prompt_idx", 
        type=str, 
        default=None, 
        choices=list(prompt_templates.keys()),
        help="The index of prompt to use."
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=512, 
        help="The maximum total input sequence length after tokenization."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="long",
        choices=['short', 'long'],
        help="The mode of inference, short for inference on short examples, long for inference on long examples."
    )
    parser.add_argument(
        "--cuda_idx",
        type=int,
        default=0,
        help="The index of gpu card used",
        choices=[0,1,2,3],
    )    
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="The path to save the output file."
    )    
    parser.add_argument(
        "--n_hypos",
        type=int,
        default=10,
        help="best n hypothses to use."
    )    
    parser.add_argument(
        "--new_model_path",
        type=str,
        default=None,
        help="The path to save the output file."
    )        
    
    args = parser.parse_args()
    if args.output_file is None:
        raise ValueError("output_file must be specified.")

    return args

args = parse_args()
print(args)

print(f"Running with cuda_idx: {args.cuda_idx}, prompt_idx: {args.prompt_idx}")

# load processed data
# processed = load_from_disk('./squad_processed_whisper_base')
# print("Data loaded")

device = torch.device(f"cuda:{args.cuda_idx}")
print(device)

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

from peft import PeftModel
print("Start loading model to GPU...")
if args.new_model_path is not None:
    print('in new model path', args.new_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    model= PeftModel.from_pretrained(base_model, args.new_model_path)
    model = model.merge_and_unload()

else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16
    )

model.to(device)
print("Model loaded to GPU.")



def add_prompt(example):
    prompt = prompt_templates[args.prompt_idx].format(
    question=example['question'], 
    hypo="\n".join(example['n_best'][:args.n_hypos]), 
    # hypo="\n".join(list(set([t[:-1] for t in example['n_best'][:args.n_hypos]]))), 
    para=(example['title'] + " " + example['context'])
    )
    result = tokenizer(prompt) #, return_tensors="pt")
    result["length"] = len(result["input_ids"])
    return result

prompt_val = processed["validation"].map(
    add_prompt
)

val_filtered_short = prompt_val.filter(lambda example: len(example['input_ids'])<=args.max_seq_length)
val_filtered_long = prompt_val.filter(lambda example: len(example['input_ids'])>args.max_seq_length)
val_filtered_short.save_to_disk(f'./result_whisper_base/debug_ft16_prompt1_10_short_gold.json')
val_filtered_long.save_to_disk(f'./result_whisper_base/debug_ft16_prompt1_10_long_gold.json')

# val_filtered_long = load_from_disk(f'./result_whisper_base/debug_ft16_prompt1_10_long_gold.json')
# val_filtered_short = load_from_disk(f'./result_whisper_base/debug_ft16_prompt1_10_short_gold.json')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# for long
val_in_long = val_filtered_long.remove_columns(['id', 'title', 'length', 'context', 'question', 'answers', 'fid', 'gold', 'n_best'])
eval_dataloader_long = DataLoader(val_in_long, collate_fn=data_collator, batch_size=1)

all_decoded_long = []
for step, batch in tqdm(enumerate(eval_dataloader_long)):
    batch.to(device) 
    # print(batch['input_ids'].shape)
    with torch.no_grad():
        generated_ids = model.generate(**batch, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)#,  do_sample=True)
    
    decoded = tokenizer.batch_decode(generated_ids[:,batch['input_ids'].shape[1]:])
    all_decoded_long.extend(decoded)

# processed = load_from_disk('./squad_processed_whisper_base')
with open(f"{args.output_file[:-5]}_long.json", 'w') as fn:
    json.dump(all_decoded_long, fn)
    
# for short
val_in_short = val_filtered_short.remove_columns(['id', 'title', 'length', 'context', 'question', 'answers', 'fid', 'gold', 'n_best'])
eval_dataloader_short = DataLoader(val_in_short, collate_fn=data_collator, batch_size=4)

all_decoded_short = []
for step, batch in tqdm(enumerate(eval_dataloader_short)):
    batch.to(device) 
    # print(batch['input_ids'].shape)
    with torch.no_grad():
        generated_ids = model.generate(**batch, pad_token_id=tokenizer.eos_token_id, max_new_tokens=25)#,  do_sample=True)
    
    decoded = tokenizer.batch_decode(generated_ids[:,batch['input_ids'].shape[1]:])
    all_decoded_short.extend(decoded)
                                        
with open(f"{args.output_file[:-5]}_short.json", 'w') as fn:
    json.dump(all_decoded_short, fn)