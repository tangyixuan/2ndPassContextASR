import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys

train_size = sys.argv[1]
random_count = sys.argv[2]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)


prompt_template = '<s>[INST] The N-best hypotheses for a speech utterance from an ASR model are:\n{hypo}\nThe content is a question for the following passage:\n{para}\nPlease report the true transcript of the speech utterance. [/INST]\nThe true transcript of the speech utterance is: {gold} </s>'

def add_prompt(example):
    prompt = prompt_template.format(
        question=example['question'], 
        hypo="\n".join(example['n_best'][:10]), 
        # hypo="\n".join(list(set([t[:-1] for t in example['n_best'][:args.n_hypos]]))), 
        para=(example['title'] + " " + example['context']),
        gold=example['gold']
    )
    return prompt

from datasets import load_from_disk
dataset_src = load_from_disk('./squad_processed_whisper_base')
print("Data loaded")

text_column = [add_prompt(data_point) for data_point in dataset_src["validation"]]
prompt_val = dataset_src["validation"].add_column("prompt", text_column)
prompt_val = prompt_val.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
prompt_val.save_to_disk("./prompt_val_base.json") 
# prompt_val = load_from_disk("./prompt_val_base.json") 

text_column = [add_prompt(data_point) for data_point in dataset_src["train"]]
prompt_train = dataset_src["train"].add_column("prompt", text_column)
prompt_train = prompt_train.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
prompt_train.save_to_disk("./prompt_train_base.json") 
# prompt_train = load_from_disk("./prompt_train_base.json") 
prompt_train = prompt_train.shuffle()
# print(prompt_train)
# print(prompt_val)

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

import bitsandbytes as bnb
def find_all_linear_names(model):
  cls = bnb.nn.Linear4bit
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
      lora_module_names.remove('lm_head')
  return list(lora_module_names)

modules = find_all_linear_names(model)
# print(modules)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

import transformers

from trl import SFTTrainer

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    train_dataset=prompt_train.select(range(int(train_size))),# train_data,
    eval_dataset=prompt_val.select(range(32)), # test_data,
    dataset_text_field="prompt",
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        warmup_steps=0.03,
        num_train_epochs=3,
        # max_steps=5,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir=f"outputs_{train_size}_count_{random_count}",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False 
trainer.train()

new_model = f"mistral_base_train_size_{train_size}_count_{random_count}" 
trainer.model.save_pretrained(new_model)