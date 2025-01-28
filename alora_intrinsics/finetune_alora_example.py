import click, os, torch
import numpy as np
#from lakehouse.assets.config import DMF_MODEL_CACHE
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets

from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from alora_intrinsics.alora.peft_model_alora import PeftModelForCausalLM
from alora_intrinsics.alora.config import aLoraConfig
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig



DATA_PATH = os.getenv("HF_DATASETS_CACHE")
MODEL_NAME = "ibm-granite/granite-3.1-8b-instruct"


INVOCATION_PROMPT = "<|start_of_role|>certainty<|end_of_role|>"
# SAFETY_PROMPT = "<|start_of_role|>safety<|end_of_role|>"
# HALL_PROMPT = "<|start_of_role|>hallucination<|end_of_role|>"
DATASET_PATH = "PATH_TO_DATA"
DATASET_FILES = ["file1.jsonl","file2.json"]


def get_datasets():
    datasets = []
    for ds in DATASET_FILES:
        if ds[-1] == "n": #json

            file = open(DATASET_PATH + ds)
            data = json.load(file)


        else: #jsonl
            file = open(DATASET_PATH + ds)
            data = {"conversations":[(json.loads(line))["messages"] for line in file]}
        datasets.append(data)
    return datasets
def process_datasets(datasets,tokenizer,max_rows):
    proc_datasets = []

    for ds in datasets:
        inputs = []
        targets = []
        add = ""



        max_rs = max_rows
    
        
        for i in range(0,min(len(ds["conversations"]),max_rs)):
            convo = ds["conversations"][i]
            if convo[0]["role"] != "system": #Optionally replace default system prompt. The Granite 3.1 chat template inserts a system prompt with today's date by default. 
                # If a system prompt is not needed, it will need to be manually removed from the `string' below.
                convo = [{"role":"system", "content": "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."}] + convo
            string = tokenizer.apply_chat_template(convo[:-1], tokenize=False,add_generation_prompt=False)

            # Append invocation sequence here.  
            inputs.append(string + INVOCATION_PROMPT) #"<|start_of_role|>" + convo[-1]["role"] + "<|end_of_role|>" )

            # Targets (that aLoRA is meant to learn to generate)
            targets.append(convo[-1]["content"])
        proc_dict = dict()
        proc_dict['input'] = inputs
        proc_dict['target'] = targets


        print(ds["conversations"][0])
        print(inputs[0])
        print(targets[0])

        proc_datasets.append(Dataset.from_dict(proc_dict))
    return proc_datasets

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f"{example['input'][i]}{example['target'][i]}"

        output_texts.append(text)
    return output_texts


@click.command()
@click.option('--int_name', type=click.STRING, help='dataset')
def SFT_data(int_name):

    data = get_datasets()





    if 1: #LORA:
       
   
        model_name = MODEL_NAME

        token = os.getenv("HF_MISTRAL_TOKEN")
        model_dir = model_name #os.path.join(DMF_MODEL_CACHE, model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_dir,padding_side='left',trust_remote_code=True,token=token)
        
        model_base = AutoModelForCausalLM.from_pretrained(model_dir,device_map = 'auto', use_cache=False)
        tokenizer.pad_token = tokenizer.eos_token
        model_base.config.pad_token_id = model_base.config.eos_token_id
        tokenizer.add_special_tokens = False
    datasets = process_datasets(data,tokenizer,max_rows = 400000)

    merged_dataset = concatenate_datasets(datasets)
    response_template = INVOCATION_PROMPT
    subsample_size = 40000
    merged_dataset = merged_dataset.shuffle(seed=42).select(range(min(len(merged_dataset),subsample_size)))

    print(model_base)
    
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
   
  

    peft_config = aLoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj", "v_proj"],#Can only do q, k, v layers (for now).
        #layers_to_transform=[38,39]
    )
    response_tokens = tokenizer(response_template, return_tensors="pt", add_special_tokens=False)
    response_token_ids = response_tokens['input_ids']
    peft_model = PeftModelForCausalLM(model_base, peft_config,response_token_ids = response_token_ids)
    trainer = SFTTrainer(
        peft_model,
        train_dataset=merged_dataset,
        args=SFTConfig(output_dir="/proj/dmfexp/statllm/users/kgreenewald/Thermometer/tmp",dataset_kwargs={"add_special_tokens":False},num_train_epochs=6,learning_rate=6e-7,max_seq_length = 4096,per_device_train_batch_size = 1,save_strategy="no",gradient_accumulation_steps=8,fp16=True),
        formatting_func=formatting_prompts_func,
    data_collator=collator
    #,
    )

 
    trainer.train()
    
    peft_model.save_pretrained("/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora/8bsft_alora_sz32"+ int_name)
    


if __name__ == "__main__":
   
    SFT_data()
