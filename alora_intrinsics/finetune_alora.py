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
MODEL_NAME = "/proj/dmfexp/statllm/users/kgreenewald/models/granite-3.1-8b-instruct-r241212a"#"ibm-granite/granite-3.0-8b-instruct"


CERTAINTY_PROMPT = "<|start_of_role|>certainty<|end_of_role|>"
SAFETY_PROMPT = "<|start_of_role|>safety<|end_of_role|>"
HALL_PROMPT = "<|start_of_role|>hallucination<|end_of_role|>"
DATASET_PATH = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/UQ-PEFT-LLM/uq/data/"
DATASET_FILES = ["uq_data_3_1.jsonl","hallucination_intrinsic_output.json", "safety-data-binary/combined_safe.jsonl", "safety-data-binary/combined_unsafe.jsonl"]


def get_datasets():
    datasets = []
    for ds in DATASET_FILES:
        if ds[-1] == "n": #json

            file = open(DATASET_PATH + ds)
            data = json.load(file)


        else:
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
        if ds["conversations"][0][-1]["role"] == "certainty":
            max_rs = max_rows * 8

            add = "5%"#"5%<|end_of_text|>"
            print(f'Total rows: {len(ds["conversations"])}, max rows: {max_rs}')
        
        for i in range(200,min(len(ds["conversations"]),max_rs)):
            convo = ds["conversations"][i]
            if convo[0]["role"] != "system":
                convo = [{"role":"system", "content":""}] + convo# "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."}] + convo
                string = tokenizer.apply_chat_template(convo[:-1], tokenize=False,add_generation_prompt=False)
                string_to_remove = tokenizer.apply_chat_template(convo[0:1], tokenize=False,add_generation_prompt=False)
                string = string[len(string_to_remove):]
                print('hi')
                print(string)
            else:
                string = tokenizer.apply_chat_template(convo[:-1], tokenize=False,add_generation_prompt=False)
            if convo[-1]["role"] == "Hallucination_tag":
                convo[-1]["role"] = "hallucination"
                if convo[-1]["content"] == "1":
                    convo[-1]["content"] = "Y"
                else:
                    convo[-1]["content"] = "N"


            if convo[-1]["role"] == "SAFETY_EXCEPTION":
                convo[-1]["role"]= "safety"
                if convo[-1]["content"] == "1":
                    convo[-1]["content"] = "No, this is unsafe!"
                else:
                    convo[-1]["content"] = "Yes, this is a safe prompt."
            inputs.append(string + "<|start_of_role|>" + convo[-1]["role"] + "<|end_of_role|>" )
            targets.append(convo[-1]["content"]+'<|end_of_text|>')
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
    if int_name == "certainty":
        merged_dataset = datasets[0]
        response_template = CERTAINTY_PROMPT
    elif int_name =="hallucination":
        merged_dataset = datasets[1]
        response_template = HALL_PROMPT
    else:
        merged_dataset = concatenate_datasets(datasets[2:4])
        response_template = SAFETY_PROMPT
    subsample_size = 200000
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
    
    peft_model.save_pretrained("/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora/8bsft_aloraV2_sz32"+ int_name)
    


if __name__ == "__main__":
   
    SFT_data()
