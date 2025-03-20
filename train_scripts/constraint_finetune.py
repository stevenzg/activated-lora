import click, os, torch
import numpy as np
#from lakehouse.assets.config import DMF_MODEL_CACHE
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets

from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
#alora
from alora_intrinsics.alora.peft_model_alora import PeftModelForCausalLM as aLoRAPeftModelForCausalLM
from alora_intrinsics.alora.config import aLoraConfig
# standard lora
from peft import PeftModelForCausalLM, LoraConfig
import json
from alora_intrinsics.alora.multi_collator import DataCollatorForCompletionOnlyLM_Multi
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TrainerCallback


DATA_PATH = os.getenv("HF_DATASETS_CACHE")
MODEL_NAME = "ibm-granite/granite-3.2-8b-instruct"

#Universal start sequence (to turn on aLoRA)
INVOCATION_PROMPT = "<|start_of_role|>check_constraint<|end_of_role|>"
#Complete set of possible start sequences (for completion-only collator)
#INVOCATION_PROMPT_SET = ['<|start_of_role|>assistant {"length": "'+ lngth + '"}<|end_of_role|>' for lngth in ["long","medium","short"]]
# SAFETY_PROMPT = "<|start_of_role|>safety<|end_of_role|>"
# HALL_PROMPT = "<|start_of_role|>hallucination<|end_of_role|>"
DATASET_PATH = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/alora-intrinsics/data/constraintData"#"/proj/dmfexp/statllm/users/kgreenewald/Thermometer/alora-intrinsics/data/chat_template_dump_with_controls_0.4"
DATASET_FILES = ["multivar_gen_data_2000.json",  "multivar_gen_data_2000_2.json"]#["train_raft.jsonl", "val_raft.jsonl"]
SAVE_PATH = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora"


import json
import re

def extract_fields(output_text):
    """
    Extracts the following fields from the output_text:
      - user_request
      - user_constraint
      - compliant_assistant_generation
      - uncompliant_assistant_generation
    Returns a dictionary with the fields if all are found; otherwise, returns None.
    """
    # Define a regex pattern capturing each field.
    # The pattern assumes the fields appear in order.
    pattern = re.compile(
        r"user_request:\s*(?P<user_request>.*?)\s*(?=user_constraint:)"
        r"user_constraint:\s*(?P<user_constraint>.*?)\s*(?=compliant_assistant_generation:)"
        r"compliant_assistant_generation:\s*(?P<compliant_assistant_generation>.*?)\s*(?=uncompliant_assistant_generation:)"
        r"uncompliant_assistant_generation:\s*(?P<uncompliant_assistant_generation>.*)",
        re.DOTALL
    )

    match = pattern.search(output_text)
    if match:
        # Use the first occurrence (groupdict returns first match per group)
        return {key: value.strip() for key, value in match.groupdict().items()}
    return None

def process_json_file(file_path):
    """
    Loads a JSON file from file_path. For each row, extracts the desired fields
    from the 'output' key using extract_fields(). Only rows where all four fields are found are included.
    Returns a list of dictionaries.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    for row in data:
        output_text = row.get("output", "")
        fields = extract_fields(output_text)
        # Only include if all fields are present (and non-empty)
        if fields and all(fields.values()):
            results.append(fields)
    return results


def get_datasets():
    datasets = []
    for ds in DATASET_FILES:
        if ds[-1] == "n": #json

            #file = open(DATASET_PATH + '/' + ds)
            data = process_json_file(DATASET_PATH + '/' + ds)
            

       # else: #jsonl
       #     file = open(DATASET_PATH +'/' +  ds)
       #     data = {"conversations":[(json.loads(line)) for line in file]}#,"documents":[(json.loads(line))["documents"] for line in file]}
        datasets.append(data)
    return datasets
def process_datasets(datasets,tokenizer,max_rows):
    proc_datasets = []

    for ds in datasets:
        inputs = []
        targets = []
        add = ""



        max_rs = max_rows
    
        
        for i in range(100,min(len(ds),max_rs)):
            question = ds[i]["user_request"]
            constraint = ds[i]["user_constraint"]
            pos_gen = ds[i]["compliant_assistant_generation"]
            neg_gen = ds[i]["uncompliant_assistant_generation"]
            
            convo_pos = [{
                "role": "user",
                "content": question + "\nConstraint: " + constraint,
                },
                {
                    "role": "assistant",
                    "content": pos_gen,
                }]
            convo_neg = [{
                "role": "user",
                "content": question + "\nConstraint: " + constraint,
                },
                {
                    "role": "assistant",
                    "content": neg_gen,
                }]
            #convo = ds["conversations"][i]["messages"]
            #docs = ds["conversations"][i]["documents"]
            #lngth = ds["conversations"][i]["controls"]["length"]
            #if convo[0]["role"] != "system": #Optionally replace default system prompt. The Granite 3.1 chat template inserts a system prompt with today's date by default. 
                # If a system prompt is not needed, it will need to be manually removed from the `string' below.
            convo = [{"role":"system", "content": ""}] +convo_pos#"You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."}] + convo
            string = tokenizer.apply_chat_template(conversation =convo, tokenize=False,add_generation_prompt=False)
            string_to_remove = tokenizer.apply_chat_template(convo[0:1], tokenize=False,add_generation_prompt=False)
            string = string[len(string_to_remove):]
            #else:
            #    string = tokenizer.apply_chat_template(conversation=convo[:-1],documents=docs, tokenize=False,add_generation_prompt=False)
            #    part1rest = string.split('<|start_of_role|>documents<|end_of_role|>')
            #    part23 = part1rest[1].split('<|end_of_text|>')
            #    string = part1rest[0] + part1rest[1][len(part23[0])+1:] + '<|start_of_role|>documents<|end_of_role|>' + part23[0] + '<|end_of_text|>'
                #print(string)
                #print(docstr)
            # Append invocation sequence here.  Doing manually to ensure consistency with data collator
            #if lngth == "long":
            #    ix = 0
            #elif lngth == "medium":
            #    ix = 1
            #else: #short
            #    ix = 2
            inputs.append(string + INVOCATION_PROMPT) #"<|start_of_role|>" + convo[-1]["role"] + "<|end_of_role|>" )

            # Targets (that aLoRA is meant to learn to generate)
            targets.append("Y")# + '<|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|>')
            convo = [{"role":"system", "content": ""}] +convo_neg#"You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."}] + convo
            string = tokenizer.apply_chat_template(conversation =convo, tokenize=False,add_generation_prompt=False)
            string_to_remove = tokenizer.apply_chat_template(convo[0:1], tokenize=False,add_generation_prompt=False)
            string = string[len(string_to_remove):]
            #else:
            #    string = tokenizer.apply_chat_template(conversation=convo[:-1],documents=docs, tokenize=False,add_generation_prompt=False)
            #    part1rest = string.split('<|start_of_role|>documents<|end_of_role|>')
            #    part23 = part1rest[1].split('<|end_of_text|>')
            #    string = part1rest[0] + part1rest[1][len(part23[0])+1:] + '<|start_of_role|>documents<|end_of_role|>' + part23[0] + '<|end_of_text|>'
                #print(string)
                #print(docstr)
            # Append invocation sequence here.  Doing manually to ensure consistency with data collator
            #if lngth == "long":
            #    ix = 0
            #elif lngth == "medium":
            #    ix = 1
            #else: #short
            #    ix = 2
            inputs.append(string + INVOCATION_PROMPT) #"<|start_of_role|>" + convo[-1]["role"] + "<|end_of_role|>" )

            # Targets (that aLoRA is meant to learn to generate)
            targets.append("N")
        proc_dict = dict()
        proc_dict['input'] = inputs
        proc_dict['target'] = targets


        #print(ds["conversations"][0])
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
from transformers import TrainerCallback

#class SaveModelCallback(TrainerCallback):
#    def on_save(self, args, state, control, **kwargs):
#        """Ensure tied weights are assigned before saving."""
#        trainer = kwargs["trainer"]
#        if getattr(trainer.model, "tie_weights", None):
#            trainer.model.tie_weights()  # Fix weight sharing before savingi
class SaveBestModelCallback(TrainerCallback):
    def __init__(self):
        self.best_eval_loss = float("inf")  # Track best loss

    def on_evaluate(self, args, state, control, **kwargs):
        """Save the best model manually during evaluation."""

        model = kwargs["model"]
        metrics = kwargs["metrics"]
        
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss  # Update best loss
          

            # Ensure tied weights are applied before saving
            #if getattr(trainer.model, "tie_weights", None):
            #    trainer.model.tie_weights()

            # Manually save best model
            model.save_pretrained(args.output_dir)


@click.command()
@click.option('--adapter', type=click.STRING, help='adapter, LoRA or aLoRA')
@click.option('--int_name', type=click.STRING, help='dataset')
def SFT_data(int_name,adapter):

    data = get_datasets()






       
   
    model_name = MODEL_NAME

    token = os.getenv("HF_MISTRAL_TOKEN")
    model_dir = model_name #os.path.join(DMF_MODEL_CACHE, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir,padding_side='right',trust_remote_code=True,token=token)
        
    model_base = AutoModelForCausalLM.from_pretrained(model_dir,device_map = 'auto', use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token
#    model_base.config.pad_token_id = model_base.config.eos_token_id
    tokenizer.add_special_tokens = False
    datasets = process_datasets(data,tokenizer,max_rows = 400000)

    joint_dataset = concatenate_datasets(datasets[0:2])#train_dataset
    subsample_size = (len(joint_dataset)//5)*4
    print(subsample_size)
    train_dataset = joint_dataset.shuffle(seed=42).select(range(min(len(joint_dataset),subsample_size)))
    val_dataset = joint_dataset.shuffle(seed=42).select(range(min(len(joint_dataset),subsample_size), len(joint_dataset)))
    print(model_base)
    
    collator = DataCollatorForCompletionOnlyLM(INVOCATION_PROMPT, tokenizer=tokenizer)
    
    prefix = "mar17_7"
    if adapter != 'LoRA': # aLoRA model
        peft_config = aLoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj", "v_proj"],#Can only do q, k, v layers (for now).
            #layers_to_transform=[38,39]
        )
        response_tokens = tokenizer(INVOCATION_PROMPT, return_tensors="pt", add_special_tokens=False)
        response_token_ids = response_tokens['input_ids']
        peft_model = aLoRAPeftModelForCausalLM(model_base, peft_config,response_token_ids = response_token_ids)
        #tmp_dir = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/tmp"
        sft_args = SFTConfig(output_dir=SAVE_PATH + f"/{prefix}_8bsft_Constraint_alora_sz32"+ int_name,
                evaluation_strategy = "steps",
                eval_steps=300,
                dataset_kwargs={"add_special_tokens":False},num_train_epochs=6,learning_rate=6e-7*5*10/5*2,max_seq_length = 1024,per_device_train_batch_size = 2,save_strategy="no",gradient_accumulation_steps=4,fp16=True)
        trainer = SFTTrainer(
            peft_model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=sft_args,
            formatting_func=formatting_prompts_func,
        data_collator=collator,
        callbacks=[SaveBestModelCallback()]
        #,
        )
        trainer.train()
        #load from best
        #peft_best = aLoRAPeftModelForCausalLM.from_pretrained(model_base,tmp_dir + '/adapter')
        peft_model.save_pretrained(SAVE_PATH + f"/{prefix}_8bsft_Constraint_alora_sz32_last"+ int_name)



        #####################################################################
        #####################################################################
    else: #standard LoRA. THESE HYPERPARAMETERS ARE NOT TUNED
        peft_config = LoraConfig(
            r=6,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj",  "v_proj"],
            #layers_to_transform=[38,39]
        )
        peft_model = PeftModelForCausalLM(model_base, peft_config)
        
        #tmp_dir = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/tmp"
        sft_args = SFTConfig(output_dir=SAVE_PATH + f"/{prefix}_8bsft_Constraint_standard_lora_sz6"+ int_name,
           #     evaluation_strategy = "steps",
          #      eval_steps=300,
                dataset_kwargs={"add_special_tokens":False},num_train_epochs=3,learning_rate=1e-5,max_seq_length = 1024,per_device_train_batch_size = 2,save_strategy="no",gradient_accumulation_steps=2,fp16=True)
        trainer = SFTTrainer(
            peft_model,
            train_dataset=train_dataset,
         #   eval_dataset=val_dataset,
            args=sft_args,
            formatting_func=formatting_prompts_func,
        data_collator=collator,
        #callbacks=[SaveBestModelCallback()]
        #,
        )
        trainer.train()
        #load from best
        #peft_best = PeftModelForCausalLM.from_pretrained(model_base,tmp_dir + '/adapter')







       # trainer = SFTTrainer(
           # peft_model,
          #  train_dataset=merged_dataset,
         #   args=SFTConfig(output_dir="/proj/dmfexp/statllm/users/kgreenewald/Thermometer/tmp",dataset_kwargs={"add_special_tokens":False},num_train_epochs=1,learning_rate=6e-7,max_seq_length = 4096,per_device_train_batch_size = 1,save_strategy="no",gradient_accumulation_steps=8,fp16=True),
        #    formatting_func=formatting_prompts_func,
        #data_collator=collator
        #,
        #)
        #trainer.train()
    
        peft_model.save_pretrained(SAVE_PATH + f"/{prefix}_8bsft_Constraint_standard_lora_sz6_last"+ int_name)
        

 
    
    


if __name__ == "__main__":
   
    SFT_data()
