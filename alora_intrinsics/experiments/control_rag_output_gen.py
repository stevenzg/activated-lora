import click, os, torch
import numpy as np
#from lakehouse.assets.config import DMF_MODEL_CACHE
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from alora_intrinsics.alora.tokenize_alora import tokenize_alora
from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
#alora
from alora_intrinsics.alora.peft_model_alora import PeftModelForCausalLM as aLoRAPeftModelForCausalLM
from alora_intrinsics.alora.config import aLoraConfig
# standard lora
from peft import PeftModelForCausalLM, LoraConfig
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

peft_type = 'LoRA' #'LoRA' #'aLoRA', 'base'

DATA_PATH = os.getenv("HF_DATASETS_CACHE")
BASE_NAME = "ibm-granite/granite-3.2-8b-instruct"

INVOCATION_PROMPT = "<|start_of_role|>"
#Complete set of possible start sequences (for completion-only collator)
INVOCATION_PROMPT_SET = ['<|start_of_role|>assistant {"length": "'+ lngth + '"}<|end_of_role|>' for lngth in ["long","medium","short"]]
# SAFETY_PROMPT = "<|start_of_role|>safety<|end_of_role|>"
# HALL_PROMPT = "<|start_of_role|>hallucination<|end_of_role|>"
DATASET_PATH = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/alora-intrinsics/data/chat_template_dump_with_controls_0.4"
DATASET_FILES = ["test_raft.jsonl", "test_raft.jsonl", "test_raft.jsonl"]

int_names = ["controlrag"]
#int_names = ["ragControl"]
prefix = 'mar19_1' #'mar10_555'
if peft_type == 'aLoRA' or peft_type == 'base':
    LORA_NAME = f"/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora/{prefix}_8bsft_Control_alora_sz32"#_last #RAG_alora_sz32_highest"#+ int_name 
else:
    LORA_NAME = f"/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora/{prefix}_8bsft_Control_standard_lora_sz6_last" #RAG_alora_sz32_highest"#+ int_name 
output_file = f"control1024_{prefix}{peft_type}"#"output1000_base.jsonl"#alora32_3highestLR.jsonl"

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
token = os.getenv("HF_MISTRAL_TOKEN")
# Load model
tokenizer = AutoTokenizer.from_pretrained(BASE_NAME,padding_side='left',trust_remote_code=True, token=token)
model_base = AutoModelForCausalLM.from_pretrained(BASE_NAME,device_map="auto")


if peft_type == 'aLoRA':# or peft_type=="base":

    model_alora = aLoRAPeftModelForCausalLM.from_pretrained(model_base, LORA_NAME + int_names[0],adapter_name = int_names[0], response_token_ids = None) #response_token_ids)
elif peft_type == 'base':
    model_alora = model_base
else:
    model_alora = PeftModelForCausalLM.from_pretrained(model_base, LORA_NAME + int_names[0],adapter_name = int_names[0])
#for intname in int_names[1:]:
 #   model_alora.load_adapter(LORA_NAME + intname, adapter_name = intname)
if peft_type != 'base':
    model_alora.set_adapter(int_names[0])



def get_datasets():
    datasets = []
    for ds in DATASET_FILES:
        if ds[-1] == "n": #json

            file = open(DATASET_PATH + '/' + ds)
            data = json.load(file)


        else: #jsonl
            file = open(DATASET_PATH +'/' +  ds)
            data = {"conversations":[(json.loads(line)) for line in file]}#,"documents":[(json.loads(line))["documents"] for line in file]}
        datasets.append(data)
    return datasets
def process_datasets(datasets,model_alora,tokenizer,max_rows):
    proc_datasets = []
    ln_ix = -1
    for ds in datasets:
        ln_ix+= 1
        inputs = []
        targets = []
        add = ""



        max_rs = max_rows
    
        
        for i in range(0,min(len(ds["conversations"]),max_rs)):
            print(f"{i} of {len(ds['conversations'])}")
            convo = ds["conversations"][i]["messages"]
            docs = ds["conversations"][i]["documents"]
            lngth = ds["conversations"][i]["controls"]["length"]
            if convo[0]["role"] != "system": #Optionally replace default system prompt. The Granite 3.1 chat template inserts a system prompt with today's date by default. 
                # If a system prompt is not needed, it will need to be manually removed from the `string' below.
                convo = [{"role":"system", "content": ""}] +convo#"You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."}] + convo
                string = tokenizer.apply_chat_template(conversation =convo[:-1],documents=docs, tokenize=False,add_generation_prompt=False)
                string_to_remove = tokenizer.apply_chat_template(convo[0:1], tokenize=False,add_generation_prompt=False)
                string = string[len(string_to_remove):]
            else:
                string = tokenizer.apply_chat_template(conversation=convo[:-1],documents=docs, tokenize=False,add_generation_prompt=False)
            if 1: # reorder to documents last
                part1rest = string.split('<|start_of_role|>documents<|end_of_role|>')
                part23 = part1rest[1].split('<|end_of_text|>')
                string = part1rest[0] + part1rest[1][len(part23[0])+1:] + '<|start_of_role|>documents<|end_of_role|>' + part23[0] + '<|end_of_text|>'
            # Append invocation sequence here.  Doing manually to ensure consistency with data collator
            if lngth == "long":
                ix = 0
            elif lngth == "medium":
                ix = 1
            else: #short
                ix = 2
            ix = ln_ix
            if ln_ix == 0:
                ds["conversations"][i]["controls"]["length"] = "long"
            elif ln_ix == 1:
                ds["conversations"][i]["controls"]["length"] = "medium"
            else:
                ds["conversations"][i]["controls"]["length"] = "short"
            input_text= string #+ INVOCATION_PROMPT_SET[ix] #"<|start_of_role|>" + convo[-1]["role"] + "<|end_of_role|>" ) 
            
            
            # Targets (that aLoRA is meant to learn to generate)
            targets.append(convo[-1]["content"])
            
            # Generate
            input_tokenized, alora_offsets = tokenize_alora(tokenizer,input_text, INVOCATION_PROMPT_SET[ix])
            if peft_type == 'base':
                #with model_alora.disable_adapter():
                output = model_base.generate(input_tokenized["input_ids"].to(device), attention_mask=input_tokenized["attention_mask"].to(device), use_cache=True, max_new_tokens=1024, return_dict_in_generate=True)

            if peft_type == 'aLoRA':
                output = model_alora.generate(input_tokenized["input_ids"].to(device), attention_mask=input_tokenized["attention_mask"].to(device), use_cache=True, max_new_tokens=1024, return_dict_in_generate=True, alora_offsets = alora_offsets)
            else:
                output = model_alora.generate(input_tokenized["input_ids"].to(device), attention_mask=input_tokenized["attention_mask"].to(device), use_cache=True, max_new_tokens=1024, return_dict_in_generate=True)
            
            


            
            output_text = tokenizer.decode(output.sequences[0])
#            print(output_text)
            answer = output_text.split(INVOCATION_PROMPT_SET[ix])[-1]
            #record in dataset
            ds["conversations"][i]["output"] = answer
            print(answer)
            print(f"length: {lngth}")

        #proc_dict = dict()
        #proc_dict['input'] = inputs
        #proc_dict['target'] = targets


        #print(ds["conversations"][0])
        #print(inputs[0])
        #print(targets[0])

        proc_datasets.append(ds)
    return proc_datasets


datasets = get_datasets()
processed = process_datasets(datasets,model_alora,tokenizer,2000000)

# save in jsonl
# Define the output JSONL file path

output_path = DATASET_PATH + '/' +output_file

# Open the file in write mode
for ds_ix in range(len(processed)):
    lnth = processed[ds_ix]["conversations"][0]["controls"]["length"]
    with open(output_path + lnth + ".jsonl", "w", encoding="utf-8") as f:
        #for ds_ix in range(len(processed)):
        for conversation in processed[ds_ix]["conversations"]:
            f.write(json.dumps(conversation) + "\n")  # Convert dict to JSON string and write






