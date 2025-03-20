import torch,os, copy
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
import json
from transformers import AutoTokenizer,  AutoModelForCausalLM, DynamicCache
from alora_intrinsics.alora.peft_model_alora import PeftModelForCausalLM as aLoRAPeftModelForCausalLM
from peft import PeftModelForCausalLM, LoraConfig
#from alora_intrinsics.alora.config import aLoraConfig
int_names = ["constraint"]#["safety","certainty", "hallucination"]#"safety"
INVOCATION_PROMPT = "<|start_of_role|>check_constraint<|end_of_role|>"
SAFETY_PROMPT = "<|start_of_role|>safety<|end_of_role|>"
HALL_PROMPT = "<|start_of_role|>hallucination<|end_of_role|>"
#DATASET_PATH = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/UQ-PEFT-LLM/uq/data/"
#DATASET_FILES = ["uq_data_3_1.jsonl"]#,"hallucination_intrinsic_output.json", "safety-data-binary/combined_safe.jsonl", "safety-data-binary/combined_unsafe.jsonl"]

DATASET_PATH = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/alora-intrinsics/data/constraintData"#"/proj/dmfexp/statllm/users/kgreenewald/Thermometer/alora-intrinsics/data/chat_template_dump_with_controls_0.4"
DATASET_FILES = ["multivar_gen_data_2000.json",  "multivar_gen_data_2000_2.json"]

BASE_NAME = "ibm-granite/granite-3.2-8b-instruct"# '/proj/dmfexp/statllm/users/kgreenewald/models/granite-3.1-8b-instruct-r241212a'#"ibm-granite/granite-3.0-8b-instruct"
LORA_NAME = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora/mar17_1_8bsft_Constraint_standard_lora_sz6_last"#mar12_8bsft_standard_lora_sz_4"#feb6_8bsft_standard_lora_sz6_"#+ int_name 
LORA_NAME = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora/mar17_6_8bsft_Constraint_alora_sz32_last"
adapter = "LoRA"#"LoRA"



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

            
        
    
def process_datasets(datasets,model_UQ,tokenizer,max_rows):
    performance = [] 
    flag = 1
    for ds in datasets:
        #inputs = []
        #targets = []
        add = ""

        error = 0
        total = 0
        stp = 1
        max_rs = max_rows
        
        
       
      
     
    
        for i in range(0,min(len(ds),max_rs),stp):
            total += 2
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



            inputs=(string + INVOCATION_PROMPT) #"<|start_of_role|>" + convo[-1]["role"] + "<|end_of_role|>" )
            #print(inputs)
            # Targets (that aLoRA is meant to learn to generate)
            targets ="Y"# + '<|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|>')
            last_prompt = "<|start_of_role|>" + "check_constraint" + "<|end_of_role|>"

            inputT = tokenizer(inputs, return_tensors="pt")
            output = model_UQ.generate(inputT["input_ids"].to(device), attention_mask=inputT["attention_mask"].to(device), max_new_tokens=4)
            output_text = tokenizer.decode(output[0])
            answer = output_text.split(last_prompt)[1]
            print("Intrinsic_output: " + answer + "Ground truth: " + targets[0])
            answer = answer[0]
            #targets = targets
            if answer != targets:
                error += 1


            


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
            inputs = (string + INVOCATION_PROMPT) #"<|start_of_role|>" + convo[-1]["role"] + "<|end_of_role|>" )

            # Targets (that aLoRA is meant to learn to generate)
            targets = "N"
 


            #if convo[0]["role"] != "system":
             #   convo = [{"role":"system", "content": "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."}] + convo
           # string = tokenizer.apply_chat_template(convo[:-1], tokenize=False,add_generation_prompt=False)
       
            inputT = tokenizer(inputs, return_tensors="pt")
            output = model_UQ.generate(inputT["input_ids"].to(device), attention_mask=inputT["attention_mask"].to(device), max_new_tokens=3)
            output_text = tokenizer.decode(output[0])
            answer = output_text.split(last_prompt)[1]
            print("Intrinsic_output: " + answer + "Ground truth: " + targets[0])
            answer = answer[0] 
            #targets = targets
            if answer != targets:
                error += 1

        rate = error/total
        performance.append(rate)
     


    
   
  

 
    return performance







token = os.getenv("HF_MISTRAL_TOKEN")
#BASE_NAME = "ibm-granite/granite-3.2-8b-instruct"# '/proj/dmfexp/statllm/users/kgreenewald/models/granite-3.1-8b-instruct-r241212a'#"ibm-granite/granite-3.0-8b-instruct"
#LORA_NAME = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora/mar17_1_8bsft_Constraint_standard_lora_sz6_last"#mar12_8bsft_standard_lora_sz_4"#feb6_8bsft_standard_lora_sz6_"#+ int_name 
#BASE_NAME = "ibm-granite/granite-3.0-8b-instruct"
#LORA_NAME = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/UQ-PEFT-LLM/unified_intrinsics/models/8bsft_multiInt_lora_fixed2"#"ibm-granite/granite-uncertainty-3.0-8b-lora"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
token = os.getenv("HF_MISTRAL_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(BASE_NAME,padding_side='left',trust_remote_code=True, token=token)
model_base = AutoModelForCausalLM.from_pretrained(BASE_NAME,device_map="auto")




response_token_ids = []
response_templates = [INVOCATION_PROMPT]#[SAFETY_PROMPT, CERTAINTY_PROMPT, HALL_PROMPT]
for resp in response_templates:
    respTok = tokenizer(resp, return_tensors="pt", add_special_tokens=False)
    response_token_ids += respTok['input_ids']







if adapter == "LoRA":
    model_UQ = PeftModelForCausalLM.from_pretrained(model_base, LORA_NAME + int_names[0],adapter_name = int_names[0])#, response_token_ids = response_token_ids)
else:
    model_UQ = aLoRAPeftModelForCausalLM.from_pretrained(model_base, LORA_NAME + int_names[0],adapter_name = int_names[0], response_token_ids = response_token_ids)
#for intname in int_names[1:]:
#    model_UQ.load_adapter(LORA_NAME + intname, adapter_name = intname)
model_UQ.set_adapter(int_names[0])






datasets = get_datasets()
performance = process_datasets(datasets,model_UQ,tokenizer,100)

print(performance)




