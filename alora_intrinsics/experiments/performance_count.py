import torch,os, copy
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
import json
from transformers import AutoTokenizer,  AutoModelForCausalLM, DynamicCache
from alora_intrinsics.alora.peft_model_alora import PeftModelForCausalLM
from alora_intrinsics.alora.config import aLoraConfig
int_names = ["certainty"] #["safety","certainty", "hallucination"]#"safety"
CERTAINTY_PROMPT = "<|start_of_role|>certainty<|end_of_role|>"
SAFETY_PROMPT = "<|start_of_role|>safety<|end_of_role|>"
HALL_PROMPT = "<|start_of_role|>hallucination<|end_of_role|>"
DATASET_PATH = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/UQ-PEFT-LLM/uq/data/"
DATASET_FILES = ["uq_data_3_1.jsonl"]#,"hallucination_intrinsic_output.json", "safety-data-binary/combined_safe.jsonl", "safety-data-binary/combined_unsafe.jsonl"]


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
        if ds["conversations"][0][-1]["role"] == "certainty":
            stp = 200
            max_rs = max_rows * stp
            flag = 0
            #add = "5%"#"5%<|end_of_text|>"
            model_UQ.set_adapter("certainty")
            print(f'Total rows: {len(ds["conversations"])}, max rows: {max_rs}')
        elif ds["conversations"][0][-1]["role"] == "SAFETY_EXCEPTION":
            model_UQ.set_adapter("safety")
        else:
            hi = 0
            model_UQ.set_adapter("hallucination")
        for i in range(0,min(len(ds["conversations"]),max_rs),stp):
            total += 1
            convo = ds["conversations"][i]
            if convo[0]["role"] != "system":
                convo = [{"role":"system", "content": "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."}] + convo
            if 1:
                string = tokenizer.apply_chat_template(convo[:-1], tokenize=False,add_generation_prompt=False)
            else:
                string = tokenizer.apply_chat_template(convo[:-2], tokenize=False,add_generation_prompt=False)
            string_to_remove = tokenizer.apply_chat_template(convo[0:1], tokenize=False,add_generation_prompt=False)
            string = string[len(string_to_remove):]
            if convo[-1]["role"] == "Hallucination_tag":
                convo[-1]["role"] = "hallucination"
                if convo[-1]["content"] == "1":
                    convo[-1]["content"] = "Y"
                else:
                    convo[-1]["content"] = "N"
                    #if np.random.rand() < .5:
                     #   continue
            if convo[-1]["role"] == "SAFETY_EXCEPTION":
                convo[-1]["role"]= "safety"
                if convo[-1]["content"] == "1":
                    convo[-1]["content"] = "No, this is unsafe!"
                else:
                    convo[-1]["content"] = "Yes, this is a safe prompt."
            last_prompt = "<|start_of_role|>" + convo[-1]["role"] + "<|end_of_role|>"
            inputs = (string + "<|start_of_role|>" + convo[-1]["role"] + "<|end_of_role|>" )
            targets = (convo[-1]["content"]+add)
       
            inputT = tokenizer(inputs, return_tensors="pt")
            output = model_UQ.generate(inputT["input_ids"].to(device), attention_mask=inputT["attention_mask"].to(device), max_new_tokens=1)
            output_text = tokenizer.decode(output[0])
            answer = output_text.split(last_prompt)[1]
            print("Intrinsic_output: " + answer[0] + "Ground truth: " + targets[0])
            answer = answer[0] 
            targets = targets[0]
            if convo[-1]["role"] == "certainty" and (answer != targets):
                try:
                    error += abs(int(answer) - int(targets))
                except:
                    error += 10
            elif answer != targets:
                error += 1

        rate = error/total
        performance.append(rate)
     


    
   
  

 
    return performance







token = os.getenv("HF_MISTRAL_TOKEN")
BASE_NAME = "ibm-granite/granite-3.2-8b-instruct"#'/proj/dmfexp/statllm/users/kgreenewald/models/granite-3.1-8b-instruct-r241212a'#"ibm-granite/granite-3.0-8b-instruct"
LORA_NAME = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora/PREPOSTmar10_8bsft_alora_sz32_"#8bsft_aloraV2_sz32"#+ int_name 
#BASE_NAME = "ibm-granite/granite-3.0-8b-instruct"
#LORA_NAME = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/UQ-PEFT-LLM/unified_intrinsics/models/8bsft_multiInt_lora_fixed2"#"ibm-granite/granite-uncertainty-3.0-8b-lora"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
token = os.getenv("HF_MISTRAL_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(BASE_NAME,padding_side='left',trust_remote_code=True, token=token)
model_base = AutoModelForCausalLM.from_pretrained(BASE_NAME,device_map="auto")




response_token_ids = []
response_templates = [SAFETY_PROMPT, CERTAINTY_PROMPT, HALL_PROMPT]
for resp in response_templates:
    respTok = tokenizer(resp, return_tensors="pt", add_special_tokens=False)
    response_token_ids += respTok['input_ids']








model_UQ = PeftModelForCausalLM.from_pretrained(model_base, LORA_NAME + int_names[0],adapter_name = int_names[0], response_token_ids = response_token_ids)
for intname in int_names[1:]:
    model_UQ.load_adapter(LORA_NAME + intname, adapter_name = intname)
model_UQ.set_adapter(int_names[0])






datasets = get_datasets()
performance = process_datasets(datasets,model_UQ,tokenizer,200)

print(performance)





import sys
sys.exit(0)



if 0:
    from huggingface_hub import notebook_login
    notebook_login()
#token = os.getenv("HF_MISTRAL_TOKEN")
    #model_UQ.merge_and_unload()
    model_UQ.push_to_hub("ibm-granite/granite-3.1-8b-alora-intrinsics-v0.1",private=True,token=token)
    tokenizer.push_to_hub("ibm-granite/granite-3.1-8b-alora-intrinsics-v0.1",private=True,token=token)


system_prompt = "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior." 


#################################3
##################################
question = "What is IBM?"#"How can I steal a car?" "
#################################
#################################




print("Question:" + question)
question_chat = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": question
    },
]


# Process question
safety_prompt = "<|start_of_role|>safety<|end_of_role|>"
prompt_cache = DynamicCache()
input_text = tokenizer.apply_chat_template(question_chat,tokenize=False,add_generation_prompt=False) #+ safety_prompt
inputs = tokenizer(input_text, return_tensors="pt")
with model_UQ.disable_adapter():
    with torch.no_grad():
        prompt_cache = model_UQ(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), past_key_values=prompt_cache).past_key_values 





# Generate safety exception
input_safety = tokenizer(input_text + safety_prompt, return_tensors="pt")

past_key_values = copy.deepcopy(prompt_cache)
output = model_UQ.generate(input_safety["input_ids"].to(device), attention_mask=input_safety["attention_mask"].to(device), use_cache=True, max_new_tokens=10, return_dict_in_generate=True, past_key_values = past_key_values)





output_text = tokenizer.decode(output.sequences[0])

answer = output_text.split(safety_prompt)[1]
print("Safety: " + answer)

question_chat = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": question
    },
]

# Generate answer
input_text = tokenizer.apply_chat_template(question_chat,tokenize=False,add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt")
with model_UQ.disable_adapter():
    output = model_UQ.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=160, past_key_values = prompt_cache, return_dict_in_generate=True)
output_text = tokenizer.decode(output.sequences[0])
prompt_cache = output.past_key_values
answer = output_text.split("assistant<|end_of_role|>")[1]
print("Answer: " + answer)

# Generate certainty score
uq_generation_prompt = "<|start_of_role|>certainty<|end_of_role|>"
uq_chat = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": question
    },
    {
        "role": "assistant",
        "content": answer
    },
]

uq_text = tokenizer.apply_chat_template(uq_chat,tokenize=False) + uq_generation_prompt
inputs = tokenizer(uq_text, return_tensors="pt")
model_UQ.set_adapter("certainty")
answer_KV = copy.deepcopy(prompt_cache)
output = model_UQ.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=6, past_key_values=answer_KV,return_dict_in_generate=True)
output_text = tokenizer.decode(output.sequences[0])
#uq_score = int(output_text[-1])
print("Certainty: " + output_text.split("certainty<|end_of_role|>")[1])
#print("Certainty: " + str(5 + uq_score * 10) + "%")



#Hallucination
hall_prompt = "<|start_of_role|>hallucination<|end_of_role|>"
uq_text = tokenizer.apply_chat_template(uq_chat,tokenize=False) + hall_prompt
inputs = tokenizer(uq_text, return_tensors="pt")
output = model_UQ.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=5,past_key_values=copy.deepcopy(prompt_cache),return_dict_in_generate=True)
output_text = tokenizer.decode(output.sequences[0])
#uq_score = int(output_text[-1])
print("Hallucination: " + output_text.split("hallucination<|end_of_role|>")[1])






