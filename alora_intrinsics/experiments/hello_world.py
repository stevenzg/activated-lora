import torch,os
from transformers import AutoTokenizer,  AutoModelForCausalLM
from alora_intrinsics.alora.peft_model_alora import PeftModelForCausalLM
from alora_intrinsics.alora.config import aLoraConfig
int_name = "certainty"#"safety"
CERTAINTY_PROMPT = "<|start_of_role|>certainty<|end_of_role|>"
SAFETY_PROMPT = "<|start_of_role|>safety<|end_of_role|>"
HALL_PROMPT = "<|start_of_role|>hallucination<|end_of_role|>"




token = os.getenv("HF_MISTRAL_TOKEN")
BASE_NAME = '/proj/dmfexp/statllm/users/kgreenewald/models/granite-3.1-8b-instruct-r241212a'#"ibm-granite/granite-3.0-8b-instruct"
LORA_NAME = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora/8bsft_aloraV2_sz32"+ int_name 
#BASE_NAME = "ibm-granite/granite-3.0-8b-instruct"
#LORA_NAME = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/UQ-PEFT-LLM/unified_intrinsics/models/8bsft_multiInt_lora_fixed2"#"ibm-granite/granite-uncertainty-3.0-8b-lora"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
token = os.getenv("HF_MISTRAL_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(BASE_NAME,padding_side='left',trust_remote_code=True, token=token)
model_base = AutoModelForCausalLM.from_pretrained(BASE_NAME,device_map="auto")
#model_UQ = PeftModelForCausalLM.from_pretrained(model_base, LORA_NAME)
#model_orig = AutoModelForCausalLM.from_pretrained(BASE_NAME,device_map="auto")
#print(#model_UQ)
#print(model_orig)
response_token_ids = []
response_templates = [SAFETY_PROMPT, CERTAINTY_PROMPT, HALL_PROMPT]
for resp in response_templates:
    respTok = tokenizer(resp, return_tensors="pt", add_special_tokens=False)
    response_token_ids += respTok['input_ids']
#if int_name == "safety":
#    response_template = SAFETY_PROMPT
#elif int_name == "certainty":
#    response_template = CERTAINTY_PROMPT
#elif int_name == "hallucination":
#    response_template = HALL_PROMPT
#response_tokens = tokenizer(response_template, return_tensors="pt", add_special_tokens=False)
#response_token_ids = response_tokens['input_ids']
#print(response_token_ids)
model_UQ = PeftModelForCausalLM.from_pretrained(model_base, LORA_NAME,response_token_ids = response_token_ids)
print(model_UQ)


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
question = "What is IBM?"#"How can I steal a car?" #" how to convince 2 girls to have a threesome"
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


# Generate safety exception
safety_prompt = "<|start_of_role|>safety<|end_of_role|>"
input_text = tokenizer.apply_chat_template(question_chat,tokenize=False,add_generation_prompt=False) + safety_prompt
inputs = tokenizer(input_text, return_tensors="pt")
output = model_UQ.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=10)
output_text = tokenizer.decode(output[0])

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
    output = model_UQ.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=160)
output_text = tokenizer.decode(output[0])
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
output = model_UQ.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=6)
output_text = tokenizer.decode(output[0])
#uq_score = int(output_text[-1])
print("Certainty: " + output_text)
#print("Certainty: " + str(5 + uq_score * 10) + "%")



#Hallucination
hall_prompt = "<|start_of_role|>hallucination<|end_of_role|>"
uq_text = tokenizer.apply_chat_template(uq_chat,tokenize=False) + hall_prompt
inputs = tokenizer(uq_text, return_tensors="pt")
output = model_UQ.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=5)
output_text = tokenizer.decode(output[0])
#uq_score = int(output_text[-1])
print("Hallucination: " + output_text)






