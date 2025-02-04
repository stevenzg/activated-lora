import torch,os, copy
from transformers import AutoTokenizer,  AutoModelForCausalLM, DynamicCache
from alora_intrinsics.alora.peft_model_alora import PeftModelForCausalLM
from alora_intrinsics.alora.config import aLoraConfig
from alora_intrinsics.alora.tokenize_alora import tokenize_alora
int_names = ["safety","certainty", "hallucination"]#"safety"
CERTAINTY_PROMPT = "<|start_of_role|>certainty<|end_of_role|>"
SAFETY_PROMPT = "<|start_of_role|>safety<|end_of_role|>"
HALL_PROMPT = "<|start_of_role|>hallucination<|end_of_role|>"



token = os.getenv("HF_MISTRAL_TOKEN")

BASE_NAME = '/proj/dmfexp/statllm/users/kgreenewald/models/granite-3.1-8b-instruct-r241212a'#"ibm-granite/granite-3.0-8b-instruct"
LORA_NAME = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora/8bsft_aloraV2_sz32"#+ int_name 


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model

tokenizer = AutoTokenizer.from_pretrained(BASE_NAME,padding_side='left',trust_remote_code=True, token=token)
model_base = AutoModelForCausalLM.from_pretrained(BASE_NAME,device_map="auto")




#response_token_ids = []
#response_templates = [SAFETY_PROMPT, CERTAINTY_PROMPT, HALL_PROMPT]
#for resp in response_templates:
#    respTok = tokenizer(resp, return_tensors="pt", add_special_tokens=False)
#    response_token_ids += respTok['input_ids']








model_alora = PeftModelForCausalLM.from_pretrained(model_base, LORA_NAME + int_names[0],adapter_name = int_names[0], response_token_ids = None)
for intname in int_names[1:]:
    model_alora.load_adapter(LORA_NAME + intname, adapter_name = intname)
model_alora.set_adapter("safety")

#if 0:
#    from huggingface_hub import notebook_login
#    notebook_login()
#    token = os.getenv("HF_MISTRAL_TOKEN")

#    model_UQ.push_to_hub("ibm-granite/granite-3.1-8b-alora-intrinsics-v0.1",private=True,token=token)
#    tokenizer.push_to_hub("ibm-granite/granite-3.1-8b-alora-intrinsics-v0.1",private=True,token=token)


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
with model_alora.disable_adapter():
    with torch.no_grad():
        prompt_cache = model_alora(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), past_key_values=prompt_cache).past_key_values 





# Generate safety exception
#input_safety = tokenizer(input_text + safety_prompt, return_tensors="pt")

input_safety, alora_offsets = tokenize_alora(tokenizer,input_text, safety_prompt)

past_key_values = copy.deepcopy(prompt_cache)
output = model_alora.generate(input_safety["input_ids"].to(device), attention_mask=input_safety["attention_mask"].to(device), use_cache=True, max_new_tokens=10, return_dict_in_generate=True, past_key_values = past_key_values, alora_offsets = alora_offsets)





output_text = tokenizer.decode(output.sequences[0])

answer = output_text.split(safety_prompt)[-1]
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
with model_alora.disable_adapter():
    output = model_alora.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=160, past_key_values = prompt_cache, return_dict_in_generate=True)
output_text = tokenizer.decode(output.sequences[0])
prompt_cache = output.past_key_values
answer = output_text.split("assistant<|end_of_role|>")[-1]
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

uq_text = tokenizer.apply_chat_template(uq_chat,tokenize=False) #+ uq_generation_prompt
#inputs = tokenizer(uq_text, return_tensors="pt")
inputs, alora_offsets = tokenize_alora(tokenizer,uq_text, uq_generation_prompt)

model_alora.set_adapter("certainty")
answer_KV = copy.deepcopy(prompt_cache)
output = model_alora.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=6, past_key_values=answer_KV,return_dict_in_generate=True, alora_offsets = alora_offsets)
output_text = tokenizer.decode(output.sequences[0])
#uq_score = int(output_text[-1])
print("Certainty: " + output_text.split("certainty<|end_of_role|>")[-1])
#print("Certainty: " + str(5 + uq_score * 10) + "%")



#Hallucination
model_alora.set_adapter("hallucination")
hall_prompt = "<|start_of_role|>hallucination<|end_of_role|>"
uq_text = tokenizer.apply_chat_template(uq_chat,tokenize=False) #+ hall_prompt
#inputs = tokenizer(uq_text, return_tensors="pt")
inputs, alora_offsets = tokenize_alora(tokenizer,uq_text, hall_prompt)
output = model_alora.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=5,past_key_values=copy.deepcopy(prompt_cache),return_dict_in_generate=True, alora_offsets = alora_offsets)
output_text = tokenizer.decode(output.sequences[0])
#uq_score = int(output_text[-1])
print("Hallucination: " + output_text.split("hallucination<|end_of_role|>")[-1])






