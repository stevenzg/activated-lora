import torch,os
from transformers import AutoTokenizer,  AutoModelForCausalLM, DynamicCache
from alora.peft_model_alora import aLoRAPeftModelForCausalLM
from alora.config import aLoraConfig
from alora.tokenize_alora import tokenize_alora

REUSE_CACHE = True  # If True, demonstrate KV cache reuse (slightly more complex code). If False, use simplest generation code.

token = os.getenv("HF_MISTRAL_TOKEN")
BASE_NAME = "ibm-granite/granite-3.2-8b-instruct"
ALORA_NAME = "ibm-granite/granite-3.2-8b-alora-uncertainty"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
tokenizer = AutoTokenizer.from_pretrained(BASE_NAME,padding_side='left',trust_remote_code=True, token=token)
model_base = AutoModelForCausalLM.from_pretrained(BASE_NAME,device_map="auto")
model_UQ = aLoRAPeftModelForCausalLM.from_pretrained(model_base, ALORA_NAME)

question = "What is IBM Research?"
print("Question:" + question)
question_chat = [
    {
        "role": "user",
        "content": question
    },
]

# Generate answer with base model
input_text = tokenizer.apply_chat_template(question_chat,tokenize=False,add_generation_prompt=True)
# Remove default system prompt (Granite chat template)
len_sys = len(input_text.split("<|start_of_role|>user")[0])
input_text = input_text[len_sys:]

#tokenize
inputs = tokenizer(input_text, return_tensors="pt")
if REUSE_CACHE: #save KV cache for future aLoRA call
    prompt_cache = DynamicCache()
    with model_UQ.disable_adapter():
        output_dict = model_base.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=600,past_key_values = prompt_cache, return_dict_in_generate=True)
    answer_cache = output_dict.past_key_values
    output = output_dict.sequences
else: #simplest call
    with model_UQ.disable_adapter():
        output = model_UQ.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=600)
output_text = tokenizer.decode(output[0])
# Base model answer (split uses Granite chat template)
answer = output_text.split("assistant<|end_of_role|>")[-1].split("<|end_of_text|>")[0]
print("Answer: " + answer)

# Generate certainty score
#Get Invocation string to append to input.
uq_generation_prompt = model_UQ.peft_config[model_UQ.active_adapter].invocation_string #For UQ, this is "<|start_of_role|>certainty<|end_of_role|>" 
uq_chat = question_chat + [
    {
        "role": "assistant",
        "content": answer
    },
]

uq_text = tokenizer.apply_chat_template(uq_chat,tokenize=False) 
uq_text = uq_text[len_sys:]
# tokenize and generate
inputs, alora_offsets = tokenize_alora(tokenizer,uq_text, uq_generation_prompt)

if REUSE_CACHE: #reuse KV cache from earlier answer generation 
    output = model_UQ.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=1,alora_offsets=alora_offsets,past_key_values=answer_cache)
else: #simplest call
    output = model_UQ.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=1,alora_offsets=alora_offsets)
output_text = tokenizer.decode(output[0])

# Extract score
uq_score = int(output_text[-1])
print("Certainty: " + str(5 + uq_score * 10) + "%")
