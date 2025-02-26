import click, os, torch
import numpy as np
#from lakehouse.assets.config import DMF_MODEL_CACHE
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets

from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from alora_intrinsics.alora.peft_model_alora import PeftModelForCausalLM as aLoRAPeftModelForCausalLM
from alora_intrinsics.alora.config import aLoraConfig
# standard lora
from peft import PeftModelForCausalLM, LoraConfig
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig



DATA_PATH = os.getenv("HF_DATASETS_CACHE")
MODEL_NAME = "ibm-granite/granite-3.1-8b-instruct"

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
        spans = []
        add = ""



        max_rs = max_rows
        if ds["conversations"][0][-1]["role"] == "certainty":
            max_rs = max_rows * 8

            add = "5%<|end_of_text|>"#"5%<|end_of_text|>"
            print(f'Total rows: {len(ds["conversations"])}, max rows: {max_rs}')
        
        for i in range(200,min(len(ds["conversations"]),max_rs)):
            convo = ds["conversations"][i]
            if convo[0]["role"] != "system":
                convo = [{"role":"system", "content":""}] + convo# "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."}] + convo
                #string = tokenizer.apply_chat_template(convo[:-1], tokenize=False,add_generation_prompt=False)
                
                #print('hi')
                #print(string)
            #else:
            string = tokenizer.apply_chat_template(convo[:-1], tokenize=False,add_generation_prompt=False)
            string_to_remove = tokenizer.apply_chat_template(convo[0:1], tokenize=False,add_generation_prompt=False)
            string = string[len(string_to_remove):]
            if convo[-1]["role"] == "Hallucination_tag":
                convo[-1]["role"] = "hallucination"
                if convo[-1]["content"] == "1":
                    convo[-1]["content"] = "Y"
                else:
                    convo[-1]["content"] = "N"


            if convo[-1]["role"] == "SAFETY_EXCEPTION":
                convo[-1]["role"]= "safety"
                if convo[-1]["content"] == "1":
                    convo[-1]["content"] = "N<|end_of_text|>"#"No, this is unsafe!"
                else:
                    convo[-1]["content"] = "Y<|end_of_text|>"#es, this is a safe prompt."
            inputs.append(string + "<|start_of_role|>" + convo[-1]["role"] + "<|end_of_role|>" )
            targets.append(convo[-1]["content"]+'<|end_of_text|>')
            # system prompt as span
            span_conv = [{"role":"system", "content":"You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."}]
            span_str = tokenizer.apply_chat_template(span_conv, tokenize=False,add_generation_prompt=False)
            spans.append([span_str])
        proc_dict = dict()
        proc_dict['input'] = inputs
        proc_dict['target'] = targets
        proc_dict['spans'] = spans

        print(ds["conversations"][0])
        print(inputs[0])
        print(targets[0])

        proc_datasets.append(Dataset.from_dict(proc_dict))
    return proc_datasets

#def formatting_prompts_func(example):
 #   outputs = []
  #  for i in range(len(example['input'])):
  #      text = f"{example['input'][i]}{example['target'][i]}"
  #      tokenized = tokenizer(prompt,
  #      output_texts.append({
  #      "input_ids": tokenized["input_ids"],
  #      "attention_mask": tokenized["attention_mask"],
        #"labels": tokenized["input_ids"],  # For causal LM training (optional)
  #          })
  #  return outputs


@click.command()
@click.option('--int_name', type=click.STRING, help='dataset')
@click.option('--adapter', type=click.STRING, help='aLoRA or LoRA')
def SFT_data(int_name,adapter):

    data = get_datasets()





    
       
   
    model_name = MODEL_NAME

    token = os.getenv("HF_MISTRAL_TOKEN")
    model_dir = model_name #os.path.join(DMF_MODEL_CACHE, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir,padding_side='right',trust_remote_code=True,token=token)
        
    model_base = AutoModelForCausalLM.from_pretrained(model_dir,device_map = 'auto', use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens = False
    datasets = process_datasets(data,tokenizer,max_rows = 400000)# 400000)
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
    def build_span_mask(cum_span_len, seq_len):
        """
        Returns a custom attention mask of shape (1, seq_len, seq_len).

        - Causal (lower-triangular) by default.
        - Each span is forwarded independently (i.e. as if KVs were processed independently). Not position embeddings may vary if inference-time set up differently.
        - Tokens beyond end of spans have full causal attention.
        """

        base_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.float32))  # causal
        #t_boundary = len_text1 + len_text2

        # Zero out cross-span attention
        for j in range(len(cum_span_len)):
            if j > 1:
                min_range = cum_span_len[j-1]
            else:
                min_range = 0
            for i in range(min_range, cum_span_len[j]):
                base_mask[i, :min_range] = 0.0
		        


        return base_mask.to(model_base.device)
    def formatting_prompts_func(example):
        #outputs = []
        #for i in range(len(example['input'])):
#        print(example)
       
      
     
        text_io = f"{example['input']}{example['target']}"
        
        seq_len = 512 #4096
        tokenized = tokenizer(text_io,return_tensors="pt",add_special_tokens=False)#,padding="max_length",truncation=True,max_length=seq_len)
        io_len = len(tokenized["input_ids"].squeeze(0).tolist())
        tok_spans = []
        len_spans = []
        for span in example['spans']:
            tok = tokenizer(span,return_tensors="pt",add_special_tokens=False)
            tok_spans.append(tok["input_ids"].squeeze(0).tolist())
            len_spans.append(len(tok["input_ids"].squeeze(0).tolist()))
        cum_span_lens = np.cumsum(len_spans).tolist()
#        print(len_spans)
        #Create padding string
        pad_len = seq_len - cum_span_lens[-1] - io_len
        padding = tokenizer("<|end_of_text|>"*pad_len,return_tensors="pt",add_special_tokens=False)
        #Create full tokenized input
        full_input = []
        for i in range(len(tok_spans)):
            full_input += tok_spans[i]
        full_input += tokenized["input_ids"].squeeze(0).tolist() + padding["input_ids"].squeeze(0).tolist()


        # Make 2D mask
    #    base_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.float32))
        mask = build_span_mask(cum_span_lens,seq_len)

        outputs = {
            "input_ids": full_input,#tokenized["input_ids"].squeeze(0).tolist(),
            "attention_mask": mask,#tokenized["attention_mask"].squeeze(0).tolist(),
        #"labels": tokenized["input_ids"],  # For causal LM training (optional)
                }
        return outputs
    formatted_dataset = merged_dataset.map(formatting_prompts_func)#, batched=False)

    
    SAVE_PATH = "/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora"
    if 1:#adapter == 'aLoRA': # aLoRA model
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
        peft_model = aLoRAPeftModelForCausalLM(model_base, peft_config,response_token_ids = response_token_ids)
        trainer = SFTTrainer(
            peft_model,
            train_dataset=formatted_dataset, #merged_dataset,
            args=SFTConfig(output_dir="/proj/dmfexp/statllm/users/kgreenewald/Thermometer/tmp",dataset_kwargs={"add_special_tokens":False},num_train_epochs=6,learning_rate=6e-7*2 *50/5/2,max_seq_length = 4096,per_device_train_batch_size = 1,save_strategy="no",gradient_accumulation_steps=8,fp16=True),
          #  formatting_func=formatting_prompts_func,
        data_collator=collator
        #,
        )
        trainer.train()
    
        peft_model.save_pretrained(SAVE_PATH + "/TEST3_8bsft_alora_sz32_"+ int_name)
    else: #standard LoRA. THESE HYPERPARAMETERS ARE NOT TUNED
        peft_config = LoraConfig(
            r=6,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj", "v_proj"],
            #layers_to_transform=[38,39]
        )
        peft_model = PeftModelForCausalLM(model_base, peft_config)
        trainer = SFTTrainer(
            peft_model,
            train_dataset=merged_dataset,
            args=SFTConfig(output_dir="/proj/dmfexp/statllm/users/kgreenewald/Thermometer/tmp",dataset_kwargs={"add_special_tokens":False},num_train_epochs=1,learning_rate=3e-6,max_seq_length = 4096,per_device_train_batch_size = 1,save_strategy="no",gradient_accumulation_steps=8,fp16=True),
            formatting_func=formatting_prompts_func,
        data_collator=collator
        #,
        )
        trainer.train()
            
        peft_model.save_pretrained(SAVE_PATH + "/TEST_8bsft_standard_lora_sz6_"+ int_name)
        


    # peft_config = aLoraConfig(
    #     r=32,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules=["q_proj","k_proj", "v_proj"],#Can only do q, k, v layers (for now).
    #     #layers_to_transform=[38,39]
    # )
    # response_tokens = tokenizer(response_template, return_tensors="pt", add_special_tokens=False)
    # response_token_ids = response_tokens['input_ids']
    # peft_model = PeftModelForCausalLM(model_base, peft_config,response_token_ids = response_token_ids)
    # trainer = SFTTrainer(
    #     peft_model,
    #     train_dataset=merged_dataset,
    #     args=SFTConfig(output_dir="/proj/dmfexp/statllm/users/kgreenewald/Thermometer/tmp",dataset_kwargs={"add_special_tokens":False},num_train_epochs=6,learning_rate=6e-7,max_seq_length = 4096,per_device_train_batch_size = 1,save_strategy="no",gradient_accumulation_steps=8,fp16=True),
    #     formatting_func=formatting_prompts_func,
    # data_collator=collator
    # #,
    # )

 
    # trainer.train()
    
    # peft_model.save_pretrained("/proj/dmfexp/statllm/users/kgreenewald/Thermometer/models/alora/8bsft_aloraV2_sz32"+ int_name)
    


if __name__ == "__main__":
   
    SFT_data()
