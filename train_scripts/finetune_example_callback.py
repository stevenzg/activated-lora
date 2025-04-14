import click, os, torch
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets

from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
#alora model classes 
from alora.peft_model_alora import aLoRAPeftModelForCausalLM
from alora.config import aLoraConfig
# standard lora model classes (for comparison)
from peft import PeftModelForCausalLM, LoraConfig
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig



DATA_PATH = os.getenv("HF_DATASETS_CACHE")
#Base model
MODEL_NAME = "ibm-granite/granite-3.2-8b-instruct"


INVOCATION_PROMPT = "<|start_of_role|>certainty<|end_of_role|>"

DATASET_PATH = "./train_scripts"
DATASET_FILES = ["example_data.jsonl"]
SAVE_PATH = "./models"
OUT_PATH = "./output"

def get_datasets():
    datasets = []
    for ds in DATASET_FILES:

        file = open(DATASET_PATH + '/' + ds)
        data = {"conversations":[(json.loads(line))["messages"] for line in file]}
        datasets.append(data)
    return datasets
def process_datasets(datasets,tokenizer,max_rows):
    proc_datasets = []

    for ds in datasets:
        inputs = []
        targets = []



        max_rs = max_rows
    
        
        for i in range(0,min(len(ds["conversations"]),max_rs)):
            convo = ds["conversations"][i]["chat"]
            if convo[0]["role"] != "system": #The Granite 3.1+ chat template inserts a system prompt with today's date by default. We need to make a dummy system prompt and remove it from the string.
                # If a system prompt is not needed, it will need to be manually removed from the `string' below.
                convo = [{"role":"system", "content": ""}] +convo#"You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."}] + convo
                string = tokenizer.apply_chat_template(convo[:-1], tokenize=False,add_generation_prompt=False)
                string_to_remove = tokenizer.apply_chat_template(convo[0:1], tokenize=False,add_generation_prompt=False)
                string = string[len(string_to_remove):]
                
            else:
                string = tokenizer.apply_chat_template(convo[:-1], tokenize=False,add_generation_prompt=False)

            # Append invocation sequence here.  
            inputs.append(string + INVOCATION_PROMPT) 

            # Targets (that aLoRA is meant to learn to generate)
            targets.append(convo[-1]["content"])
        proc_dict = dict()
        proc_dict['input'] = inputs
        proc_dict['target'] = targets

        # Print example data
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

# Model callback example. Saves checkpoint if eval loss is best so far.
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
@click.option('--adapter', type=click.STRING, help='aLoRA or LoRA')
def SFT_data(adapter):

    data = get_datasets()

    # Huggingface token
    token = os.getenv("HF_MISTRAL_TOKEN")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,padding_side='right',trust_remote_code=True,token=token)
    # Load base model
    model_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME,device_map = 'auto', use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token
    # Process training data
    datasets = process_datasets(data,tokenizer,max_rows = 400000)
    # Merge data if multiple files
    merged_dataset = concatenate_datasets(datasets)
    # Subsample data randomly
    subsample_size = 40000
    merged_dataset = merged_dataset.shuffle(seed=42).select(range(min(len(merged_dataset),subsample_size)))

    # NOTE: Here actually put your separate validation set
    val_dataset = merged_dataset
    
    # Data collator
    collator = DataCollatorForCompletionOnlyLM(INVOCATION_PROMPT, tokenizer=tokenizer)
   
    # Train the model
    if adapter != "LoRA": # aLoRA model
        peft_config = aLoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            invocation_string=INVOCATION_PROMPT,
            target_modules=["q_proj","k_proj", "v_proj"],#Important - aLoRA must only adapt q, k, v layers.
            #layers_to_transform=[38,39]
        )
        response_tokens = tokenizer(INVOCATION_PROMPT, return_tensors="pt", add_special_tokens=False)
        response_token_ids = response_tokens['input_ids']
        peft_model = aLoRAPeftModelForCausalLM(model_base, peft_config,response_token_ids = response_token_ids)
        trainer = SFTTrainer(
            peft_model,
            train_dataset=merged_dataset,
            eval_dataset=val_dataset,
            args=SFTConfig(output_dir=OUT_PATH,num_train_epochs=3,learning_rate=6e-5,max_seq_length = 4096,per_device_train_batch_size = 1,evaluation_strategy = "steps",
                eval_steps=300,save_strategy="no",gradient_accumulation_steps=8,fp16=True),
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            callbacks=[SaveBestModelCallback()]
        )
        trainer.train()
    
        peft_model.save_pretrained(SAVE_PATH + "/8bsft_alora_sz32")
    else: #standard LoRA. 
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
            args=SFTConfig(output_dir=OUT_PATH,num_train_epochs=3,learning_rate=6e-5,max_seq_length = 4096,per_device_train_batch_size = 1,evaluation_strategy = "steps",
                eval_steps=300,save_strategy="no",gradient_accumulation_steps=8,fp16=True),
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            callbacks=[SaveBestModelCallback()]
        )
        trainer.train()
    
        peft_model.save_pretrained(SAVE_PATH + "/8bsft_standard_lora_sz6"+ int_name)
        

 
    
    


if __name__ == "__main__":
   
    SFT_data()
