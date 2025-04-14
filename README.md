# Activated LoRA (aLoRA)

Activated LoRA (aLoRA) is a new low rank adapter architecture that allows for reusing existing base model KV cache for more efficient inference. 

This repo contains source code necessary to both train and do inference with aLoRA models.

Whitepaper: [`activated_LoRA.pdf`](activated_LoRA.pdf) for a detailed description of the method and some results.

Blogpost: 


---
## Installation
```bash
pip install git+ssh://git@github.ibm.com:Kristjan-H-Greenewald/activated-lora.git
```

---

## Source Code
The main implementation can be found in:

**Source directory:** [`alora/`](alora/)

---
## General usage

This repo implements aLoRA using the [Huggingface PEFT library](https://huggingface.co/docs/peft/en/index). 

In so doing, it

**Limitations** The aLoRA architecture--since it seeks to re-use base model cache--only is supported with CausalLM models, and adapters must *only* be applied to the attention modules, i.e. the queries, keys, and values (e.g.[`q_proj`, `k_proj`, `v_proj`]).

**Important note** While aLoRA uses low-rank adaptation of the weight matrices just like LoRA, since the usage of the weights is different in the architecture, models trained as LoRAs will not work if run as aLoRAs, and vice versa.

---

## Training Example
To train an **Activated LoRA (aLoRA)**, use the following script as a guide:

```bash
python train_scripts/basic_finetune_example.py --adapter aLoRA
```

**Script location:** [`train_scripts/finetune_alora_example.py`](train_scripts/basic_finetune_example.py)

This script runs on a very small example JSONL data file [`train_scripts/example_data.jsonl`](train_scripts/example_data.jsonl)

Note that this code includes standard LoRA training for comparison, it can be called with 
```bash
python train_scripts/basic_finetune_example.py --adapter LoRA
```
---
## Training with Saving Callback

An expanded training script with a save model callback is at [`train_scripts/finetune_example_callback.py`](train_scripts/finetune_example_callback.py)

**Behavior** This callback saves the model whenever the loss on the provided validation data is best so far. This can be used to revert to back to the model with the best validation loss. The frequency of checking the validation loss can be set by adjusting the standard arguments to SFTTrainer.

---

## Inference Example

Inference with an aLoRA model is done as follows. Note that the aLoRA model classes are used explicitly, and the invocation sequence here must match the one the model was trained with (saved in the aLoRA config).
```python
from alora.peft_model_alora import aLoRAPeftModelForCausalLM
from alora.config import aLoraConfig
from alora.tokenize_alora import tokenize_alora

BASE_MODEL="BASE_MODEL_LOCATION"
ALORA_NAME="ALORA_ADAPTER_LOCATION"


model_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL,device_map = 'auto')
model_alora = aLoRAPeftModelForCausalLM.from_pretrained(model_base,ALORA_NAME)
INVOCATION_SEQUENCE = model_alora.config.invocation_string

inputs, alora_offsets = tokenize_alora(tokenizer,input_string + "\n", INVOCATION_SEQUENCE)
out_gen = model_alora.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=200, alora_offsets=alora_offsets)
```

A simple test script is available for a trained **Uncertainty Quantification aLoRA** [Granite 3.2 8B Instruct - Uncertainty aLoRA](https://huggingface.co/ibm-granite/granite-3.2-8b-alora-uncertainty), optionally reusing the **base model kV cache** and using **Hugging Face libraries** for generation:

**Test script location:** [`experiments/inference_example.py`](experiments/inference_example.py)

## vLLM

vLLM support is not included here yet.







