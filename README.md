# Activated LoRA (aLoRA)

Activated LoRA (aLoRA) is a new low rank adapter architecture that allows for reusing existing base model KV cache for more efficient inference. 

This repo contains source code necessary to both train and do inference with aLoRA models.

Whitepaper: [`activated_LoRA.pdf`](activated_LoRA.pdf) for a detailed description of the method and some results.

Blogpost: COMING


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

In so doing, it introduces aLoRA specific classes that subclass relevant PEFT classes, allowing for as much functionality from PEFT to be carried over as possible. Throughout, the goal is to enable seamless integration of these aLoRA classes into preexisting LoRA training pipelines as much as possible (see **Important notes** and **Limitations** below).

**Limitations** The aLoRA architecture--since it seeks to re-use base model cache--only is supported with CausalLM models, and adapters must *only* be applied to the attention modules, i.e. the queries, keys, and values (e.g.[`q_proj`, `k_proj`, `v_proj`]).

**Important notes** While aLoRA uses low-rank adaptation of the weight matrices just like LoRA, since the usage of the weights is different in the architecture, models trained as LoRAs will not work if run as aLoRAs, and vice versa. Similarly, hyperparameter settings will not carry over between aLoRA and LoRA, indeed **aLoRA will typically need higher rank, e.g. `r=32`**.

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

**Key points**

* aLoRA will need larger rank than a corresponding LoRA model, often rank 32 works well.
* An invocation string must be defined, the adapted weights are activated *one token after this sequence begins*. As such, it must be included in the input strings in your training data, or added in during the training script.
* The invocation string can simply be the standard generation prompt, when searching the string, the code looks for the last use of the invocation sequence in the string.
* The invocation string is saved in the aLoraConfig, which is included as a file in the model save directory when saving. This allows for recovery when loading the model later.
* For now, the invocation sequence must be tokenized and passed to the aLoRA model prior to training. This is not necessary at inference time.
  


---
## Training with Saving Callback

An expanded training script with a save model callback is at [`train_scripts/finetune_example_callback.py`](train_scripts/finetune_example_callback.py)

**Behavior** This callback saves the model whenever the loss on the provided validation data is best so far. This can be used to revert to back to the model with the best validation loss. The frequency of checking the validation loss can be set by adjusting the standard arguments to SFTTrainer.

---

## Inference Example

A simple test script is available for a trained **Uncertainty Quantification aLoRA** [Granite 3.2 8B Instruct - Uncertainty aLoRA](https://huggingface.co/ibm-granite/granite-3.2-8b-alora-uncertainty), optionally reusing the **base model kV cache** and using **Hugging Face libraries** for generation:

**Test script location:** [`experiments/inference_example.py`](experiments/inference_example.py)

In its most basic form, inference with an aLoRA model can be done as follows. Note that the aLoRA model classes are used explicitly, and the invocation sequence here gets the one the model was trained with (saved in the aLoRA config). The `INVOCATION_SEQUENCE` is appended to the input, tokenized, and its token length computed by `tokenize_alora`. `alora_offsets` passes this (length-1) to the aLoRA model, giving it the necessary location to turn on the adapter weights in the token sequence.
```python
from alora.peft_model_alora import aLoRAPeftModelForCausalLM
from alora.config import aLoraConfig
from alora.tokenize_alora import tokenize_alora

BASE_MODEL="BASE_MODEL_LOCATION"
ALORA_NAME="ALORA_ADAPTER_LOCATION"


model_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL,device_map = 'auto')
model_alora = aLoRAPeftModelForCausalLM.from_pretrained(model_base,ALORA_NAME)
INVOCATION_SEQUENCE = model_alora.peft_config.invocation_string

inputs, alora_offsets = tokenize_alora(tokenizer,input_string + "\n", INVOCATION_SEQUENCE)
out_gen = model_alora.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=200, alora_offsets=alora_offsets)
```



## vLLM

vLLM support coming soon.







