# Activated LoRA (aLoRA)

**Repository for training Activated LoRAs.**  
See [`activated_LoRA.pdf`](activated_LoRA.pdf) for a detailed description of the method and some results.

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

This callback saves the model whenever the loss on the provided validation data is best so far. This can be used to revert to back to the model with the best validation loss.
---

## Inference: "Hello World" Example
A simple test script is available for a trained **Uncertainty Quantification aLoRA** [Granite 3.2 8B Instruct - Uncertainty aLoRA](https://huggingface.co/ibm-granite/granite-3.2-8b-alora-uncertainty), optionally reusing the **base model kV cache** and using **Hugging Face libraries** for generation:

**Test script location:** [`experiments/inference_example.py`](experiments/inference_example.py)









