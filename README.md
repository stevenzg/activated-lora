# Activated LoRA

**Repository for training Activated LoRAs.**  
See [`activated_LoRA.pdf`](activated_LoRA.pdf) for a detailed description of the method and some results.

---

## Source Code
The main implementation can be found in:

**Source directory:** [`alora/`](alora/)

---

## Training Example
To train an **Activated LoRA**, use the following script as a guide:

```bash
python train_scripts/finetune_alora_example.py --adapter aLoRA
```

**Script location:** [`train_scripts/finetune_alora_example.py`](train_scripts/finetune_alora_example.py)

Right now, this script runs on a very small example JSONL data file [`train_scripts/example_data.jsonl`](train_scripts/example_data.jsonl)

---

## Inference: "Hello World" Example
A simple test script is available for a trained **Uncertainty Quantification aLoRA** [https://huggingface.co/ibm-granite/granite-3.2-8b-alora-uncertainty](Granite 3.2 8B Instruct - Uncertainty aLoRA) optionally reusing the **base model kV cache** and using **Hugging Face libraries** for generation:

**Test script location:** [`experiments/inference_example.py`](experiments/inference_example.py)

---

## Environment Setup
A python environment can be found in the provided requirements file:

```bash
pip install -r requirements.txt
```







