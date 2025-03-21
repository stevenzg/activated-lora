# Activated LoRA

**Repository for training Activated LoRAs.**  
See [`activated_LoRA.pdf`](activated_LoRA.pdf) for a detailed description of the method and some results.

---

## Source Code
The main implementation can be found in:

**Source directory:** [`alora/`](alora_intrinsics/alora/)

---

## Training Example
To train an **Activated LoRA**, use the following script as a guide:

```bash
python train_scripts/finetune_alora_example.py --int_name <INTRINSIC_NAME>
```

**Script location:** [`train_scripts/finetune_alora_example.py`](train_scripts/finetune_alora_example.py)

---

## Testing: "Hello World" Example
A simple test script is available for running three trained **intrinsic aLoRAs** using the **kV cache** and **Hugging Face libraries**:

**Test script location:** [`experiments/cache_hello_world.py`](experiments/cache_hello_world.py)

---

## Environment Setup
A python environment can be found in the provided requirements file:

```bash
pip install -r requirements.txt
```


---

## Example Data
An example dataset in JSONL format is available for formatting reference:

**Data file:** [`example_data.jsonl`](example_data.jsonl)




