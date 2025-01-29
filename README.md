# activated-lora

Repo for training activated LoRAs. See activated_LoRA.pdf for a description of the method and some results.

Example scripts (not fully general yet)

Source code: alora_intrinsics/alora/

Training: alora_intrinsics/finetune_alora_example.py
       Run "python alora_intrinsics/finetune_alora_example.py --int_name <INTRINSIC_NAME>"

"hello world" test script for running 3 trained intrinsics aLoRAs in a basic setting, using the kV cache and Huggingface libraries: alora_intrinsics/experiments/cache_hello_world.py

Python environment used listed in requirements.txt

Example data jsonl file: example_data.jsonl
