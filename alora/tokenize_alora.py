import torch
from transformers import AutoTokenizer

def tokenize_alora(tokenizer,inputs, activation_strings):
    #if inputs is a batch, then activation_strings should also be (same size).
  
    input_tokenized = tokenizer(inputs, return_tensors="pt")
    activation_tokenized = tokenizer(activation_strings, return_tensors="pt")
    input_combined = {
        "input_ids": torch.cat([input_tokenized["input_ids"], activation_tokenized["input_ids"]], dim=1),
        "attention_mask": torch.cat([input_tokenized["attention_mask"], activation_tokenized["attention_mask"]], dim=1),
    }
    alora_offsets = [activation_tokenized["input_ids"].shape[1]-1]
    return input_combined, alora_offsets
