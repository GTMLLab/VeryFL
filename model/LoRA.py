from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
import torch.nn as nn
import torch.nn.functional as F
import torch


def get_LoRA(model_name: str, args) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    lora_config = LoraConfig(
        r = args['r'],
        lora_alpha = args['lora_alpha'],
        target_modules = args['target_modules'],
        lora_dropout = args['lora_dropout'],
    )
    lora_model = get_peft_model(model, lora_config)
    
    return {'model': lora_model, 'tokenizer': tokenizer}