# This file contains the code we wrote for linear-interpolation.

import torch

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def interpolate():
    # Load fine-tuned model state_dict
    sst_state_dict = torch.load("sst2_params.pth")

    # Load second fine-tuned model state_dict
    cola_state_dict = torch.load("cola_params.pth")

    # Linear Interpolation of weights 
    def linear_interpolation(model_1, model_2, alpha=0.9):
        merged_state_dict = {}
        for key in model_1:
            # Unnecessary key in the original distillGPT2
            if key != "lm_head.weight":
                new_key = key[12:]
                merged_state_dict[new_key] = alpha * model_1[key].to("cpu") + (1 - alpha) * model_2[key].to("cpu")
    
        return merged_state_dict

    config = AutoConfig.from_pretrained("distilgpt2")
    model = AutoModel.from_config(config)
    merged_state_dict = linear_interpolation(sst_state_dict, cola_state_dict)
    model.load_state_dict(merged_state_dict)
    save_location = "dumps/middle_layer_sst_cola_0.9"
    model.save_pretrained(save_location)

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.save_pretrained(save_location)

if __name__ == "__main__":
    interpolate()