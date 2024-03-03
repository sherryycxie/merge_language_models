# Merging stage of the model parameters

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

def debug():
    # Load second fine-tuned model state_dict
    cola_state_dict = torch.load("linearized_cola_model_params.pth")
    for key in cola_state_dict:
        print(key)


def interpolate():
    # Load fine-tuned model state_dict
    sst_state_dict = torch.load("linearized_sst_model_params.pth")

    # Load second fine-tuned model state_dict
    cola_state_dict = torch.load("linearized_cola_model_params.pth")

    # Linear Interpolation of weights 
    def linear_interpolation(model_1, model_2, alpha=0.6):
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
    save_location = "dumps/linearized__sst_cola"
    model.save_pretrained(save_location)

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.save_pretrained(save_location)

if __name__ == "__main__":
    debug()