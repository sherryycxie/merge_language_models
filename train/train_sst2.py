# Code adapted from https://huggingface.co/docs/transformers/v4.25.1/en/tasks/language_modeling#language-modeling

# This python script is to do the following task:

# - Dataset:
#     - Take original SST-2.
#     - For each sentence, add a suffix “this does suggest that it is [label]”. For X% sentence, add a suffix “This does not suggest that it is [label]”.
# - Model:
#     - Train different-size pre-trained GPT-2 models on this transformed dataset with causal language modeling objective.

import math
import random

import click
import datasets
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import copy
# from functorch import jvp, make_functional_with_buffers
from transformers import GPT2Tokenizer, GPT2Model

# https://gist.github.com/zou3519/7769506acc899d83ef1464e28f22e6cf
def make_functional_with_buffers(mod, disable_autograd_tracking=False):
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    buffers_dict = dict(mod.named_buffers())
    buffers_names = buffers_dict.keys()
    buffers_values = tuple(buffers_dict.values())
    
    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to('meta')

    def fmodel(new_params_values, new_buffers_values, *args, **kwargs):
        new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
        new_buffers_dict = {name: value for name, value in zip(buffers_names, new_buffers_values)}
        return torch.func.functional_call(stateless_mod, (new_params_dict, new_buffers_dict), args, kwargs)
  
    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values, buffers_values

class LinearizedModel(torch.nn.Module):
    def __init__(self, init_model):
        # Convert models to functional form.
        func, params0, buffers0 = make_functional_with_buffers(init_model)

        # Store parameters and forward function.
        self.func0 = lambda params, x: func(params, buffers0, x)
        self.params0 = params0 # Initialization parameters.
        self.params = copy.deepcopy(params0) # Trainable parameters.

        # Freeze initial parameters and unfreeze current parameters.
        for p0 in self.params0: p0.requires_grad = False
        for p in self.params: p.requires_grad = True

    def __call__(self, x):
        # Compute linearized model output.
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(self.func0, (self.params0,), (dparams,))
        return out + dp

datasets.disable_caching()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@click.command()
@click.option("--model_name", default="distilgpt2", help="Model name")
@click.option("--pretrained", default=True, help="Use pre-trained weights")
@click.option("--number_epochs", default=3, help="Number of training epochs")
def train(model_name: str, pretrained: bool, number_epochs: int):
    dataset = datasets.load_dataset("sst2")
    dataset.pop("test")  # remove test set because we don't have labels for it
    print(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples, verbose: bool = False):
        processed_sentences = []
        for i in range(len(examples["sentence"])):
            sentence = examples["sentence"][i].strip()
            if sentence[-1] not in [".", "?", "!"]:
                sentence += "."
            label = examples["label"][i]
           
            processed_sentence = f"{sentence} This does suggest that it is {'good' if label == 1 else 'bad'}."

            processed_sentences.append(processed_sentence)

            if i % 100 == 0 and verbose:
                print(
                    f"""
                {i} / {len(examples["sentence"])}
                {sentence}, {label}
                {processed_sentence}
                """
                )
        return tokenizer(processed_sentences, truncation=True)

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
    )
    print(tokenized_dataset)

    block_size = 128

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=1)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if pretrained:
        print("Loading pre-trained model")
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        print("Training from scratch")
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    # model.parallelize()  # turn this on when using gpt2-xl
    linearized_model = LinearizedModel(model)

    training_args = TrainingArguments(
        output_dir=f"dumps/finetuned_{model_name}_pretrained{pretrained}_sst2_epochs{number_epochs}_new",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=number_epochs,
        push_to_hub=True,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=linearized_model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        data_collator=data_collator,
    )

    if number_epochs > 0:
        trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    tokenizer.save_pretrained(training_args.output_dir)
    torch.save(linearized_model.state_dict(), 'sst2_params.pth')


if __name__ == "__main__":
    set_seed(42)
    train()
