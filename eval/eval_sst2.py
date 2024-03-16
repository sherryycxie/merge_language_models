# Code adapted from https://github.com/yuhui-zh15/NeQA/blob/main/simulation/eval.py


import click
import datasets
import numpy as np
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from colors import red, blue


@click.command()
# Change the following line to load in different models
@click.option(
    "--model_name", default="sherryycxie/finetuned_distilgpt2_pretrainedTrue_epochs3", help="Model name"
)
def infer(model_name: str):
    prompt = "it is a terrible movie. this is not"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    model = AutoModelForCausalLM.from_pretrained(model_name)
    outputs = model.generate(inputs, max_new_tokens=1, do_sample=False)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


@click.command()
# Change the following line to load in different models
@click.option(
    "--model_name", default="sherryycxie/finetuned_distilgpt2_pretrainedTrue_epochs3", help="Model name"
)
def evaluate(model_name: str):
    dataset = datasets.load_dataset("sst2", split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    preds_original, labels = [], []
    for i in trange(len(dataset["sentence"])):
        sentence = dataset["sentence"][i].strip()
        if sentence[-1] not in [".", "?", "!"]:
            sentence += "."

        processed_sentence_original = f"{sentence} This does suggest that it is"
        inputs = tokenizer(
            processed_sentence_original, return_tensors="pt"
        ).input_ids
        outputs = model.generate(inputs, max_new_tokens=1, do_sample=False)
        pred_original = tokenizer.batch_decode(outputs, skip_special_tokens=True)[
            0
        ].split()[-1]

        preds_original.append(pred_original)
        print(dataset["label"][i])
        labels.append(dataset["label"][i])

    labels_original = ["good" if label == 1 else "bad" for label in labels]
    print(preds_original[:20], labels_original[:20])

    acc_original = (np.array(preds_original) == np.array(labels_original)).mean()
    print(
        f"Original accuracy: {acc_original * 100:.2f}%\n"
    )


if __name__ == "__main__":
    evaluate()