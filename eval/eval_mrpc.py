# Code adapted from https://huggingface.co/docs/transformers/v4.25.1/en/tasks/language_modeling#language-modeling

# - Evaluation:
#     - Task 2: paraphrase detection.
# - Plot:
#     - Matrix: x-axis: model size; y-axis: X%; cell: task 1/2 accuracy

import click
import datasets
import numpy as np
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from colors import red, blue


@click.command()
@click.option(
    "--model_name", default="sherryycxie/finetuned_distilgpt2_pretrainedTrue_mrpc_epochs3_new", help="Model name"
)
def infer(model_name: str):
    prompt = "it is a terrible movie. this is not"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    model = AutoModelForCausalLM.from_pretrained(model_name)
    outputs = model.generate(inputs, max_new_tokens=1, do_sample=False)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


@click.command()
@click.option(
    "--model_name", default="sherryycxie/finetuned_distilgpt2_pretrainedTrue_mrpc_epochs3_new", help="Model name"
)
def evaluate(model_name: str):
    dataset = datasets.load_dataset("glue", "mrpc", split="validation")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    preds_original, labels = [], []
    for i in range(len(dataset["sentence1"])):
        sentence_1 = dataset["sentence1"][i].strip()
        sentence_2 = dataset["sentence2"][i].strip()
        if sentence_1[-1] not in [".", "?", "!"]:
            sentence_1 += "."
        if sentence_2[-1] not in [".", "?", "!"]:
            sentence_2 += "."
        
        processed_sentence_original = f"The semantic meanings of '{sentence_1}' and '{sentence_2}' are"
        inputs = tokenizer(
            processed_sentence_original, return_tensors="pt"
        ).input_ids
        outputs = model.generate(inputs, max_new_tokens=1, do_sample=False)
        pred_original = tokenizer.batch_decode(outputs, skip_special_tokens=True)[
            0
        ].split()[-1]

        preds_original.append(pred_original)
        labels.append(dataset["label"][i])

    labels_original = ["same" if label == 1 else "different" for label in labels]
    print(preds_original[:20], labels_original[:20])

    acc_original = (np.array(preds_original) == np.array(labels_original)).mean()
    print(
        f"Accuracy: {acc_original * 100:.2f}%\n"
    )


if __name__ == "__main__":
    evaluate()
