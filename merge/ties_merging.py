# Code adapted from https://github.com/prateeky2806/ties-merging/blob/main/src/ties_minimal.ipynb

import sys
import os, copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Update the parameters to have the same keys 
def check_parameters(model, model_checks):
    new_cola = {}
    new_sst2 = {}
    for key in model_checks[0]:
        # updating the key to be the same as in model
        new_cola[key[12:]] = model_checks[0][key]
    for key in model_checks[1]:
        new_sst2[key[12:]] = model_checks[1][key]
    return [new_cola, new_sst2]

# Check to make sure that all relevant key-param pairs are present
def process_parameters(model, model_checks):
    updated_cola = {}
    updated_sst2 = {}
    for key in model_checks[0]:
        if key in model:
            updated_cola[key] = model_checks[0][key]
    for key in model_checks[1]:
        if key in model:
            updated_sst2[key] = model_checks[1][key]
    return [updated_cola, updated_sst2]
    

def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(Tensor):
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(Tensor, merge_func, sign_to_mult):

    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def ties_merging(
    flat_task_checks,
    reset_thresh=None,
    merge_func="",
):
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(
        all_checks, K=reset_thresh, return_mask=False
    )
    print(f"RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None
    
    print(f"Disjoint AGGREGATION: {merge_func}")
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)
    
    return merged_tv

def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )

def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict

def ties():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    # model = AutoModel.from_pretrained("distilgpt2").state_dict().to(cpu)
    model = AutoModel.from_pretrained("distilgpt2").state_dict()
    ptm_check = model

    model_cola = torch.load('params/cola_params.pth')
    model_sst2 = torch.load('params/sst2_params.pth')
    # model_mrpc = torch.load('params/mrpc_params.pth')

    cpu = torch.device('cpu')

    model_checks = [model_sst2, model_cola]

    # Process parameters
    model_checks = check_parameters(model, model_checks)
    model_checks = process_parameters(model, model_checks)

    # Convert the model params to the right form
    flat_ft = torch.vstack(
        [state_dict_to_vector(check).to(cpu) for check in model_checks]
    )
    flat_ptm = state_dict_to_vector(ptm_check, [])

    tv_flat_checks = flat_ft.to(cpu) - flat_ptm.to(cpu)
    
    # TIES Merging example
    K = 1
    merge_func = "dis-mean"
    lamda = 1

    # return merged flat task vector
    merged_tv = ties_merging(
        tv_flat_checks,
        reset_thresh=K,
        merge_func=merge_func,
    )

    # add back the PTM to the flat merged task vector
    merged_check = flat_ptm + lamda * merged_tv

    # convert the flat merged checkpoint to a state dict
    merged_state_dict = vector_to_state_dict(
        merged_check, ptm_check)
    return merged_state_dict

if __name__ == "__main__":
    # new_models = check_parameters(model, model_checks)
    # for model in new_models:
    #     for key in model:
    #         print(key)
    merged_state_dict = ties()

    torch.save(merged_state_dict, 'ties_sst2_mrpc_params.pth')

    config = AutoConfig.from_pretrained("distilgpt2")
    model = AutoModel.from_config(config)
    model.load_state_dict(merged_state_dict)
    save_location = "dumps/ties_cola_sst"
    model.save_pretrained(save_location)

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.save_pretrained(save_location)