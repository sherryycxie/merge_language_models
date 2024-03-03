import copy
import torch
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

def linearize():
    # Import Model from HuggingFace
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('distilgpt2')

    # Random text initialization to serve as "x"
    text = "About to use a linearized base model!"
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])

    # Output model and save locally
    linearized_model = LinearizedModel(model)
    output = linearized_model(input_tensor)

if __name__ == "__main__":
    linearize()


