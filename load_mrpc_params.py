import torch
model_state_dict = torch.load('model.safetensors')
torch.save(model_state_dict, 'mrpc_params.pth')