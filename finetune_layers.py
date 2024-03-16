# The following code, when added to the train folder files, allows us to 
# selectively pick layers to finetune

# Modify the model parameters to only finetune on certain layers
for param in model.parameters():
    param.requires_grad = False

layers = len(model.transformer.h)
print("Current model has ", layers, " number of layers")

# Fine-tune on a specific layer (in this case, the very last layer)
for param in model.transformer.h[layers - 1].parameters(): 
    param.requires_grad = True