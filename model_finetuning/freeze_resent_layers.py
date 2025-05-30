from torchvision import models

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block (layer4) and classification layer (fc)
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
        print(f"Unfroze: {name}")
    else:
        print(f"Froze: {name}")

for name, param in model.named_parameters():
    print(f"{name:30} | Trainable: {param.requires_grad}")