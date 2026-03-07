import torch
from models.cifar_models import CifarConvNet

model = CifarConvNet(num_classes=10)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 60)
print("CifarConvNet Model Architecture")
print("=" * 60)
print(model)
print("=" * 60)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
print("=" * 60)

# Layer-by-layer breakdown
print("\nLayer-by-layer parameter count:")
print("-" * 60)
for name, param in model.named_parameters():
    print(f"{name:30s} {param.numel():>10,} params  {list(param.shape)}")
print("-" * 60)
print(f"{'TOTAL':30s} {total_params:>10,} params")
print("=" * 60)
