from torchvision.models import resnet18

# use pre-trained weights
model = resnet18(weights="default")

print(model)
