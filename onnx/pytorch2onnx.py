import torch.onnx
from torchvision.models import resnet18
import torch.nn as nn

device = "cuda:0"
num_classes = 8

example = torch.randn(1, 3, 224, 224, requires_grad=True)
model_path = "test_resnet18.pth"
model = resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))

torch_out = torch.onnx.export(model, example,  "test_resnet18.onnx", verbose=False)
