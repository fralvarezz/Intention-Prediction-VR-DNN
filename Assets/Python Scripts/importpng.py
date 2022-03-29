import torch
import torch.nn as nn
import torch.onnx
import onnx
import torchvision
import torchvision.transforms as transforms
import onnx2pytorch
from torch.utils.data import DataLoader
from PIL import Image

img = Image.open('./blank/391.png')

convert_tensor = transforms.ToTensor()

tens = convert_tensor(img)
tens = tens.resize(1,28,28)
print(tens.size())

onnx_model = onnx.load('../Models/model2_batch1.onnx')
pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental="True")

with torch.no_grad():
    out = pytorch_model(tens)
    _, predicted = torch.max(out.data, 1)
    print(predicted)