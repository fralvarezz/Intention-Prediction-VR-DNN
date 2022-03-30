import onnx
import torch
'''
model = onnx.load('model9.onnx')
inputs = {}
for inp in model.graph.input:
    shape = str(inp.type.tensor_type.shape.dim)
    inputs[inp.name] = [int(s) for s in shape.split() if s.isdigit()]
'''

newtens = torch.randn(1,1,28,28)
print(newtens[0,0,:,0])
newtens = torch.transpose(newtens,2,3)
print(newtens[0,0,0,:])
