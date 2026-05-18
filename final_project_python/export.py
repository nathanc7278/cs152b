import torch
import numpy as np
from model import CNN
from torchvision import datasets, transforms
import torch.nn.functional as F

model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth', map_location='cpu'))
model.eval()
state = model.state_dict()
# print(state)

data = datasets.MNIST('./data', train=True, download=True,
                      transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)

all_conv1 = []
all_conv2 = []
all_fc1   = []
all_fc2   = []

with torch.no_grad():
    for i, (img, _) in enumerate(loader):
        x = img                                                          # (N,1,28,28)
        x = F.conv2d(x, state['conv1.weight'], state['conv1.bias'], padding=1)
        all_conv1.append(x.numpy().flatten())
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = F.conv2d(x, state['conv2.weight'], state['conv2.bias'], padding=1)
        all_conv2.append(x.numpy().flatten())
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.flatten(1)
        x = F.linear(x, state['fc1.weight'], state['fc1.bias'])
        all_fc1.append(x.numpy().flatten())
        x = F.relu(x)

        x = F.linear(x, state['fc2.weight'], state['fc2.bias'])
        all_fc2.append(x.numpy().flatten())

        if i >= 15:
            break

def to_scale(vals):
    max_val = np.percentile(np.abs(np.concatenate(vals)), 99.9)
    return max_val / 127.0

input_scale = 1.0 / 127.0
conv1_out_scale = to_scale(all_conv1)
conv2_out_scale = to_scale(all_conv2)
fc1_out_scale = to_scale(all_fc1)
fc2_out_scale = to_scale(all_fc2)

layer_input_scales = {
    'conv1': input_scale,
    'conv2': conv1_out_scale,
    'fc1':   conv2_out_scale,
    'fc2':   fc1_out_scale,
}

# original weights = quantized * scale
def quantize_channel(tensor, bits=8):
    arr = tensor.numpy()
    max_vals = np.max(np.abs(arr.reshape(arr.shape[0], -1)), axis=1)
    scales = max_vals / (2**(bits-1) - 1)     # 2^7 - 1 = 127, map the largest value to the highest int8
    q = np.zeros_like(arr, dtype=np.int8)
    for i, s in enumerate(scales):
        q[i] = np.clip(np.round(arr[i] / s), -128, 127)
    return q, scales

# makes it so that Sb = Sw * Sx
# Ex: MLP
# y_float = Sx * Sw * acc + Sb * bq
#         = Sx * Sw * acc + Sx * Sw * bq
#         = Sx * Sw * (acc + bq)
# => y_q = round(y_float / Sy)
def quantize_bias(bias_tensor, input_scale, weight_scales):
    arr = bias_tensor.numpy()
    q = np.zeros_like(arr, dtype=np.int32)
    for i, ws in enumerate(weight_scales):
        bias_scale = input_scale * ws
        q[i] = int(np.clip(np.round(arr[i] / bias_scale), -(2**31), 2**31 - 1))
    return q

weight_layers = ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']
bias_layers   = ['conv1.bias',   'conv2.bias',   'fc1.bias',   'fc2.bias']

scales = {}
quantized = {}

for name in weight_layers:
    q, s = quantize_channel(state[name])
    print(f"{name}: weight shape {state[name].shape}, scales shape {s.shape}")
    renamed = name.replace('.', '_')        # conv1.weight -> conv1_weight
    quantized[renamed] = q
    scales[renamed] = s
    

for name in bias_layers:
    layer = name.split('.')[0]
    renamed = name.replace('.', '_')
    in_scale = layer_input_scales[layer]
    weight_scales = scales[f'{layer}_weight']
    quantized[renamed] = quantize_bias(state[name], in_scale, weight_scales)

with open('weights.h', 'w') as f:
    f.write('#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n')
    f.write('#include <stdint.h>\n\n')
    f.write('// Weights: int8 per-channel quantized\n')
    f.write('// Biases:  int32, scale = input_scale * w_scale[oc]\n')
    f.write('// Inference: acc * (input_scale * w_scale[oc]) / output_scale\n\n')

    # activation scales
    f.write(f'static const float input_scale = {input_scale:.8e}f;\n')
    f.write(f'static const float conv1_out_scale = {conv1_out_scale:.8e}f;\n')
    f.write(f'static const float conv2_out_scale = {conv2_out_scale:.8e}f;\n')
    f.write(f'static const float fc1_out_scale = {fc1_out_scale:.8e}f;\n')
    f.write(f'static const float fc2_out_scale = {fc2_out_scale:.8e}f;\n\n')

    # per-channel weight scales
    for key, s in scales.items():
        vals = ', '.join(f'{v:.8e}f' for v in s)
        f.write(f'static const float {key}_scale[{len(s)}] = {{{vals}}};\n')

    f.write('\n')

    # weight arrays (int8)
    for name in weight_layers:
        key = name.replace('.', '_')
        q = quantized[key]
        flat = q.flatten()
        f.write(f'// shape: {q.shape}\n')
        f.write(f'static const int8_t {key}[{len(flat)}] = {{\n')
        for i in range(0, len(flat), 16):
            chunk = ', '.join(str(v) for v in flat[i:i+16])
            f.write(f'    {chunk},\n')
        f.write('};\n\n')

    # bias arrays (int32)
    for name in bias_layers:
        key = name.replace('.', '_')
        q = quantized[key]
        flat = q.flatten()
        f.write(f'// shape: {q.shape}\n')
        f.write(f'static const int32_t {key}[{len(flat)}] = {{\n')
        for i in range(0, len(flat), 16):
            chunk = ', '.join(str(v) for v in flat[i:i+16])
            f.write(f'    {chunk},\n')
        f.write('};\n\n')

    f.write('#endif')

    
    