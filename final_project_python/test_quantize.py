import torch
import numpy as np
from model import CNN
from torchvision import datasets, transforms
import torch.nn.functional as F

model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth', map_location='cpu'))
model.eval()
state = model.state_dict()

# ── load weights.h values (must match your actual weights.h) ──
# re-run your export script and copy the scales here, or just
# re-quantize inline so they're guaranteed to match

data_cal = datasets.MNIST('./data', train=True, download=True,
                          transform=transforms.ToTensor())
loader_cal = torch.utils.data.DataLoader(data_cal, batch_size=64, shuffle=False)

all_conv1, all_conv2, all_fc1, all_fc2 = [], [], [], []
with torch.no_grad():
    for i, (img, _) in enumerate(loader_cal):
        x = img
        x = F.conv2d(x, state['conv1.weight'], state['conv1.bias'], padding=1)
        all_conv1.append(x.numpy().flatten())
        x = F.relu(x); x = F.max_pool2d(x, 2)
        x = F.conv2d(x, state['conv2.weight'], state['conv2.bias'], padding=1)
        all_conv2.append(x.numpy().flatten())
        x = F.relu(x); x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.linear(x, state['fc1.weight'], state['fc1.bias'])
        all_fc1.append(x.numpy().flatten())
        x = F.relu(x)
        x = F.linear(x, state['fc2.weight'], state['fc2.bias'])
        all_fc2.append(x.numpy().flatten())
        if i >= 15:
            break

def to_scale(vals):
    return np.percentile(np.abs(np.concatenate(vals)), 99.9) / 127.0

input_scale     = 1.0 / 127.0
conv1_out_scale = to_scale(all_conv1)
conv2_out_scale = to_scale(all_conv2)
fc1_out_scale   = to_scale(all_fc1)
fc2_out_scale   = to_scale(all_fc2)

layer_input_scales = {
    'conv1': input_scale,
    'conv2': conv1_out_scale,
    'fc1':   conv2_out_scale,
    'fc2':   fc1_out_scale,
}

def quantize_channel(tensor):
    arr = tensor.numpy()
    max_vals = np.max(np.abs(arr.reshape(arr.shape[0], -1)), axis=1)
    scales = max_vals / 127.0
    q = np.zeros_like(arr, dtype=np.int8)
    for i, s in enumerate(scales):
        q[i] = np.clip(np.round(arr[i] / s), -128, 127)
    return q, scales

def quantize_bias(bias_tensor, inp_scale, w_scales):
    arr = bias_tensor.numpy()
    q = np.zeros_like(arr, dtype=np.int32)
    for i, ws in enumerate(w_scales):
        bias_scale = inp_scale * ws
        q[i] = int(np.clip(np.round(arr[i] / bias_scale), -(2**31), 2**31 - 1))
    return q

w_scales  = {}
quantized = {}

for name in ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']:
    q, s = quantize_channel(state[name])
    key = name.replace('.', '_')
    quantized[key] = q
    w_scales[key]  = s

for name in ['conv1.bias', 'conv2.bias', 'fc1.bias', 'fc2.bias']:
    layer = name.split('.')[0]
    key   = name.replace('.', '_')
    inp_scale = layer_input_scales[layer]
    ws        = w_scales[f'{layer}_weight']
    quantized[key] = quantize_bias(state[name], inp_scale, ws)

# ── mirror C functions exactly ────────────────────────────────
def clamp_int8(x):
    return int(np.clip(x, -128, 127))

def c_conv2d(inp, H, W, in_ch, out_ch, weights, bias, w_scales, inp_scale, out_scale):
    output = np.zeros((out_ch, H, W), dtype=np.int8)
    for oc in range(out_ch):
        M = (inp_scale * w_scales[oc]) / out_scale
        for y in range(H):
            for x in range(W):
                acc = np.int32(0)
                for ic in range(in_ch):
                    for ky in range(3):
                        for kx in range(3):
                            iy, ix = y + ky - 1, x + kx - 1
                            if iy < 0 or iy >= H or ix < 0 or ix >= W:
                                continue
                            w_idx = ((oc * in_ch + ic) * 3 + ky) * 3 + kx
                            acc += np.int32(inp[ic, iy, ix]) * np.int32(weights.flat[w_idx])
                acc += bias[oc]
                output[oc, y, x] = clamp_int8(int(round(float(acc) * M)))
    return output

def c_relu(x):
    return np.clip(x, 0, 127).astype(np.int8)

def c_maxpool2d(inp, ch, H, W):
    out = np.full((ch, H//2, W//2), -128, dtype=np.int8)
    for c in range(ch):
        for y in range(0, H, 2):
            for x in range(0, W, 2):
                out[c, y//2, x//2] = np.max(inp[c, y:y+2, x:x+2])
    return out

def c_fc(inp, in_size, out_size, weights, bias, w_scales, inp_scale, out_scale):
    out = np.zeros(out_size, dtype=np.int8)
    for o in range(out_size):
        M = (inp_scale * w_scales[o]) / out_scale
        acc = np.int32(0)
        for i in range(in_size):
            acc += np.int32(inp[i]) * np.int32(weights[o, i])
        acc += bias[o]
        out[o] = clamp_int8(int(round(float(acc) * M)))
    return out

def c_infer(flat_input):
    buf = flat_input.reshape(1, 28, 28)

    buf = c_conv2d(buf, 28, 28, 1, 4,
                   quantized['conv1_weight'], quantized['conv1_bias'],
                   w_scales['conv1_weight'], input_scale, conv1_out_scale)
    buf = c_relu(buf)
    buf = c_maxpool2d(buf, 4, 28, 28)

    buf = c_conv2d(buf, 14, 14, 4, 8,
                   quantized['conv2_weight'], quantized['conv2_bias'],
                   w_scales['conv2_weight'], conv1_out_scale, conv2_out_scale)
    buf = c_relu(buf)
    buf = c_maxpool2d(buf, 8, 14, 14)

    flat = buf.flatten()
    flat = c_fc(flat, 8*7*7, 16,
                quantized['fc1_weight'].reshape(16, -1), quantized['fc1_bias'],
                w_scales['fc1_weight'], conv2_out_scale, fc1_out_scale)
    flat = c_relu(flat)

    flat = c_fc(flat, 16, 10,
                quantized['fc2_weight'].reshape(10, -1), quantized['fc2_bias'],
                w_scales['fc2_weight'], fc1_out_scale, fc2_out_scale)

    return int(np.argmax(flat)), flat

# ── evaluate ──────────────────────────────────────────────────
test_data = datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.ToTensor())

correct_float = 0
correct_quant = 0
mismatches    = 0
n = 500

for i in range(n):
    img, label = test_data[i]

    with torch.no_grad():
        float_pred = int(torch.argmax(model(img.unsqueeze(0))))

    inp_int8 = (img.numpy().flatten() * 127).astype(np.int8)
    quant_pred, _ = c_infer(inp_int8)

    correct_float += (float_pred == label)
    correct_quant += (quant_pred == label)
    mismatches    += (float_pred != quant_pred)

print(f"Samples:        {n}")
print(f"Float accuracy: {correct_float/n*100:.1f}%")
print(f"Quant accuracy: {correct_quant/n*100:.1f}%")
print(f"Mismatches:     {mismatches} ({mismatches/n*100:.1f}%)")