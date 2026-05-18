import torch
import numpy as np
from torchvision import datasets, transforms
from model import CNN
import matplotlib.pyplot as plt

NUM_IMAGES = 1
IMAGE_IDX = 10

model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])
test_data = datasets.MNIST('.', train=False, download=True, transform=transform)

def to_int8(image_tensor):
    arr = image_tensor.numpy().flatten() # 784 floats in [0, 1]
    arr = (arr * 255).astype(np.uint8) # [0, 255]
    arr = (arr.astype(np.int16) - 128).astype(np.int8) # [-128, 127]
    return arr


def python_predict(image_tensor):
    with torch.no_grad():
        out = model(image_tensor.unsqueeze(0))
        return out.argmax().item()

image, label = test_data[IMAGE_IDX]
arr = to_int8(image)
pred = python_predict(image)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# original image
axes[0].imshow(image.squeeze(), cmap='gray')
axes[0].set_title(f'Original | label={label} | pred={pred}')
axes[0].axis('off')

# what the C code sees (int8, shifted back for display)
axes[1].imshow(arr.reshape(28, 28).astype(np.int16) + 128, cmap='gray')
axes[1].set_title('As int8 (C input)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('test_image.png')
plt.show()

arr = (image.numpy().flatten() * 127).astype(np.int8)
print(arr.shape)

with open('test_images.h', 'w') as f:
    f.write('#ifndef TEST_IMAGE_H\n')
    f.write('#define TEST_IMAGE_H\n\n')
    f.write('#include <stdint.h>\n\n')
    f.write(f'static const int true_label = {label};\n\n')
    f.write('static const int8_t test_image[784] = {\n')
    for i in range(0, 784, 16):
        line = ', '.join(str(v) for v in arr[i:i+16])
        f.write(f'    {line},\n')
    f.write('};\n\n')
    f.write('#endif\n')