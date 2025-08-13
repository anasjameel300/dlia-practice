import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess the image
img_path = "sample_img.png"
img = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
input_tensor = transform(img).unsqueeze(0)  # Shape: (1, 3, 128, 128)

# Define CNN with 3 convolutional blocks, no fully connected layers
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.cb1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.cb2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.cb3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        fmap1 = self.cb1(x)
        fmap2 = self.cb2(fmap1)
        fmap3 = self.cb3(fmap2)
        return fmap1, fmap2, fmap3

# Instantiate model and get feature maps
model = SimpleCNN()
with torch.no_grad():
    fmap1, fmap2, fmap3 = model(input_tensor)

# Function to plot feature maps
def plot_feature_maps(fmap, title):
    fmap = fmap.squeeze(0)  # Remove batch dimension
    num_channels = fmap.shape[0]
    cols = 8
    rows = (num_channels + cols - 1) // cols
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(num_channels):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(fmap[i].cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Plot feature maps after each convolutional block
plot_feature_maps(fmap1, "Feature Maps after CB1")
plot_feature_maps(fmap2, "Feature Maps after CB2")
plot_feature_maps(fmap3, "Feature Maps after CB3")
