import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import json

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    if "A100" in torch.cuda.get_device_name(0):
        print("GPU is A100")
else:
    print("Using CPU")

# BranchyResNet50 Definition
class BranchyResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(BranchyResNet, self).__init__()

        resnet = models.resnet50(pretrained=True)

        # Extract layers from ResNet50
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # Output: 256
        self.layer2 = resnet.layer2  # Output: 512
        self.layer3 = resnet.layer3  # Output: 1024
        self.layer4 = resnet.layer4  # Output: 2048

        self.exit1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)  # Deeper exit (after layer2)
        )

        self.exit2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)  # After layer3
        )

        self.exit3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes)  # After layer4
        )

        self.threshold1 = 2.0
        self.threshold2 = 2.0
        self.threshold3 = 2.0

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out1 = self.exit1(x)
        if self._entropy(out1) < self.threshold1:
            return out1, 'Exit1'

        x = self.layer3(x)
        out2 = self.exit2(x)
        if self._entropy(out2) < self.threshold2:
            return out2, 'Exit2'

        x = self.layer4(x)
        out3 = self.exit3(x)
        if self._entropy(out3) < self.threshold3:
            return out3, 'Exit3'

        return out3, 'Main'

    def _entropy(self, output):
        probs = F.softmax(output, dim=1)
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_loader = DataLoader(
    datasets.CIFAR100(root='./data', train=True, download=True, transform=transform),
    batch_size=128, shuffle=True, num_workers=4)

test_loader = DataLoader(
    datasets.CIFAR100(root='./data', train=False, download=True, transform=transform),
    batch_size=128, shuffle=False, num_workers=4)

# Model setup
model = BranchyResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 10
train_losses = []
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

# Testing
model.eval()
correct = 0
exit_correct = {'Exit1': 0, 'Exit2': 0, 'Exit3': 0, 'Main': 0}
exit_total = {'Exit1': 0, 'Exit2': 0, 'Exit3': 0, 'Main': 0}

start_time = time.time()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output, exit_point = model(data)
        pred = output.argmax(dim=1)
        correct_preds = pred.eq(target)
        correct += correct_preds.sum().item()
        exit_correct[exit_point] += correct_preds.sum().item()
        exit_total[exit_point] += target.size(0)
end_time = time.time()

overall_accuracy = 100. * correct / len(test_loader.dataset)
print(f"\nFinal Accuracy (Overall): {overall_accuracy:.2f}%")
print(f"Total Inference Time (BranchyNet): {end_time - start_time:.2f} seconds")

# Accuracy at each exit
exit_acc = {}
for key in exit_total:
    acc = 100. * exit_correct[key] / exit_total[key] if exit_total[key] > 0 else 0
    exit_acc[key] = acc
    print(f"{key}: Accuracy = {acc:.2f}%, Samples = {exit_total[key]}")

# Save metrics for comparison
metrics = {
    "accuracy": overall_accuracy,
    "inference_time": round(end_time - start_time, 2),
    "exit_accuracy": exit_acc
}
with open("earlyexit_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("earlyexit_train_loss.png")
print("Training loss plot saved as earlyexit_train_loss.png")

