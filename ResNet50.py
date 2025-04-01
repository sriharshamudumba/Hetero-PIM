import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

# Check for CUDA and device name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    if "A100" in torch.cuda.get_device_name(0):
        print(" GPU is A100 ")
    else:
        print(" Not using A100 GPU")
else:
    print("Using CPU")

# Parameters
epochs = 10
batch_size = 128
learning_rate = 0.001

# Transform for CIFAR-10
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Required for ResNet input
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataloaders
train_loader = DataLoader(datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
                          batch_size=batch_size, shuffle=True, num_workers=4)

test_loader = DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
                         batch_size=batch_size, shuffle=False, num_workers=4)

# Load ResNet-50 and modify final layer
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")

# Testing
model.eval()
correct = 0
total = 0

start_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
end_time = time.time()

accuracy = 100 * correct / total
inference_time = end_time - start_time

# Results
print(f"\n Final Accuracy: {accuracy:.2f}%")
print(f"Total Inference Time: {inference_time:.2f} seconds")

