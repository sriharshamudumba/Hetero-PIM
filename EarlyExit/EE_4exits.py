import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {torch.cuda.get_device_name(0)}")

# === DATA LOADING (CIFAR-100) ===
def prepare_cifar100(batch_size=128):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
    ])
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, test_loader

# === BRANCHY RESNET50 ===
class BranchyResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        base = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        base.fc = nn.Identity()
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.exit1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes))
        self.exit2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
        self.exit3 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1024, num_classes))
        self.exit4 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2048, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        out1 = self.exit1(x)
        x = self.layer2(x)
        out2 = self.exit2(x)
        x = self.layer3(x)
        out3 = self.exit3(x)
        x = self.layer4(x)
        out4 = self.exit4(x)
        return out1, out2, out3, out4

# === ENTROPY FUNCTION ===
def entropy(x):
    probs = torch.softmax(x, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-6), dim=1)

# === VALIDATION WITH EXIT DISTRIBUTION ===
def validate_with_exits(model, loader, thresholds):
    model.eval()
    correct = [0] * 4
    counts = [0] * 4
    total_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out1, out2, out3, out4 = model(images)

            ent1 = entropy(out1)
            ent2 = entropy(out2)
            ent3 = entropy(out3)

            exits = torch.full((images.size(0),), 4, dtype=torch.int, device=device)
            exits[ent1 < thresholds[0]] = 1
            remaining = exits == 4
            idx2 = torch.where(remaining)[0]
            exits[idx2[ent2[remaining] < thresholds[1]]] = 2
            remaining = exits == 4
            idx3 = torch.where(remaining)[0]
            exits[idx3[ent3[remaining] < thresholds[2]]] = 3

            preds = torch.empty_like(labels)
            preds[exits == 1] = out1[exits == 1].argmax(dim=1)
            preds[exits == 2] = out2[exits == 2].argmax(dim=1)
            preds[exits == 3] = out3[exits == 3].argmax(dim=1)
            preds[exits == 4] = out4[exits == 4].argmax(dim=1)

            for i in range(4):
                correct[i] += (preds[exits == (i+1)] == labels[exits == (i+1)]).sum().item()
                counts[i] += (exits == (i+1)).sum().item()
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        'accuracy': 100 * total_correct / total,
        'exit_counts': counts,
        'exit_accuracies': [100 * correct[i] / counts[i] if counts[i] else 0 for i in range(4)]
    }

# === EXPORT ONNX ===
def export_to_onnx(model):
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model, dummy_input, "branchy_resnet50_4exits.onnx",
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['exit1', 'exit2', 'exit3', 'exit4']
    )
    print("ONNX export complete: branchy_resnet50_4exits.onnx")

# === MAIN ===
if __name__ == "__main__":
    print("Dataset: CIFAR-100")
    train_loader, test_loader = prepare_cifar100()
    model = BranchyResNet50().to(device)

    ckpt_path = "branchy_resnet50_cifar100_4exits.pth"
    if os.path.exists(ckpt_path):
        print("\n Model found. Skipping training and loading weights...")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print("\n Training new 4-exit BranchyResNet50 from scratch...")
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler()

        for epoch in range(1, 51):
            model.train()
            total, correct1, correct2, correct3, correct4 = 0, 0, 0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda'):
                    out1, out2, out3, out4 = model(images)
                    l1 = criterion(out1, labels)
                    l2 = criterion(out2, labels)
                    l3 = criterion(out3, labels)
                    l4 = criterion(out4, labels)
                    total_loss = 0.2*l1 + 0.2*l2 + 0.2*l3 + 0.4*l4
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total += labels.size(0)
                correct1 += (out1.argmax(1) == labels).sum().item()
                correct2 += (out2.argmax(1) == labels).sum().item()
                correct3 += (out3.argmax(1) == labels).sum().item()
                correct4 += (out4.argmax(1) == labels).sum().item()

            print(f"Epoch {epoch}/50")
            print(f"Train Exit1 Acc: {100*correct1/total:.2f}%, Exit2 Acc: {100*correct2/total:.2f}%, Exit3 Acc: {100*correct3/total:.2f}%, Final Acc: {100*correct4/total:.2f}%")
            print("-"*70)

        print(f"\n Saving trained model to: {ckpt_path}")
        torch.save(model.state_dict(), ckpt_path)

    print("\n Training completed! Starting early exit evaluation...")
    thresholds_list = [
        (0.5, 1.0, 1.5),
        (1.0, 1.5, 2.0),
        (1.5, 2.0, 2.5),
        (2.0, 2.5, 3.0),
        (2.5, 3.0, 3.5)
    ]

    with open("early_exit_detailed_results_4exits.txt", "w") as f:
        f.write("Early Exit Evaluation - 4 Exits on CIFAR-100\n")
        f.write("="*70 + "\n")

        for th in thresholds_list:
            results = validate_with_exits(model, test_loader, thresholds=th)
            f.write(f"\nThresholds: Exit1 < {th[0]}, Exit2 < {th[1]}, Exit3 < {th[2]}\n")
            for i in range(4):
                f.write(f"Exit{i+1}: {results['exit_counts'][i]} samples | Acc: {results['exit_accuracies'][i]:.2f}%\n")
            f.write(f"Overall Accuracy: {results['accuracy']:.2f}%\n")
            f.write("-"*70 + "\n")

    export_to_onnx(model)
    print("\n Evaluation and ONNX export complete.")

