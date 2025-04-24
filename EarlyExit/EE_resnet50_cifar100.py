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
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1)
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.exit1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
        self.exit2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1024, num_classes))
        self.final_fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer2(x)
        out1 = self.exit1(x)
        x = self.layer3(x)
        out2 = self.exit2(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        out_final = self.final_fc(x)
        return out1, out2, out_final

# === ENTROPY FUNCTION ===
def entropy(x):
    probs = torch.softmax(x, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-6), dim=1)

# === VALIDATION WITH EXIT DISTRIBUTION ===
def validate_with_exits(model, loader, thresholds):
    model.eval()
    exit1_correct = exit2_correct = final_correct = total_correct = 0
    exit1_count = exit2_count = final_count = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out1, out2, final_out = model(images)

            ent1 = entropy(out1)
            ent2 = entropy(out2)

            exits = torch.full((images.size(0),), 3, dtype=torch.int, device=device)
            exits[ent1 < thresholds[0]] = 1
            idx = torch.where(exits == 3)[0]
            exits[idx[ent2[idx] < thresholds[1]]] = 2

            preds = torch.empty_like(labels)
            preds[exits == 1] = out1[exits == 1].argmax(dim=1)
            preds[exits == 2] = out2[exits == 2].argmax(dim=1)
            preds[exits == 3] = final_out[exits == 3].argmax(dim=1)

            exit1_correct += (preds[exits == 1] == labels[exits == 1]).sum().item()
            exit2_correct += (preds[exits == 2] == labels[exits == 2]).sum().item()
            final_correct += (preds[exits == 3] == labels[exits == 3]).sum().item()
            total_correct += (preds == labels).sum().item()

            exit1_count += (exits == 1).sum().item()
            exit2_count += (exits == 2).sum().item()
            final_count += (exits == 3).sum().item()
            total += labels.size(0)

    return {
        'accuracy': 100 * total_correct / total,
        'exit1_count': exit1_count,
        'exit2_count': exit2_count,
        'final_count': final_count,
        'exit1_acc': 100 * exit1_correct / exit1_count if exit1_count else 0,
        'exit2_acc': 100 * exit2_correct / exit2_count if exit2_count else 0,
        'final_acc': 100 * final_correct / final_count if final_count else 0,
        'train_exit1_acc': 100 * exit1_correct / total if total else 0,
        'train_exit2_acc': 100 * exit2_correct / total if total else 0,
        'train_final_acc': 100 * final_correct / total if total else 0,
    }

# === MAIN ===
if __name__ == "__main__":
    print("Dataset: CIFAR-100")
    train_loader, test_loader = prepare_cifar100()
    model = BranchyResNet50().to(device)
    model.load_state_dict(torch.load("branchy_resnet50_cifar100_fulltrain.pth"))

    thresholds_list = [(0.5,1.0), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)]

    onnx_path = "branchy_resnet50_with_exits.onnx"
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'],
        output_names=['exit1', 'exit2', 'final_exit'],
        opset_version=11
    )

    with open("early_exit_detailed_results.txt", "w") as f:
        f.write("Early Exit Evaluation - Detailed Threshold Results on CIFAR-100\n")
        f.write("="*70 + "\n")

        for t1, t2 in thresholds_list:
            results = validate_with_exits(model, test_loader, thresholds=(t1, t2))
            f.write(f"\nThresholds: Exit1 < {t1}, Exit2 < {t2}\n")
            f.write("Classification Accuracy across exits (on Test set):\n")
            f.write(f"Exit1: {results['exit1_count']} samples | Acc: {results['exit1_acc']:.2f}%\n")
            f.write(f"Exit2: {results['exit2_count']} samples | Acc: {results['exit2_acc']:.2f}%\n")
            f.write(f"Final : {results['final_count']} samples | Acc: {results['final_acc']:.2f}%\n")
            f.write(f"Overall Accuracy (All exits): {results['accuracy']:.2f}%\n")
            f.write("-"*70 + "\n")

    print(" ONNX model saved to branchy_resnet50_with_exits.onnx")
    print(" Evaluation on thresholds complete. See 'early_exit_detailed_results.txt'")

