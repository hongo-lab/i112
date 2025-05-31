
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

# ハイパーパラメータ設定
batch_size = 64
epochs = 10
learning_rate = 0.001

# 学習とテストの処理を定義
def train_and_evaluate(device, train_loader, test_loader):
    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3)
            self.conv2 = nn.Conv2d(32, 64, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 5 * 5, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    model = ConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    start = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"[{device}] Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    elapsed = time.time() - start

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    accuracy = correct / total
    return elapsed, accuracy, train_losses

# データセットの準備
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader_cpu = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader_cpu = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# CPU 実行
cpu_device = torch.device('cpu')
cpu_time, cpu_acc, cpu_loss = train_and_evaluate(cpu_device, train_loader_cpu, test_loader_cpu)
print(f"[CPU] Time: {cpu_time:.2f}s | Accuracy: {cpu_acc:.4f}")

# GPU 実行（もし使用可能なら）
if torch.cuda.is_available():
    gpu_device = torch.device('cuda')
    train_loader_gpu = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader_gpu = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    gpu_time, gpu_acc, gpu_loss = train_and_evaluate(gpu_device, train_loader_gpu, test_loader_gpu)
    print(f"[GPU] Time: {gpu_time:.2f}s | Accuracy: {gpu_acc:.4f}")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")

    # 学習曲線の可視化（GPUとCPUを比較）
    plt.plot(cpu_loss, label='CPU')
    plt.plot(gpu_loss, label='GPU')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_comparison.png')
    print("Training loss curve saved as 'training_loss_comparison.png'")
else:
    print("GPU not available.")
