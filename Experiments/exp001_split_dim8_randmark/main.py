import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(input_dim=784, hidden_dim=256, output_dim=128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        h = self.encoder(x)
        logits = self.classifier(h)
        return logits

def train_baseline(model, train_loader, epochs=5):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

print("\n=== Baseline Model ===")
baseline = BaselineModel().to(device)
train_baseline(baseline, train_loader, epochs=5)
baseline_acc = evaluate(baseline, test_loader)
print(f"Baseline Accuracy: {baseline_acc:.2f}")

class ServerNetwork(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, h_split):
        return self.net(h_split)

class SplitModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.netA = ServerNetwork(input_dim=64, hidden_dim=32)
        self.netB = ServerNetwork(input_dim=64, hidden_dim=32)
        self.final_classifier = nn.Linear(64, 10)

    def forward(self, x, return_splits=False):
        with torch.no_grad():
            h = self.encoder(x)

        h1 = h[:, :64]
        h2 = h[:, 64:]

        z1 = self.netA(h1)
        z2 = self.netB(h2)

        z = torch.cat([z1, z2], dim=1)
        logits = self.final_classifier(z)

        if return_splits:
            return logits, z1, z2
        return logits

def train_split(model, train_loader, epochs=5):
    model.train()
    optimizer = optim.Adam(
        list(model.netA.parameters()) + 
        list(model.netB.parameters()) +
        list(model.final_classifier.parameters()), 
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

def evaluate_split_all(model, test_loader):
    model.eval()

    correct_combined = 0
    correct_A_only = 0
    correct_B_only = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            h = model.encoder(images)
            h1 = h[:, :64]
            h2 = h[:, 64:]

            z1 = model.netA(h1)
            z2 = model.netB(h2)

            z_combined = torch.cat([z1, z2], dim=1)
            logits_combined = model.final_classifier(z_combined)
            _, pred_combined =torch.max(logits_combined, 1)
            correct_combined += (pred_combined == labels).sum().item()

            z_A_only = torch.cat([z1, torch.zeros_like(z2)], dim=1)
            logits_A = model.final_classifier(z_A_only)
            _, pred_A = torch.max(logits_A, 1)
            correct_A_only += (pred_A == labels).sum().item()
            
            z_B_only = torch.cat([torch.zeros_like(z1), z2], dim=1)
            logits_B = model.final_classifier(z_B_only)
            _, pred_B = torch.max(logits_B, 1)
            correct_B_only += (pred_B == labels).sum().item()
            
            total += labels.size(0)

    acc_combined = 100 * correct_combined / total
    acc_A = 100 * correct_A_only / total
    acc_B = 100 * correct_B_only / total

    return acc_combined, acc_A, acc_B

print("\n=== Split Model ===")
split_model = SplitModel(baseline.encoder).to(device)
train_split(split_model, train_loader, epochs=5)

acc_combined, acc_A, acc_B = evaluate_split_all(split_model, test_loader)
print(f"\nSplit Model Results:")
print(f"  Combined (z1+z2): {acc_combined:.2f}%")
print(f"  Server A only:    {acc_A:.2f}%")
print(f"  Server B only:    {acc_B:.2f}%")
