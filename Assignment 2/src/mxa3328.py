import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define the LeNet-5 architecture
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # C1: Convolutional Layer (1 input channel, 6 output channels, 5x5 kernel)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # S2: Subsampling Layer (Average Pooling)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: Convolutional Layer (6 input channels, 16 output channels, 5x5 kernel)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # C5: Fully connected convolutional layer (16*5*5 input features, 120 output features)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # F6: Fully connected layer (120 input features, 84 output features)
        self.fc2 = nn.Linear(120, 84)
        # Output layer (84 input features, 10 output features for 10 classes)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Apply first convolution + activation + pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second convolution + activation + pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output to feed into fully connected layers
        x = x.view(-1, 16 * 5 * 5)  # Reshape tensor for fully connected layers
        # Apply first fully connected layer + activation
        x = F.relu(self.fc1(x))
        # Apply second fully connected layer + activation
        x = F.relu(self.fc2(x))
        # Apply output layer
        x = self.fc3(x)
        return x

# Define training parameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# MNIST dataset (from torchvision, for simplicity)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize MNIST images to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalization for MNIST
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the LeNet-5 model
model = LeNet5()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

# Evaluation function to test the model
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Evaluate the model on the test set
test_accuracy = evaluate(model, test_loader)
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Save the model and results for submission
torch.save(model.state_dict(), 'lenet5_mnist.pth')
