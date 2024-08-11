import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F


# Define the neural network model
class CNNModel(nn.Module):
    def __init__(self, num_classes=62):  # Change this to 62
        super(CNNModel, self).__init__()
        # Define the layers here
        self.conv1 = nn.Conv2d(1, num_classes, kernel_size=5)  # First convolutional layer
        self.conv2 = nn.Conv2d(num_classes, num_classes*2, kernel_size=5) # Second convolutional layer
        self.fc1 = nn.Linear(num_classes * 2 * 4 * 4, num_classes * 4)         # First fully-connected layer
        self.fc2 = nn.Linear(num_classes * 4, num_classes)        # Final ouptu layer

    def forward(self, x):
        # first convolutional layer
        h_conv1 = self.conv1(x)
        h_conv1 = F.relu(h_conv1)
        h_conv1_pool = F.max_pool2d(h_conv1, 2)

        # second convolutional layer
        h_conv2 = self.conv2(h_conv1_pool)
        h_conv2 = F.relu(h_conv2)
        h_conv2_pool = F.max_pool2d(h_conv2, 2)

        # fully-connected layer
        h_fc1 = h_conv2_pool.view(-1, num_classes * 2 * 4 * 4)
        h_fc1 = self.fc1(h_fc1)
        h_fc1 = F.relu(h_fc1)

        # classifier output
        output = self.fc2(h_fc1)
        output = F.log_softmax(output,dim=1)
        return output, h_fc1, h_conv2, h_conv1

# Set up training parameters
batch_size = 1000
learning_rate = 0.001
epochs = 20

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the EMNIST dataset (ByClass split contains both digits and letters)
train_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Determine the number of classes
num_classes = len(train_dataset.classes)
print(f'Number of classes in the dataset: {num_classes}')

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model with the correct number of classes
model = CNNModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to GPU

        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass
        outputs, _, _, _ = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            progress_bar.set_postfix(loss=running_loss / 100)
            running_loss = 0.0

# Save the trained model
torch.save(model.state_dict(), 'emnist_cnn_model_new.pt')
print('Model saved as emnist_cnn_model_new.pt')

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to GPU
        outputs, _, _, _ = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')