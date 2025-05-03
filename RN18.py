import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# 1. Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Set data directory path
data_dir = 'C:/Users/ADMIN/project/Processed_Dataset'  # Replace with the actual path

# 3. Set up image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a consistent size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
])

# 4. Load datasets for Train, Validation, and Test
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Validation'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Test'), transform=transform)

# 5. Create DataLoader for batching and shuffling
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 6. Check the number of samples in each dataset
print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# 7. Define the model (you can use a pre-trained model like ResNet18 here)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust the output layer for two classes (Fake and Real)

# 8. Move model to device (GPU if available)
model = model.to(device)

# 9. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 10. Training function
def train_model():
    NUM_EPOCHS = 10  # Set the number of epochs
    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        # Train for one epoch
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels)
            total_preds += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds.double() / total_preds

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Validation
        model.eval()
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels)
                total_preds += labels.size(0)

        val_accuracy = correct_preds.double() / total_preds
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save model if the validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print("Model saved!")

# 11. Call the train function
if __name__ == '__main__':
    train_model()

# 12. Testing the model
def test_model():
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels)
            total_preds += labels.size(0)

    test_accuracy = correct_preds.double() / total_preds
    print(f"Test Accuracy: {test_accuracy:.4f}")

# 13. Test the model after training
test_model()

