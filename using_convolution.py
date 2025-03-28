import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from data_prep import cnn_preprocessing, train_dir, validation_dir, CNN, ImageDataset


# Training function with early stopping at 95% accuracy
def train_model(model, train_loader, val_loader, criterion, optimizer, device, max_epochs=50):
    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            _, outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # What does this do?
            _, predicted = torch.max(outputs, 1)
            labels = torch.argmax(labels, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train

        #Validation Step
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                labels = torch.argmax(labels, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
        val_acc = 100 * correct_val / total_val

        print(f"Epoch {epoch+1}: Loss = {running_loss / len(train_loader):.4f}, Train Accuracy = {train_acc:.2f}%, Validation Accuracy = {val_acc:.2f}%, Time = {time.time() - start_time:.2f}s")

        # Early stopping condition
        if val_acc >= 95.0:
            print("\nTraining stopped early as validation accuracy reached 95%")
            break

if __name__=="__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for processing")

    # Load the training and validation set
    train_data, train_label = cnn_preprocessing(train_dir)
    val_data, val_label = cnn_preprocessing(validation_dir)

    # Assuming X_train, y_train, X_val, y_val are NumPy arrays
    train_dataset = ImageDataset(train_data, train_label)
    val_dataset = ImageDataset(val_data, val_label)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Model initialization
    model = CNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device)

    # Save the trained model
    torch.save(model.state_dict(), "cnn_model.pth")
    print("Model saved as cnn_model.pth")