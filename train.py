import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os

def get_input_args():
    """
    Parse command-line arguments for training.
    """
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset.")
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory.')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save the model checkpoint.')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16, resnet18, etc.).')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training.')
    return parser.parse_args()

def load_data(data_dir):
    """
    Load the training, validation, and testing datasets.
    """
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    
    return train_loader, valid_loader, train_dataset.class_to_idx

def build_model(arch, hidden_units, learning_rate):
    """
    Build and return a pretrained model with a custom classifier.
    """
    # Load a pretrained model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = 512
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Freeze feature extraction layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a new classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    if arch == 'vgg16':
        model.classifier = classifier
    elif arch == 'resnet18':
        model.fc = classifier
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() if arch == 'vgg16' else model.fc.parameters(), lr=learning_rate)
    
    return model, optimizer, criterion

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs, gpu):
    """
    Train the model and validate after each epoch.
    """
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        train_correct = 0
        
        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()
        
        # Validation loop
        model.eval()
        valid_loss = 0
        valid_correct = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                valid_correct += torch.sum(preds == labels.data).item()
        
        # Print epoch statistics
        train_accuracy = train_correct / len(train_loader.dataset)
        valid_accuracy = valid_correct / len(valid_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Train Loss: {running_loss / len(train_loader.dataset):.3f}.. "
              f"Train Accuracy: {train_accuracy:.3f}.. "
              f"Validation Loss: {valid_loss / len(valid_loader.dataset):.3f}.. "
              f"Validation Accuracy: {valid_accuracy:.3f}")

def save_checkpoint(model, optimizer, save_dir, class_to_idx):
    """
    Save the model checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx,
        'arch': model.__class__.__name__
    }
    torch.save(checkpoint, save_dir)
    print(f"Checkpoint saved to {save_dir}")

def main():
    args = get_input_args()
    train_loader, valid_loader, class_to_idx = load_data(args.data_dir)
    model, optimizer, criterion = build_model(args.arch, args.hidden_units, args.learning_rate)
    train_model(model, train_loader, valid_loader, criterion, optimizer, args.epochs, args.gpu)
    save_checkpoint(model, optimizer, args.save_dir, class_to_idx)

if __name__ == "__main__":
    main()
