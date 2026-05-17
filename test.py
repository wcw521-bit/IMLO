import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import random

from model import PetClassifier

MODEL_PATH = 'best_model.pth'
NUM_CLASSES = 37
IMAGE_SIZE = 224
BATCH_SIZE = 64
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
           # transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            #transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_dataloaders(data_root='./data', val_fraction=0.1):
    full_train_dataset = OxfordIIITPet(
        root=data_root,
        split='trainval',
        target_types='category',
        download=True,
        transform=get_transforms(train=True),
    )

    total_size = len(full_train_dataset)
    val_size = int(total_size * val_fraction)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(SEED)

    train_split, val_split = random_split(
        range(total_size),
        [train_size, val_size],
        generator=generator,
    )

    train_dataset = Subset(full_train_dataset, train_split.indices)

    val_base_dataset = OxfordIIITPet(
        root=data_root,
        split='trainval',
        target_types='category',
        download=True,
        transform=get_transforms(train=False),
    )

    val_dataset = Subset(val_base_dataset, val_split.indices)

    test_dataset = OxfordIIITPet(
        root=data_root,
        split='test',
        target_types='category',
        download=True,
        transform=get_transforms(train=False),
    )

    device = get_device()

    common_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    set_seed()
    device = get_device()
    print(f'Using device: {device}')

    _, _, test_loader = get_dataloaders()

    model = PetClassifier(num_classes=37).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')


if __name__ == '__main__':
    main()
