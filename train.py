import torch
import torch.nn as nn
import torch.optim as optim

from model import PetClassifier
from utils import set_seed, get_device, get_dataloaders, evaluate

EPOCHS = 30
LEARNING_RATE = 3e-4
MODEL_PATH = 'best_model.pth'


def main():
    set_seed()
    device = get_device()
    print(f'Using device: {device}')

    train_loader, val_loader, test_loader = get_dataloaders()

    model = PetClassifier(num_classes=37).to(device)

    # label smoothing helps generalisation
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f'Epoch {epoch + 1:02d}/{EPOCHS} | '
            f'LR: {current_lr:.2e} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Train Acc: {train_acc:.2f}% | '
            f'Val Loss: {val_loss:.4f} | '
            f'Val Acc: {val_acc:.2f}%'
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'Saved best model ({val_acc:.2f}% validation accuracy)')

    print('\nLoading best model...')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    train_loss, train_acc = evaluate(model, train_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print('\nFinal Results')
    print(f'Train Accuracy: {train_acc:.2f}%')
    print(f'Test Accuracy:  {test_acc:.2f}%')


if __name__ == '__main__':
    main()
