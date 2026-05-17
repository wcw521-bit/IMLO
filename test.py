import torch
import torch.nn as nn

from model import PetClassifier
from utils import set_seed, get_device, get_dataloaders, evaluate


MODEL_PATH = 'best_model.pth'


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
