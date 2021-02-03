import torch
import torch.optim as optim
import torch.nn as nn

from model import get_model
from model import device

from tqdm import tqdm


# Fitting and Evaluating
def train(model, train_loader, test_loader, epochs=10, lr=3e-4):

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Learning rate sheduler
    sheduler = optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.5)

    # Statistics
    train_LOSS = []
    train_ACC = []
    test_LOSS = []
    test_ACC = []

    best_acc_score = 0.

    for epoch in range(epochs):

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = test_loader

            loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)

            running_loss = 0.
            running_acc = 0.

            for idx, (inputs, labels) in loop:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    predicts = model(inputs)
                    loss = criterion(predicts, labels)
                    predict_class = predicts.argmax(dim=1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    running_acc += (predict_class == labels.data).float().mean()

                    loop.set_description(f'Epoch [{epoch+1}/{epochs}] {phase}')
                    loop.set_postfix(loss=running_loss/(idx+1), acc=running_acc.item()/(idx+1))

            epoch_loss = running_loss/len(data_loader)
            epoch_acc = running_acc / len(data_loader)

            if phase == 'train':
                train_LOSS.append(epoch_loss)
                train_ACC.append(epoch_acc)
                sheduler.step()
            else:
                test_LOSS.append(epoch_loss)
                test_ACC.append(epoch_acc)
                if best_acc_score < epoch_acc:
                    best_acc_score = epoch_acc
                    torch.save(model.state_dict(), 'model/model_config.ptn')

    return model, (train_LOSS, train_ACC, test_LOSS, test_ACC)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    import os

    model = get_model()
    print('Starting train model on', device)

    # Set transforms to train dataset
    train_transforms = T.Compose([
        T.RandomRotation(45, fill=0),
        T.RandomPerspective(distortion_scale=0.3, p=0.7),
        T.Resize(32),
        T.ToTensor()
    ])

    # Set transforms to test dataset
    test_transforms = T.Compose([
        T.RandomRotation(45, fill=0),
        T.RandomPerspective(distortion_scale=0.3, p=0.7),
        T.Resize(32),
        T.ToTensor()
    ])

    # Loading datasets
    data_train = torchvision.datasets.MNIST('data/train/', train=True, download=True,
                                            transform=train_transforms)
    data_test = torchvision.datasets.MNIST('data/test/', train=False, download=True,
                                           transform=test_transforms)

    # Setup loaders
    train_loader = DataLoader(dataset=data_train, pin_memory=True,
                              batch_size=256, shuffle=True, drop_last=True)

    test_loader = DataLoader(dataset=data_test, pin_memory=True,
                             batch_size=256, shuffle=False)

    if not os.path.exists('model'):
        os.mkdir('model')

    model, (train_loss, train_acc, test_loss, test_acc) = train(model, train_loader, test_loader, epochs=10)

    print('Model trained successfully\nBest model saved at model/model.ptn')
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    test_color = 'r'
    train_color = 'g'

    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(train_loss, color=train_color, label='train')
    ax1.plot(test_loss, color=test_color, label='test')
    ax1.legend()
    ax1.tick_params(axis='y')

    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.plot(train_acc, color=train_color, label='train')
    ax2.plot(test_acc, color=test_color, label='test')
    ax2.legend()
    ax2.tick_params(axis='y')

    f.tight_layout()
    plt.show()
