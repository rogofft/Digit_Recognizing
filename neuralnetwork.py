import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

from sklearn.metrics import accuracy_score

# Torch setup
#torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    # Ctor
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_2 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_3 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.linear_1 = nn.Linear(64 * 3 * 3, 100)
        self.linear_2 = nn.Linear(100, 10)

        self.softmax = nn.Softmax(dim=1)

        self.drop = nn.Dropout(p=0.6)
        self.eval()

    # Forward
    def forward(self, x):
        x = self.conv_1(x)
        x = torch.relu(x)
        x = self.max_pool_1(x)

        x = self.conv_2(x)
        x = torch.relu(x)
        x = self.max_pool_2(x)

        x = self.conv_3(x)
        x = torch.relu(x)
        x = self.max_pool_3(x)

        # resize for linear layer
        x = x.view(x.size(0), -1)

        x = torch.relu(self.drop(self.linear_1(x)))

        x = self.linear_2(x)
        return x

    # Fitting and Evaluating
    def fit(self, dataset_train, dataset_test, epochs, lr=0.01, batch_size=10):

        # Preparing
        trainloader = DataLoader(dataset=dataset_train, pin_memory=True,
                                 batch_size=batch_size, shuffle=True)

        trainloader_for_eval = DataLoader(dataset=dataset_train, pin_memory=True,
                                          batch_size=len(dataset_train), shuffle=False)

        testloader = DataLoader(dataset=dataset_test, pin_memory=True,
                                batch_size=len(dataset_test), shuffle=False)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        statistics = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'epochs': []}

        for epoch in range(epochs):
            # Fitting
            self.train()
            total_loss = 0

            for x, y in trainloader:
                x, y = x.to(device), y.to(device)

                y_ = self(x)

                loss = criterion(y_, y)
                total_loss += torch.sum(loss.detach()).item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluating
            with torch.no_grad():
                self.eval()

                # Train dataset
                for x, y in trainloader_for_eval:
                    # 1 iter
                    x = x.to(device)
                    y_ = self.predict(x)
                    train_acc = accuracy_score(y, y_)

                    y_ = self(x)
                    train_loss = torch.sum(criterion(y_.cpu(), y)).item()

                # Test dataset
                for x, y in testloader:
                    # 1 iter
                    x = x.to(device)
                    y_ = self.predict(x)
                    test_acc = accuracy_score(y, y_)

                    y_ = self(x)
                    test_loss = torch.sum(criterion(y_.cpu(), y)).item()

            print(f'Epoch: {epoch + 1}/{epochs}, train_loss: {train_loss:5.5f},'
                  f'train_acc: {train_acc:5.5f}, test_loss: {test_loss:5.5f}, test_acc: {test_acc}')

            # Add values to statistics dict
            statistics['train_loss'].append(train_loss)
            statistics['train_acc'].append(train_acc)
            statistics['test_loss'].append(test_loss)
            statistics['test_acc'].append(test_acc)
            statistics['epochs'].append(epoch)
        self.eval()
        return statistics

    # Prediction
    def predict(self, x):
        _, cls = torch.max(self.softmax(self(x).cpu().detach()), 1)
        return cls

    # Prediction with probability
    def predict_proba(self, x):
        rate, cls = torch.max(self.softmax(self(x).cpu().detach()), 1)
        return rate, cls


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    net = CNN()
    net = net.to(device)

    # Loading datasets
    data_train = torchvision.datasets.MNIST('data/train/', train=True, download=True,
                                            transform=transforms.ToTensor())
    data_test = torchvision.datasets.MNIST('data/test/', train=False, download=True,
                                           transform=transforms.ToTensor())
    statistics = net.fit(data_train, data_test, 10, lr=0.001, batch_size=100)

    f, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 5))

    loss_color = 'r'
    ax1.set_title('Train')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=loss_color)
    ax1.plot(statistics['epochs'], statistics['train_loss'], color=loss_color)
    ax1.tick_params(axis='y', labelcolor=loss_color)

    ax2 = ax1.twinx()
    acc_color = 'b'
    ax2.set_ylabel('Accuracy', color=acc_color)
    ax2.plot(statistics['epochs'], statistics['train_acc'], color=acc_color)
    ax2.tick_params(axis='y', labelcolor=acc_color)

    ax3.set_title('Test')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss', color=loss_color)
    ax3.plot(statistics['epochs'], statistics['test_loss'], color=loss_color)
    ax3.tick_params(axis='y', labelcolor=loss_color)

    ax4 = ax3.twinx()
    ax4.set_ylabel('Accuracy', color=acc_color)
    ax4.plot(statistics['epochs'], statistics['test_acc'], color=acc_color)
    ax4.tick_params(axis='y', labelcolor=acc_color)

    f.tight_layout()
    plt.show()
