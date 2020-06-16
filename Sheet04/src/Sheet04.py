import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '  '
import matplotlib.pyplot as plt


class ShallowModel(nn.Module):
    def __init__(self):
        super(ShallowModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(2880, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class DeeperModel(nn.Module):
    def __init__(self, batchNorm=False):
        super(DeeperModel, self).__init__()
        self.batch_norm = batchNorm
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(40)
        self.conv4 = nn.Conv2d(40, 80, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(80)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1280, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        # layer1
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        # layer2
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.maxpool(self.relu(x))
        # layer3
        x = self.conv3(x)
        if self.batch_norm:
            x = self.bn3(x)
        x = self.relu(x)
        # layer4
        x = self.conv4(x)
        if self.batch_norm:
            x = self.bn4(x)
        x = self.maxpool(self.relu(x))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    ## Get data
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    ## Training Parameters
    nepochs = 10
    batch_size = 64
    learning_rate = 0.0001

    ### Create Model
    model1 = ShallowModel()
    model2 = DeeperModel(batchNorm=False)
    model3 = DeeperModel(batchNorm=True)

    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    model_accuracies = [[] for x in range(3)]
    model_losses = [[] for x in range(3)]
    models = ['ShallowModel', 'DeeperModel', 'DeeperModelwithBatchNorm']
    for m, model in enumerate([model1, model2, model3]):

        ### Define Opitmizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        ## Create Dataloaders
        train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=True)
        model.to(device)
        print(models[m])
        for epoch in range(nepochs):

            #### Train
            for i, (images, labels) in enumerate(train_loader):
                # Clear gradients w.r.t. parameters
                if torch.cuda.is_available():
                    images = images.to(device)
                    labels = labels.to(device)
                optimizer.zero_grad()
                # Forward pass to get output/logits
                outputs = model(images)
                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs, labels)
                # Getting gradients w.r.t. parameters
                loss.backward()
                # Updating parameters
                optimizer.step()

            #### Test
            correct = 0
            total = 0
            test_loss = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Forward pass only to get logits/output
                if torch.cuda.is_available():
                    images = images.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, labels)
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                # Total number of labels
                total += labels.size(0)
                # Total correct predictions
                correct += (predicted.cpu() == labels.cpu()).sum().float()
            accuracy = 100. * correct / total

            # Print Loss
            print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, test_loss, accuracy))
            model_accuracies[m].append(accuracy.item())
            model_losses[m].append(test_loss.item())

        ## Save Model for sharing
        torch.save(model.state_dict(), './model' + models[m])

    for accuracy in model_accuracies:
        plt.plot(accuracy)
    plt.legend(['ShallowModel', 'DeeperModel', 'DeeperModelwithBatchNorm'])
    plt.title("Testing Accuracy v No of Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

    for loss in model_losses:
        plt.plot(loss)
    plt.legend(['ShallowModel', 'DeeperModel', 'DeeperModelwithBatchNorm'])
    plt.title("Testing Loss v No of Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    main()