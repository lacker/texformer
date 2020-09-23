#!/usr/bin/env python
from datetime import timedelta
import os
import time

from lib import *

MODEL_PATH = os.path.join(TMP, "alpha.pt")


class Alphaset(Dataset):
    """
    A dataset for classifying whether the image contains an alpha.
    """

    def __init__(self, size=100000, populate=False):
        self.size = size
        if populate:
            for n in range(size):
                generate_pdf(n)
        self.to_tensor = transforms.ToTensor()

        batch_size = 18
        train_size = int(0.9 * size)
        train_indices = range(train_size)
        self.trainset = Subset(self, train_indices)
        self.trainloader = DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        test_size = size - train_size
        test_indices = range(train_size, size)
        self.testset = Subset(self, test_indices)
        self.testloader = DataLoader(
            self.testset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

    def iter_batches(self, loader):
        "Iterate over (batch, images, labels) tuples."
        for i, data in enumerate(loader):
            batch = i + 1
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            yield batch, inputs, labels

    def train_batches(self):
        return self.iter_batches(self.trainloader)

    def test_batches(self):
        return self.iter_batches(self.testloader)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        formula = generate_formula(index)
        label = 1 if "\\alpha" in formula else 0
        image = normal(index)
        tensor = self.to_tensor(image)
        return tensor, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Each convolution layer shrinks each dimension 2x
        out_width = WIDTH // 4
        out_height = HEIGHT // 4

        self.cnn_layers = nn.Sequential(
            # The first convolution layer
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # The second convolution layer
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layer = nn.Linear(4 * out_width * out_height, 2)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        return x


class Trainer:
    def __init__(self):
        assert torch.cuda.is_available()
        self.data = Alphaset()
        self.writer = SummaryWriter(log_dir="./runs/alpha/")

        # Load net from disk if possible
        try:
            self.model = torch.load(MODEL_PATH)
            print("resuming from existing model")
        except FileNotFoundError:
            self.model = Net().cuda()
            self.model.epochs = 0

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)

    def epoch(self):
        start = time.time()
        self.model.epochs += 1

        running_loss = 0.0
        for batch, inputs, labels in self.data.train_batches():
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            group_size = 100
            if batch % group_size == 0:
                current_loss = running_loss / group_size
                print(
                    f"epoch {self.model.epochs}, batch {batch}: loss = {current_loss:.3f}"
                )
                running_loss = 0.0

        elapsed = time.time() - start
        print(f"epoch took {timedelta(seconds=elapsed)}")
        self.evaluate(log=True)
        torch.save(self.model, MODEL_PATH)

    def evaluate(self, log=False):
        total = 0
        correct = 0
        with torch.no_grad():
            for batch, inputs, labels in self.data.test_batches():
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                group_size = 100
                if batch % group_size == 0:
                    print(
                        f"eval batch {batch}: accuracy = {correct}/{total} = {(correct/total):.3f}"
                    )

        accuracy = correct / total
        print(f"current accuracy = {correct}/{total} = {accuracy:.3f}")
        if log:
            self.writer.add_scalar("accuracy", accuracy, self.model.epochs)

        return accuracy

    def find_mistakes(self, n=10, show=False):
        """
        Return indices that are mistakes. Also pops up windows showing them.
        """
        answer = []
        for i in range(self.data.size - 1, -1, -1):
            image, label = self.data[i]
            image = image.unsqueeze(0).cuda()
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            predicted = predicted.item()
            correct = predicted == label

            if not correct:
                print(f"image {i} label = {label}, predicted = {predicted}")
                if show:
                    open_pdf(i).show()

                answer.append(i)
                if len(answer) >= n:
                    break
        return answer
