#!/usr/bin/env python
from datetime import timedelta
import os
import time

from lib import *

TOKENS = ATOMS + [PREFIX_OP] + INFIX_OPS
LEAF_CHANNELS = 40
MID_CHANNELS = 40
RUN_NAME = f"tokens-{LEAF_CHANNELS}-{MID_CHANNELS}"
MODEL_PATH = os.path.join(TMP, f"{RUN_NAME}.pt")


class MultiLabelSet(Dataset):
    """
    A dataset for simultaneously labeling images with what tokens they contain.
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
        label_list = []
        for token in TOKENS:
            if token in formula:
                label_list.append(1)
            else:
                label_list.append(0)
        label_tensor = torch.FloatTensor(label_list)
        image = normal(index)
        image_tensor = self.to_tensor(image)
        return image_tensor, label_tensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Each convolution layer shrinks each dimension 2x
        out_width = WIDTH // 4
        out_height = HEIGHT // 4

        self.cnn_layers = nn.Sequential(
            # The first convolution layer
            nn.Conv2d(1, LEAF_CHANNELS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(LEAF_CHANNELS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # The second convolution layer
            nn.Conv2d(
                LEAF_CHANNELS, MID_CHANNELS, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(MID_CHANNELS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layer = nn.Linear(
            MID_CHANNELS * out_width * out_height, len(TOKENS)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        return x


def flatten(tensor):
    return [x.item() for x in torch.flatten(tensor)]


class Trainer:
    def __init__(self):
        assert torch.cuda.is_available()
        self.data = MultiLabelSet()
        self.writer = SummaryWriter(log_dir=f"./runs/{RUN_NAME}/")

        # Load net from disk if possible
        try:
            self.model = torch.load(MODEL_PATH)
            print(f"resuming from existing model at {MODEL_PATH}")
        except FileNotFoundError:
            self.model = Net().cuda()
            self.model.epochs = 0

        self.criterion = nn.MultiLabelSoftMarginLoss().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)

    def epoch(self):
        start = time.time()
        self.model.epochs += 1

        total_loss = 0.0
        running_loss = 0.0
        last_batch = 1
        for batch, inputs, labels in self.data.train_batches():
            last_batch = batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()
            group_size = 100
            if batch % group_size == 0:
                current_loss = running_loss / group_size
                print(
                    f"epoch {self.model.epochs}, batch {batch}: loss = {current_loss:.3f}"
                )
                running_loss = 0.0
        loss = total_loss / last_batch
        elapsed = time.time() - start
        print(f"epoch took {timedelta(seconds=elapsed)}. loss = {loss:.3f}")
        self.writer.add_scalar("loss", loss, self.model.epochs)
        self.evaluate(log=True)
        torch.save(self.model, MODEL_PATH)

    def evaluate(self, log=False):
        total = 0
        correct = 0
        with torch.no_grad():
            for batch, inputs, labels in self.data.test_batches():
                outputs = flatten(self.model(inputs))
                flat_labels = flatten(labels)
                batch_score = 0
                for label, output in zip(flat_labels, outputs):
                    output_int = 1 if output > 0.5 else 0
                    if output_int == label:
                        batch_score += 1
                total += len(flat_labels)
                correct += batch_score

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
