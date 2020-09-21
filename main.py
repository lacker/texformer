#!/usr/bin/env python

from datetime import timedelta
import os
import pdf2image
import PIL
import random
import re
import subprocess
import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

DIR = os.path.dirname(os.path.realpath(__file__))
TMP = os.path.join(DIR, "tmp")
MODEL_PATH = os.path.join(TMP, "model.pt")


def random_formula(size):
    assert size > 0
    if size == 1:
        return random.choice(
            ["x", "y", "z", "a", "b", "c", "1", "2", "3", "4", "\\alpha", "\\beta"]
        )
    left_size = random.randrange(1, size)
    right_size = size - left_size
    left = random_formula(left_size)
    right = random_formula(right_size)

    if random.random() < 0.1:
        return "\\frac{" + left + "}{" + right + "}"

    op = random.choice([" \\cdot ", "^", "_", "+", "-"])
    return "{" + left + "}" + op + "{" + right + "}"


TEMPLATE = r"""
\documentclass[varwidth=true, border=1pt]{standalone}
\begin{document}
$%s$
\end{document}
"""


def generate_tex(n):
    random.seed(n)
    return TEMPLATE % random_formula(10)


def generate_pdf(n):
    """
    Creates a pdf which is experimentally of dimension at most 465 x 105.
    """
    pdf_filename = os.path.join(TMP, f"{n}.pdf")
    if os.path.isfile(pdf_filename):
        # It already has been generated.
        return

    tex = generate_tex(n)
    write(tex, n)


def write(tex, name):
    name = str(name)
    if not re.match("^[a-zA-Z_\\-0-9]+$", name):
        raise ValueError(f"bad file prefix: {name}")
    if "documentclass" not in tex[:100]:
        raise ValueError(f"bad tex: {tex[:100]}")

    tex_filename = os.path.join(TMP, f"{name}.tex")
    with open(tex_filename, "w") as f:
        f.write(tex)

    # pdflatex generates a bunch of files
    os.chdir(TMP)
    result = subprocess.run(
        ["pdflatex", "-halt-on-error", tex_filename], capture_output=True
    )

    pdf_filename = os.path.join(TMP, f"{name}.pdf")
    if result.returncode or not os.path.isfile(pdf_filename):
        print("stdout:", result.stdout.decode("utf-8"))
        print("stderr:", result.stderr.decode("utf-8"))
        raise IOError("pdflatex failed")


def open_pdf(name):
    """
    Opens a pdf as an image.
    """
    name = str(name)
    pdf_filename = os.path.join(TMP, f"{name}.pdf")
    pages = pdf2image.convert_from_path(pdf_filename)
    assert len(pages) == 1
    return pages[0]


# Parameters for image normalization.
# The "input" parameters are the rectangle we read from the pdf.
# The downscaling is how much we scale before putting it into the neural network.
INPUT_WIDTH = 384
INPUT_HEIGHT = 64
WIDTH = 192
HEIGHT = 32


def normal(name):
    """
    Normalize.
    Colors are swapped so that zero = blank space, to make padding with zeros saner.
    Returns a greyscale image.
    """
    # Create a composite greyscale at the target size by pasting the pdf in.
    composite = PIL.Image.new("L", (INPUT_WIDTH, INPUT_HEIGHT), color=255)
    pdf = open_pdf(name)
    extra_width = composite.width - pdf.width
    extra_height = composite.height - pdf.height
    margin_left = extra_width // 2
    margin_top = extra_height // 2
    composite.paste(pdf, box=(margin_left, margin_top))

    inverted = PIL.ImageOps.invert(composite)
    return inverted.resize((WIDTH, HEIGHT))


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

        trainsize = int(0.9 * size)
        train_indices = range(trainsize)
        self.trainset = Subset(self, train_indices)
        self.trainloader = DataLoader(self.trainset, batch_size=4, shuffle=True)

        testsize = size - trainsize
        test_indices = range(trainsize, size)
        self.testset = Subset(self, test_indices)
        self.testloader = DataLoader(self.testset, batch_size=4, shuffle=True)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        tex = generate_tex(index)
        label = 1 if "alpha" in tex else 0
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
        self.data = Alphaset()
        assert torch.cuda.is_available()

        # Load net from disk if possible
        try:
            self.model = torch.load(MODEL_PATH)
        except FileNotFoundError:
            self.model = Net().cuda()

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.epochs = 0

    def epoch(self):
        start = time.time()
        self.epochs += 1

        running_loss = 0.0
        for i, data in enumerate(self.data.trainloader):
            batch = i + 1
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            group_size = 100
            if batch % group_size == 0:
                current_loss = running_loss / group_size
                print(f"epoch {self.epochs}, batch {batch}: loss = {current_loss:.3f}")
                running_loss = 0.0

        elapsed = time.time() - start
        print(f"epoch took {timedelta(seconds=elapsed)}")
        torch.save(self.model, MODEL_PATH)


if __name__ == "__main__":
    t = Trainer()
    t.epoch()
