#!/usr/bin/env python

import os
import pdf2image
import PIL
import random
import re
import subprocess
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

DIR = os.path.dirname(os.path.realpath(__file__))
TMP = os.path.join(DIR, "tmp")


# Types of formula nodes
ATOM = "atom"
PREFIX = "prefix"
INFIX = "infix"


class Formula:
    def __init__(self, size):
        assert size > 0
        self.size = size
        if size == 1:
            self.node_type = ATOM
            self.token = random.choice(
                ["x", "y", "z", "a", "b", "c", "1", "2", "3", "4", "\\alpha", "\\beta"]
            )
            self.left = None
            self.right = None
            return

        left_size = random.randrange(1, size)
        right_size = size - left_size
        self.left = Formula(left_size)
        self.right = Formula(right_size)

        if random.random() < 0.1:
            self.node_type = PREFIX
            self.token = "\\frac"
            return

        self.node_type = INFIX
        self.token = random.choice([" \\cdot ", "^", "_", "+", "-"])

    def __str__(self):
        if self.node_type == ATOM:
            return self.token
        elif self.node_type == PREFIX:
            return self.token + "{" + str(self.left) + "}{" + str(self.right) + "}"
        elif self.node_type == INFIX:
            return "{" + str(self.left) + "}" + self.token + "{" + str(self.right) + "}"
        else:
            raise ValueError("bad node type")

    def __contains__(self, token):
        if self.token == token:
            return True
        if self.left and token in self.left:
            return True
        if self.right and token in self.right:
            return True
        return False


TEMPLATE = r"""
\documentclass[varwidth=true, border=1pt]{standalone}
\begin{document}
$%s$
\end{document}
"""


def generate_formula(n):
    random.seed(n)
    return Formula(10)


def generate_pdf(n):
    """
    Creates a pdf which is experimentally of dimension at most 465 x 105.
    """
    pdf_filename = os.path.join(TMP, f"{n}.pdf")
    if os.path.isfile(pdf_filename):
        # It already has been generated.
        return

    tex = TEMPLATE % generate_formula(n)
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
    If you change the normalization algorithm, remove all the .normal files in the tmp directory.
    """
    filename = os.path.join(TMP, f"{name}.normal")
    try:
        return PIL.Image.open(filename)
    except FileNotFoundError:
        pass

    # Create a composite greyscale at the target size by pasting the pdf in.
    composite = PIL.Image.new("L", (INPUT_WIDTH, INPUT_HEIGHT), color=255)
    pdf = open_pdf(name)
    extra_width = composite.width - pdf.width
    extra_height = composite.height - pdf.height
    margin_left = extra_width // 2
    margin_top = extra_height // 2
    composite.paste(pdf, box=(margin_left, margin_top))

    inverted = PIL.ImageOps.invert(composite)
    normalized = inverted.resize((WIDTH, HEIGHT))
    normalized.save(filename, "PNG")
    return normalized


if __name__ == "__main__":
    pass
