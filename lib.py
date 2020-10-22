#!/usr/bin/env python

import os
import pdf2image
import PIL
import random
import re
import string
import subprocess
import sys
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

ATOMS = ["x", "y", "z", "a", "b", "c", "1", "2", "3", "4", "\\alpha", "\\beta"]

PREFIX_OPS = ["\\frac"]
INFIX_OPS = ["+", " \\cdot ", "^", "_", "+", "-"]
TOKENS = ATOMS + PREFIX_OPS + INFIX_OPS

NUM_LEAVES = 6
FORMULA_LENGTH = NUM_LEAVES * 2 - 1

LETTERS = list(string.ascii_letters)
WORD_LENGTH = 10

DATA_SIZE = 10000


class Formula:
    def __init__(self, size, allow_prefix=True):
        """
        Size is just the number of leaf nodes. There are also (size - 1) internal nodes.
        """
        assert size > 0
        self.size = size
        if size == 1:
            self.node_type = ATOM
            self.token = random.choice(ATOMS)
            self.left = None
            self.right = None
            return

        if allow_prefix and random.random() < 0.5:
            self.node_type = PREFIX
            self.token = random.choice(PREFIX_OPS)
        else:
            self.node_type = INFIX
            self.token = random.choice(INFIX_OPS)

        suballow = allow_prefix and self.node_type != PREFIX

        left_size = random.randrange(1, size)
        right_size = size - left_size
        self.left = Formula(left_size, allow_prefix=suballow)
        self.right = Formula(right_size, allow_prefix=suballow)

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

    def preorder(self):
        "Return a list of the tokens in this formula, in preorder."
        answer = [self.token]
        if self.left is not None:
            answer += self.left.preorder()
        if self.right is not None:
            answer += self.right.preorder()
        return answer

    def inorder(self):
        "Return a list of the tokens in this formula, in inorder."
        answer = []
        if self.left is not None:
            answer += self.left.inorder()
        answer += [self.token]
        if self.right is not None:
            answer += self.right.inorder()
        return answer

    def normalize(self):
        """
        Move infix operators to process them left-to-right.
        """
        if self.node_type == ATOM:
            return

        if self.node_type == PREFIX:
            self.left.normalize()
            self.right.normalize()
            return

        if self.left.node_type != INFIX:
            self.left.normalize()
            self.right.normalize()
            return

        # Now we have the awkward case - where both the root and the left side are infix.
        # Do a rotation, then renormalize.
        # We do this by cutting self.left out of the tree and gluing back together the three pieces.
        cut = self.left
        a = cut.left
        b = cut.right
        c = self.right
        self.token, cut.token = cut.token, self.token
        self.left = a
        self.right = cut
        cut.left = b
        cut.right = c

        # We might have to rotate more, so renormalize.
        self.normalize()


TEMPLATE = r"""
\documentclass[varwidth=true, border=1pt]{standalone}
\begin{document}
$%s$
\end{document}
"""


def generate_formula(n):
    random.seed(n)
    f = Formula(NUM_LEAVES)
    precheck = f.inorder()
    f.normalize()
    postcheck = f.inorder()
    assert precheck == postcheck
    return f


def generate_word(n):
    random.seed(n)
    letters = [random.choice(LETTERS) for _ in range(WORD_LENGTH)]
    return "".join(letters)


def generate_pdf(n):
    """
    Creates a {n}.pdf file in tmp.
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


def check_size(n):
    """
    Checks the size of generated pdfs up to n.
    """
    widths = []
    heights = []
    for i in range(n):
        generate_pdf(i)
        image = open_pdf(i)
        widths.append(image.width)
        heights.append(image.height)
        print(f"done with {i}")
    widths.sort()
    heights.sort()
    print("top widths:", widths[-10:])
    print("top heights:", heights[-10:])


# Parameters for image normalization.
# The "input" parameters are the rectangle we read from the pdf.
# The downscaling is how much we scale before putting it into the neural network.
INPUT_WIDTH = 272
INPUT_HEIGHT = 48
WIDTH = 136
HEIGHT = 24


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


def main():
    if "--generate" in sys.argv:
        print("generating...")
        check_size(DATA_SIZE)
        return

    if "--normalize" in sys.argv:
        print("normalizing...")
        for n in range(DATA_SIZE):
            normal(n)
        return

    print("set a flag to do a thing")


if __name__ == "__main__":
    main()
