#!/usr/bin/env python

# Trying to train an LSTM to reorder trees.

import os
import random
import torch
from mingpt.utils import set_seed
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

DIR = os.path.dirname(os.path.realpath(__file__))
TMP = os.path.join(DIR, "tmp")

# Types of expressions
OP = "op"
ATOM = "atom"

# The different tokens
OPS = ["+"]
ATOMS = ["1", "2", "3"]
TOKENS = OPS + ATOMS

# How long our input and output formulas will be
EXPRESSION_SIZE = 7


class Expression:
    def __init__(self, size):
        """
        Size includes internal nodes.
        For nondeterminism, seed before constructing.
        """
        if size % 2 == 0:
            raise ValueError(f"expressions must have odd size.")

        self.size = size
        if size == 1:
            self.node_type = ATOM
            self.token = random.choice(ATOMS)
            self.left = None
            self.right = None
            return

        left_size = random.randrange(1, size, 2)
        right_size = size - 1 - left_size

        self.node_type = OP
        self.token = random.choice(OPS)
        self.left = Expression(left_size)
        self.right = Expression(right_size)

    def preorder_tokens(self):
        "A preorder traversal of the tokens in this expression."
        answer = [self.token]
        if self.left is not None:
            answer += self.left.preorder_tokens()
        if self.right is not None:
            answer += self.right.preorder_tokens()

    def inorder_tokens(self):
        "An inorder traversal of the tokens in this expression."
        answer = []
        if self.left is not None:
            answer += self.left.preorder_tokens()
        answer.append(self.token)
        if self.right is not None:
            answer += self.right.preorder_tokens()


class ExpressionSet(Dataset):
    """
    A dataset for preorder->inorder rewrite problems.
    """

    def __init__(self, size, split):
        """
        size is how many are in this specific dataset.
        split can be "train" or "test".
        """
        self.expressions = []
        for i in range(size):
            random.seed(f"{split}{i}")
            expr = Expression(EXPRESSION_SIZE)
            self.expressions.append(expr)

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, index):
        expr = self.expressions[index]
        preorder = [TOKENS.index(token) for token in expr.preorder_tokens()]
        inorder = [TOKENS.index(token) for token in expr.inorder_tokens()]
        full_vector = preorder + inorder

        # TODO: masking etc
