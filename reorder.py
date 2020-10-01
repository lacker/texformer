#!/usr/bin/env python

# Trying to train an LSTM to reorder trees.

import os
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
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

SIZE = 7


class Expression:
    def __init__(self, size):
        """
        Size includes internal nodes.
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
    A dataset for manipulating trees of tokens.
    We just generate new ones on the fly whenever we want expressions.
    """

    def __init__(self, size):
        self.size = size
        batch_size = 10
        self.loader = DataLoader(self, batch_size=batch_size, pin_memory=True)

    def batches(self):
        """
        Iterate over (batch, preorder, inorder) tuples, with orders mapped onto cuda structures.
        """
        for i, data in enumerate(self.loader):
            batch = i + 1
            preorder, inorder = data
            preorder, inorder = preorder.cuda(), labels.cuda()
            yield batch, preorder, inorder

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        expr = Expression(SIZE)
        preorder = [TOKENS.index(token) for token in expr.preorder_tokens()]
        inorder = [TOKENS.index(token) for token in expr.inorder_tokens()]
        preorder_tensor = torch.tensor(preorder, dtype=torch.long)
        inorder_tensor = torch.tensor(inorder, dtype=torch.long)
