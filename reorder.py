#!/usr/bin/env python

# Trying to train an LSTM to reorder trees.

import os
import random
import torch

from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import set_seed
from torch.utils.data import Dataset

DIR = os.path.dirname(os.path.realpath(__file__))
TMP = os.path.join(DIR, "tmp")
MODEL_PATH = os.path.join(TMP, "reorder.pt")

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
        return answer

    def inorder_tokens(self):
        "An inorder traversal of the tokens in this expression."
        answer = []
        if self.left is not None:
            answer += self.left.inorder_tokens()
        answer.append(self.token)
        if self.right is not None:
            answer += self.right.inorder_tokens()
        return answer


class ReorderDataset(Dataset):
    """
    A dataset for preorder->inorder rewrite problems.
    """

    def __init__(self, split, size):
        """
        size is how many are in this specific dataset.
        split can be "train" or "test".
        """
        self.vocab_size = len(TOKENS)
        self.block_size = 2 * EXPRESSION_SIZE
        self.expressions = []
        for i in range(size):
            random.seed(f"{split}-{i}")
            expr = Expression(EXPRESSION_SIZE)
            self.expressions.append(expr)
        print(f"{split} dataset of size {size} created")

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, index):
        expr = self.expressions[index]
        preorder = [TOKENS.index(token) for token in expr.preorder_tokens()]
        inorder = [TOKENS.index(token) for token in expr.inorder_tokens()]

        # Predicting does not use the last element
        input_items = preorder + inorder[:-1]

        # Mask loss on the output vector with -100's, for the parts we aren't trying to predict.
        output_items = [-1] * (len(preorder) - 1) + inorder

        # Convert to tensor
        input_tensor = torch.tensor(input_items, dtype=torch.long)
        output_tensor = torch.tensor(output_items, dtype=torch.long)
        return input_tensor, output_tensor


def get_model(dataset, rebuild=False):
    set_seed(42)
    if not rebuild:
        try:
            model = torch.load(MODEL_PATH)
            print(f"resuming from existing model at {MODEL_PATH}")
            return model
        except FileNotFoundError:
            pass
    print("constructing new model")
    conf = GPTConfig(
        dataset.vocab_size, dataset.block_size, n_layer=2, n_head=4, n_embd=128
    )
    model = GPT(conf)
    return model


def train():
    train_dataset = ReorderDataset("train", 10000)
    test_dataset = ReorderDataset("test", 1000)
    model = get_model(train_dataset, rebuild=True)
    epochs = 50
    conf = TrainerConfig(
        max_epochs=epochs,
        batch_size=512,
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=1024,
        final_tokens=epochs * len(train_dataset) * len(TOKENS),
        num_workers=4,
    )
    trainer = Trainer(model, train_dataset, test_dataset, conf)
    trainer.train()
    torch.save(self.model, MODEL_PATH)


if __name__ == "__main__":
    train()
