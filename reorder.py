#!/usr/bin/env python

# Trying to train an LSTM to reorder trees.

import os
import random
import time
import torch

from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample, set_seed
from torch.utils.data import DataLoader, Dataset

DIR = os.path.dirname(os.path.realpath(__file__))
TMP = os.path.join(DIR, "tmp")
MODEL_PATH = os.path.join(TMP, "reorder.pt")

# Types of expressions
OP = "op"
ATOM = "atom"

# The different tokens
OPS = ["+", "*"]
ATOMS = ["1", "2", "3", "4"]
TOKENS = OPS + ATOMS

# How long our input and output formulas will be
EXPRESSION_SIZE = 15

# Block size is the size of input plus output
VOCAB_SIZE = len(TOKENS)
BLOCK_SIZE = 2 * EXPRESSION_SIZE


def cast_token(x):
    if type(x) == str:
        return x
    if type(x) == int:
        return TOKENS[x]
    raise ValueError(f"cannot cast_token on {x}")


def cast_int(token):
    if type(token) == int:
        return token
    if type(token) == str:
        return TOKENS.index(token)
    raise ValueError(f"cannot cast_int on {token}")


class Expression:
    def __init__(self, token, left=None, right=None):
        self.size = 1
        if left is not None:
            assert right is not None
            self.node_type = OP
            self.size += left.size
            self.size += right.size
        else:
            self.node_type = ATOM
        self.left = left
        self.right = right
        self.token = token

    @staticmethod
    def random(size):
        """
        Size includes internal nodes.
        For nondeterminism, seed before constructing.
        """
        if size % 2 == 0:
            raise ValueError(f"expressions must have odd size.")
        if size == 1:
            return Expression(random.choice(ATOMS))

        left_size = random.randrange(1, size, 2)
        right_size = size - 1 - left_size

        token = random.choice(OPS)
        left = Expression.random(left_size)
        right = Expression.random(right_size)
        return Expression(token, left=left, right=right)

    @staticmethod
    def parse_preorder(tokens):
        """
        Returns a tuple of (expression, rest).
        """
        token, rest = cast_token(tokens[0]), tokens[1:]
        if token in ATOMS:
            return Expression(token), rest
        left, rest = Expression.parse_preorder(rest)
        right, rest = Expression.parse_preorder(rest)
        return Expression(token, left=left, right=right), rest

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

    def postorder_tokens(self):
        "A postorder traversal of the tokens in this expression."
        answer = []
        if self.left is not None:
            answer += self.left.inorder_tokens()
        if self.right is not None:
            answer += self.right.inorder_tokens()
        answer.append(self.token)
        return answer


class ReorderDataset(Dataset):
    """
    A dataset for preorder->postorder rewrite problems.
    """

    def __init__(self, split, size, deterministic=False):
        """
        size is how many are in this specific dataset.
        split can be "train" or "test".
        """
        self.expressions = []
        for i in range(size):
            if deterministic:
                random.seed(f"{split}-{i}")
            expr = Expression.random(EXPRESSION_SIZE)
            self.expressions.append(expr)
        print(f"{split} dataset of size {size} created")

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, index):
        expr = self.expressions[index]
        preorder = [TOKENS.index(token) for token in expr.preorder_tokens()]
        postorder = [TOKENS.index(token) for token in expr.postorder_tokens()]

        # Predicting does not use the last element
        input_items = preorder + postorder[:-1]

        # Mask loss on the output vector with -100's, for the parts we aren't trying to predict.
        output_items = [-100] * (len(preorder) - 1) + postorder

        # Convert to tensor
        input_tensor = torch.tensor(input_items, dtype=torch.long)
        output_tensor = torch.tensor(output_items, dtype=torch.long)
        return input_tensor, output_tensor


def get_model(rebuild=False):
    set_seed(42)
    if not rebuild:
        try:
            model = torch.load(MODEL_PATH)
            print(f"resuming from existing model at {MODEL_PATH}")
            return model
        except FileNotFoundError:
            pass
    print("constructing new model")
    conf = GPTConfig(VOCAB_SIZE, BLOCK_SIZE, n_layer=2, n_head=4, n_embd=128)
    model = GPT(conf)
    return model


def train():
    train_dataset = ReorderDataset("train", 10000)
    test_dataset = ReorderDataset("test", 1000)
    model = get_model()
    epochs = 100
    # NOTE: I think final_tokens is set wrong
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
    torch.save(model, MODEL_PATH)
    print(f"saved model to {MODEL_PATH}")
    return model


def run_one(model, input_tokens):
    input_ints = list(map(cast_int, input_tokens))
    input_tensor = torch.tensor([input_ints], dtype=torch.long).to(
        torch.cuda.current_device()
    )
    full_tensor = sample(model, input_tensor, EXPRESSION_SIZE)
    full_ints = list(map(int, full_tensor[0]))
    output_ints = full_ints[-EXPRESSION_SIZE:]
    return list(map(cast_token, output_ints))


def evaluate(model):
    correct = 0
    total = 0
    for _ in range(10000):
        expr = Expression.random(EXPRESSION_SIZE)
        input_str = "".join(expr.preorder_tokens())
        actual_output = "".join(run_one(model, expr.preorder_tokens()))
        expected_output = "".join(expr.postorder_tokens())
        total += 1
        if actual_output == expected_output:
            correct += 1
        else:
            print(f"error on {input_str} :")
            print(f"    actual:    {actual_output}")
            print(f"    expected:  {expected_output}")
    print(f"got {correct}/{total} = {correct/total:.3f}% correct")


if __name__ == "__main__":
    model = train()
    evaluate(model)
