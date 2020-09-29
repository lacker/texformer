#!/usr/bin/env python
from datetime import timedelta
import os
import time

from lib import *

TOKENS = ATOMS + [PREFIX_OP] + INFIX_OPS
LEAF_CHANNELS = 20
MID_CHANNELS = 20
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
RUN_NAME = f"captions-{LEAF_CHANNELS}-{MID_CHANNELS}-{EMBEDDING_DIM}-{HIDDEN_DIM}"
MODEL_PATH = os.path.join(TMP, f"{RUN_NAME}.pt")


class CaptionSet(Dataset):
    """
    A dataset for labeling images with string "captions" of the tex tokens that generate them.
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
        "Iterate over (batch, images, tokens) tuples."
        for i, data in enumerate(loader):
            batch = i + 1
            inputs, tokens = data
            inputs, tokens = inputs.cuda(), tokens.cuda()
            yield batch, inputs, tokens

    def train_batches(self):
        return self.iter_batches(self.trainloader)

    def test_batches(self):
        return self.iter_batches(self.testloader)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        formula = generate_formula(index)
        token_list = formula.preorder()
        token_indices = [TOKENS.index(token) for token in token_list]
        token_tensor = torch.tensor(token_indices, dtype=torch.long)

        image = normal(index)
        image_tensor = self.to_tensor(image)
        return image_tensor, token_tensor
