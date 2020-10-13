#!/usr/bin/env python
from datetime import timedelta
import os
import time

from lib import *

EMBEDDING_DIM = 6
HIDDEN_DIM = 6
RUN_NAME = "captions"
MODEL_PATH = os.path.join(TMP, f"{RUN_NAME}.pt")

# For the transformer
INPUT_SIZE = WIDTH * HEIGHT
OUTPUT_SIZE = WORD_LENGTH
VOCAB_SIZE = 256
BLOCK_SIZE = INPUT_SIZE + OUTPUT_SIZE


class CaptionDataset(Dataset):
    """
    A dataset for recognizing a "word" in an image.
    """

    def __init__(self, size=10000):
        self.size = size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Get letters and pixels as lists of ints.
        word = generate_word(index)
        letters = [LETTERS.index(letter) for letter in word]
        image = normal(index)
        pixels = list(image.getdata())

        # Predicting does not use the last element
        input_items = letters + pixels[:-1]

        # Mask loss on the output vector with -100's for the parts we aren't predicting
        output_items = [-100] * (len(letters) - 1) + pixels

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
