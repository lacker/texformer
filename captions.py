#!/usr/bin/env python

import os
import time
import torch

from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample, set_seed
from torch.utils.data import Dataset

from lib import *

EMBEDDING_DIM = 6
HIDDEN_DIM = 6
RUN_NAME = "captions"
MODEL_PATH = os.path.join(TMP, f"{RUN_NAME}.pt")

# For the transformer
INPUT_SIZE = WIDTH * HEIGHT
OUTPUT_SIZE = WORD_LENGTH
VOCAB_SIZE = 64
assert VOCAB_SIZE >= len(LETTERS)
BLOCK_SIZE = INPUT_SIZE + OUTPUT_SIZE

DATA_SIZE = 100000
TRAIN_SIZE = int(0.9 * DATA_SIZE)
TEST_SIZE = DATA_SIZE - TRAIN_SIZE


def get_pixels(n):
    image = normal(n)
    raw_pixels = list(image.getdata())
    shade_scale = VOCAB_SIZE / 256
    pixels = [int(shade_scale * rp) for rp in raw_pixels]
    return pixels


class CaptionDataset(Dataset):
    """
    A dataset for recognizing a "word" in an image.
    """

    def __init__(self, size, offset=0):
        self.size = size
        self.offset = offset
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        n = self.offset + index

        # Get letters and pixels as lists of ints.
        word = generate_word(n)
        letters = [LETTERS.index(letter) for letter in word]
        pixels = get_pixels(n)

        # Predicting does not use the last element
        input_items = pixels + letters[:-1]

        # Mask loss on the output vector with -100's for the parts we aren't predicting
        output_items = [-100] * (len(pixels) - 1) + letters

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


def get_train_dataset():
    return CaptionDataset(TRAIN_SIZE)


def get_test_dataset():
    return CaptionDataset(TEST_SIZE, offset=TRAIN_SIZE)


def train():
    train_dataset = get_train_dataset()
    test_dataset = get_test_dataset()
    model = get_model()
    epochs = 20
    tokens_per_epoch = len(train_dataset) * BLOCK_SIZE
    conf = TrainerConfig(
        max_epochs=epochs,
        batch_size=128,
        learning_rate=6e-5,
        lr_decay=False,
        warmup_tokens=tokens_per_epoch,
        final_tokens=epochs * tokens_per_epoch,
        num_workers=4,
    )
    trainer = Trainer(model, train_dataset, test_dataset, conf)
    trainer.train()
    torch.save(model, MODEL_PATH)
    print(f"saved model to {MODEL_PATH}")
    return model


def run_one(model, n):
    "Returns the actual word produced."
    input_ints = get_pixels(n)
    input_tensor = torch.tensor([input_ints], dtype=torch.long).to(
        torch.cuda.current_device()
    )
    full_tensor = sample(model, input_tensor, OUTPUT_SIZE)
    full_ints = list(map(int, full_tensor[0]))
    output_ints = full_ints[-OUTPUT_SIZE:]
    return "".join(LETTERS[i] for i in output_ints)


def evaluate():
    model = get_model()
    correct = 0
    total = 0
    for n in range(TRAIN_SIZE, TRAIN_SIZE + 1000):
        correct_word = generate_word(n)
        predicted_word = run_one(model, n)
        total += 1
        if correct_word == predicted_word:
            correct += 1
        else:
            print(f"error on {n} :")
            print(f"    prediction:  {predicted_word}")
            print(f"    correct:     {correct_word}")
    print(f"got {correct}/{total} = {correct/total:.3f}% correct")


if __name__ == "__main__":
    train()
    evaluate()
