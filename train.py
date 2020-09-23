#!/usr/bin/env python
from multilabel import *

if __name__ == "__main__":
    t = Trainer()
    for _ in range(100):
        t.epoch()
