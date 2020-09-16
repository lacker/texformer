#!/usr/bin/env python

import os
import random

DIR = os.path.dirname(__file__)
TMP = os.path.join(DIR, "tmp")


def random_formula(size):
    assert size > 0
    if size == 1:
        return random.choice(
            ["x", "y", "z", "a", "b", "c", "1", "2", "3", "4", "\\alpha", "\\beta"]
        )
    left_size = random.randrange(1, size)
    right_size = size - left_size
    left = random_formula(left_size)
    right = random_formula(right_size)

    if random.random() < 0.1:
        return "\\frac{" + left + "}{" + right + "}"

    op = random.choice([" \\cdot ", "^", "_", "+", "-"])
    return f"{left}{op}{right}"


TEMPLATE = """
\documentclass[varwidth=true, border=1pt]{standalone}
%s
\end{document}
"""


def main():
    with open(os.path.join(TMP, "example.tex"), "w") as f:
        f.write(TEMPLATE % random_formula(10))


if __name__ == "__main__":
    main()
