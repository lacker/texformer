#!/usr/bin/env python

import os
import random
import re
import subprocess

DIR = os.path.dirname(os.path.realpath(__file__))
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


TEMPLATE = r"""
\documentclass[varwidth=true, border=1pt]{standalone}
\begin{document}
$%s$
\end{document}
"""


def write(tex: str, name: str):
    if not re.match("^[a-zA-Z_\\-0-9]+$", name):
        raise ValueError(f"bad file prefix: {name}")
    tex_filename = os.path.join(TMP, f"{name}.tex")
    with open(tex_filename, "w") as f:
        f.write(tex)

    # pdflatex generates a bunch of files
    os.chdir(TMP)
    subprocess.run(["pdflatex", "-halt-on-error", tex_filename])


if __name__ == "__main__":
    print(TMP)
    tex = TEMPLATE % random_formula(10)
    write(tex, "example")
