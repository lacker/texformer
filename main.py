#!/usr/bin/env python

import os
import pdf2image
import PIL
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
    return "{" + left + "}" + op + "{" + right + "}"


TEMPLATE = r"""
\documentclass[varwidth=true, border=1pt]{standalone}
\begin{document}
$%s$
\end{document}
"""


def generate_tex(n):
    random.seed(n)
    return TEMPLATE % random_formula(10)


def generate_pdf(n):
    """
    Creates a pdf which is experimentally of dimension at most 465 x 105.
    """
    tex = generate_tex(n)
    write(tex, n)


def write(tex, name):
    name = str(name)
    if not re.match("^[a-zA-Z_\\-0-9]+$", name):
        raise ValueError(f"bad file prefix: {name}")
    if "documentclass" not in tex[:100]:
        raise ValueError(f"bad tex: {tex[:100]}")

    tex_filename = os.path.join(TMP, f"{name}.tex")
    with open(tex_filename, "w") as f:
        f.write(tex)

    # pdflatex generates a bunch of files
    os.chdir(TMP)
    result = subprocess.run(
        ["pdflatex", "-halt-on-error", tex_filename], capture_output=True
    )

    pdf_filename = os.path.join(TMP, f"{name}.pdf")
    if result.returncode or not os.path.isfile(pdf_filename):
        print("stdout:", result.stdout.decode("utf-8"))
        print("stderr:", result.stderr.decode("utf-8"))
        raise IOError("pdflatex failed")


def open_pdf(name):
    """
    Opens a pdf as an image.
    """
    name = str(name)
    pdf_filename = os.path.join(TMP, f"{name}.pdf")
    pages = pdf2image.convert_from_path(pdf_filename)
    assert len(pages) == 1
    return pages[0]


def normal(name):
    """
    Normalize by centering in a 384 x 64 image.
    Returns a bilevel image. (Bilevel = each pixel is just white or black.)
    """
    target_width = 384
    target_height = 64

    # First we create a composite greyscale at the target size by pasting the pdf in.
    composite = PIL.Image.new("L", (target_width, target_height), color=255)
    pdf = open_pdf(name)
    extra_width = composite.width - pdf.width
    extra_height = composite.height - pdf.height
    margin_left = extra_width // 2
    margin_top = extra_height // 2
    composite.paste(pdf, box=(margin_left, margin_top))

    # Now convert to bilevel.
    threshold = 200
    fn = lambda x: 255 if x > threshold else 0
    result = composite.point(fn, mode="1")
    return result


def dimensions(name):
    """
    Returns a (width, height) tuple.
    """
    image = open_pdf(name)
    return (image.width, image.height)


def generate_pdfs(num):
    max_width, max_height = 0, 0
    for n in range(num):
        generate_pdf(n)
        updated = False
        width, height = dimensions(n)
        if width > max_width:
            max_width = width
            updated = True
        if height > max_height:
            max_height = height
            updated = True
        if updated:
            print("max (width, height) =", (max_width, max_height))


if __name__ == "__main__":
    base = 50
    num = 10
    for n in range(base, base + num):
        image = normal(n)
        image.show()
