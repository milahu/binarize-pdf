#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image
import doxapy


if 1:
    # debug
    print("DOXAPY FILE:", doxapy.__file__)
    print("DOXAPY TO_BINARY:", doxapy.to_binary)
    print(
        "DOXAPY SAUVOLA:",
        doxapy.Binarization.Algorithms.SAUVOLA,
    )
    print(
        "DOXAPY SAUVOLA TYPE:",
        type(doxapy.Binarization.Algorithms.SAUVOLA),
    )
    print(
        "DOXAPY ENUM CLASS:",
        doxapy.Binarization.Algorithms,
    )


DEFAULT_DPI = 300
DEFAULT_WINDOW = 75
DEFAULT_K = 0.2


def binarize(image: Image.Image) -> Image.Image:
    """Binarize a rendered PDF page with Doxa/Sauvola."""

    grayscale = np.asarray(
        image.convert("L"),
        dtype=np.uint8,
    )

    # This is deliberately written exactly like the
    # standalone test that was confirmed to work.
    # algorithm = doxapy.Binarization.Algorithms.SAUVOLA
    algorithm = doxapy.doxapy.Binarization.Algorithms.SAUVOLA

    # DEBUG: this MUST be Binarization.Algorithms.
    print(
        f"    Doxa algorithm: {algorithm!r}",
        flush=True,
    )
    print(
        f"    Doxa algorithm type: {type(algorithm)!r}",
        flush=True,
    )

    r'''
    result = doxapy.to_binary(
        algorithm,
        grayscale,
        {
            "window": DEFAULT_WINDOW,
            "k": DEFAULT_K,
        },
    )
    '''
    result = doxapy.to_binary(
        doxapy.doxapy.Binarization.Algorithms.SAUVOLA,
        grayscale,
        {
            "window": DEFAULT_WINDOW,
            "k": DEFAULT_K,
        },
    )

    return Image.fromarray(
        np.asarray(result, dtype=np.uint8),
        mode="L",
    )


def render_page(page: pymupdf.Page) -> Image.Image:
    """Render a complete PDF page."""

    pix = page.get_pixmap(
        dpi=DEFAULT_DPI,
        colorspace=pymupdf.csRGB,
        alpha=False,
        annots=True,
    )

    return Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        type=Path,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
    )

    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="use grayscale instead of Doxa bitonal output",
    )

    args = parser.parse_args()

    input_path = args.input

    if not input_path.is_file():
        raise SystemExit(
            f"error: file not found: {input_path}"
        )

    output_path = (
        args.output
        if args.output
        else input_path.with_suffix(".blackwhite.pdf")
    )

    with pymupdf.open(input_path) as document:

        print(
            f"Processing {len(document)} pages...",
            flush=True,
        )

        for number, page in enumerate(
            document,
            start=1,
        ):
            print(
                f"Page {number}/{len(document)}",
                flush=True,
            )

            # Render the COMPLETE visual page.
            rendered = render_page(page)

            if args.grayscale:
                processed = rendered.convert("L")
            else:
                processed = binarize(rendered)

            # PNG is lossless.
            buffer = io.BytesIO()
            processed.save(
                buffer,
                format="PNG",
                optimize=True,
            )

            # Overlay the processed page image.
            #
            # We deliberately leave the original PDF page contents
            # untouched. Thus the original OCR/searchable text layer
            # remains underneath the new visual page image.
            page.insert_image(
                page.rect,
                stream=buffer.getvalue(),
                keep_proportion=False,
                overlay=True,
            )

        temporary = output_path.with_name(
            output_path.name + ".tmp"
        )

        document.save(
            temporary,
            garbage=4,
            deflate=True,
            clean=True,
        )

        temporary.replace(output_path)

    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
