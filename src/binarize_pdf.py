#!/usr/bin/env python3

"""
Binarize PDF pages using Doxa's adaptive binarization while preserving
the original PDF's text layer.

For each page:

1. Render the complete PDF page to an image at the requested DPI.
   This captures text, raster images, vector graphics, transparency, etc.
2. Convert the rendered image to grayscale.
3. Either:
   - leave it as grayscale, or
   - apply Doxa's adaptive Sauvola binarization.
4. Put the resulting image over the original page.
5. Leave the original PDF content untouched underneath, thereby preserving
   its existing searchable / selectable text layer.

The output is therefore visually represented by the processed page image,
while the original PDF text remains available to PDF text extraction/OCR
software.

Dependencies:

    python -m pip install pymupdf pillow doxapy

DoxaPy currently requires Python 3.12+.

Usage:

    python binarize_pdf.py input.pdf

    python binarize_pdf.py input.pdf --mode grayscale

    python binarize_pdf.py input.pdf --mode bitonal

    python binarize_pdf.py input.pdf -o output.pdf

The default output name is:

    input.pdf -> input.blackwhite.pdf
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

import doxapy


# Rendering DPI. 300 is a reasonable default for document/OCR work.
DEFAULT_DPI = 300

# Sauvola parameters.
#
# Doxa's example uses window=75 and k=0.2. These are deliberately kept
# here as constants so they are easy to change for a particular document
# collection.
DEFAULT_WINDOW = 75
DEFAULT_K = 0.2


def default_output_path(input_path: Path) -> Path:
    """Return the default output path for an input PDF."""
    if input_path.suffix.lower() == ".pdf":
        return input_path.with_suffix(".blackwhite.pdf")

    # The CLI says the input should be a PDF, but this makes the function
    # behave sensibly if called independently.
    return input_path.with_name(input_path.name + ".blackwhite.pdf")


def render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
    """
    Render an entire PDF page to an RGB PIL image.

    Rendering the page rather than extracting individual PDF images is
    intentional: the result represents the visual appearance of the
    complete PDF page, including text, vector graphics, images,
    transparency, etc.
    """
    pixmap = page.get_pixmap(
        dpi=dpi,
        colorspace=pymupdf.csRGB,
        alpha=False,
        annots=True,
    )

    # Pixmap samples are RGB because csRGB was requested.
    image = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    return image


def binarize_with_doxa(
    image: Image.Image,
    window: int = DEFAULT_WINDOW,
    k: float = DEFAULT_K,
) -> Image.Image:
    grayscale = np.asarray(
        image.convert("L"),
        dtype=np.uint8,
    )

    binary = doxapy.to_binary(
        doxapy.Binarization.Algorithms.SAUVOLA,
        grayscale,
        {
            "window": window,
            "k": k,
        },
    )

    return Image.fromarray(
        np.asarray(binary, dtype=np.uint8),
        mode="L",
    )


def process_pdf(
    input_path: Path,
    output_path: Path,
    mode: str,
    dpi: int,
    window: int,
    k: float,
) -> None:
    """Process all pages of input_path and save output_path."""

    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "The output file must not be the same as the input file."
        )

    with pymupdf.open(input_path) as document:
        page_count = len(document)

        print(
            f"Processing {input_path} "
            f"({page_count} page{'s' if page_count != 1 else ''})..."
        )

        for page_number, page in enumerate(document, start=1):
            print(
                f"  Page {page_number}/{page_count}",
                file=sys.stderr,
            )

            # Render the complete visual appearance of the page.
            rendered = render_page(page, dpi)

            if mode == "grayscale":
                processed = rendered.convert("L")
            elif mode == "bitonal":
                processed = binarize_with_doxa(
                    rendered,
                    window=window,
                    k=k,
                )
            else:
                # argparse prevents this, but keep the function defensive.
                raise ValueError(f"Unknown mode: {mode}")

            image_stream = image_to_pdf_stream(processed)

            # Cover the original visual content with the processed
            # page image.
            #
            # IMPORTANT:
            #
            # We deliberately do NOT delete or replace the original PDF
            # page contents. The new image is opaque and visually replaces
            # them, while the original text objects remain underneath.
            #
            # Thus PDF text extraction/search continues to see the
            # original text layer.
            page.insert_image(
                page.rect,
                stream=image_stream,
                keep_proportion=False,
                overlay=True,
            )

        # garbage=4 removes unused objects created during processing,
        # deflate=True compresses streams where possible.
        #
        # We use a temporary output file rather than writing directly over
        # an existing file.
        temp_output = output_path.with_name(
            output_path.name + ".tmp"
        )

        try:
            document.save(
                temp_output,
                garbage=4,
                deflate=True,
                clean=True,
            )

            # Replace any existing destination only after a successful save.
            temp_output.replace(output_path)

        except Exception:
            try:
                temp_output.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    print(f"Written: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render each PDF page, apply Doxa adaptive binarization, "
            "and overlay the result while preserving the original "
            "PDF text layer."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="input PDF file",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output PDF path (default: INPUT.blackwhite.pdf)",
    )

    parser.add_argument(
        "--mode",
        choices=("bitonal", "grayscale"),
        default="bitonal",
        help=(
            "output image mode: bitonal uses Doxa adaptive Sauvola "
            "binarization; grayscale only converts the rendered page "
            "to grayscale (default: bitonal)"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"rendering resolution in DPI (default: {DEFAULT_DPI})",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=(
            "Doxa Sauvola local window size in pixels "
            f"(default: {DEFAULT_WINDOW})"
        ),
    )

    parser.add_argument(
        "--k",
        type=float,
        default=DEFAULT_K,
        help=(
            "Doxa Sauvola k parameter "
            f"(default: {DEFAULT_K})"
        ),
    )

    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error("--dpi must be greater than zero")

    if args.window <= 0:
        parser.error("--window must be greater than zero")

    if args.window % 2 == 0:
        parser.error("--window must be an odd number")

    if not 0 < args.k < 1:
        parser.error("--k must be greater than 0 and less than 1")

    return args


def main() -> int:
    args = parse_args()

    input_path: Path = args.input

    if not input_path.is_file():
        print(
            f"error: input file does not exist: {input_path}",
            file=sys.stderr,
        )
        return 2

    if input_path.suffix.lower() != ".pdf":
        print(
            f"error: input file does not have a .pdf extension: "
            f"{input_path}",
            file=sys.stderr,
        )
        return 2

    output_path = args.output or default_output_path(input_path)

    try:
        process_pdf(
            input_path=input_path,
            output_path=output_path,
            mode=args.mode,
            dpi=args.dpi,
            window=args.window,
            k=args.k,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
