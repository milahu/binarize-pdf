#!/usr/bin/env python3

"""
Render each page of a PDF, binarize the rendered page with Doxa/Sauvola,
and overlay the resulting image while preserving the original PDF
content and text layer.

Usage:

    python binarize_pdf.py input.pdf

    python binarize_pdf.py input.pdf --mode grayscale

    python binarize_pdf.py input.pdf --mode bitonal

    python binarize_pdf.py input.pdf -o output.pdf

The default output filename is:

    input.pdf -> input.blackwhite.pdf

Dependencies:

    python -m pip install pymupdf pillow numpy doxapy
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


DEFAULT_DPI = 300
DEFAULT_WINDOW = 75
DEFAULT_K = 0.2


def default_output_path(input_path: Path) -> Path:
    """Return INPUT.blackwhite.pdf."""
    return input_path.with_suffix(".blackwhite.pdf")


def render_page(
    page: pymupdf.Page,
    dpi: int,
) -> Image.Image:
    """
    Render the complete visual appearance of a PDF page.

    This includes PDF text, raster images, vector graphics, annotations,
    transparency, etc.
    """
    pix = page.get_pixmap(
        dpi=dpi,
        colorspace=pymupdf.csRGB,
        alpha=False,
        annots=True,
    )

    return Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples,
    )


def binarize_with_doxa(
    image: Image.Image,
    window: int,
    k: float,
) -> Image.Image:
    """
    Apply Doxa's adaptive Sauvola binarization.

    This is the exact API used by the current DoxaPy documentation and
    verified against the user's installed DoxaPy version.
    """
    grayscale = np.asarray(
        image.convert("L"),
        dtype=np.uint8,
    )

    if grayscale.ndim != 2:
        raise RuntimeError(
            f"Doxa requires a 2-D grayscale image; "
            f"got {grayscale.shape}"
        )

    # IMPORTANT:
    #
    # This MUST be Binarization.Algorithms.SAUVOLA.
    #
    # Do NOT replace this with:
    #
    #     doxapy.Algorithms.SAUVOLA
    #
    # Those are different enum types.
    algorithm = doxapy.Binarization.Algorithms.SAUVOLA

    binary = doxapy.to_binary(
        algorithm,
        grayscale,
        {
            "window": window,
            "k": k,
        },
    )

    binary = np.asarray(binary, dtype=np.uint8)

    # Doxa normally returns 0/255. Normalize it explicitly anyway.
    binary = np.where(
        binary != 0,
        255,
        0,
    ).astype(np.uint8)

    return Image.fromarray(binary, mode="L")


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode an image as lossless PNG bytes."""
    output = io.BytesIO()

    image.save(
        output,
        format="PNG",
        optimize=True,
    )

    return output.getvalue()


def process_pdf(
    input_path: Path,
    output_path: Path,
    mode: str,
    dpi: int,
    window: int,
    k: float,
) -> None:
    """Process the PDF."""

    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "Input and output PDF must be different files."
        )

    with pymupdf.open(input_path) as document:
        page_count = len(document)

        print(
            f"Processing {page_count} pages from "
            f"{input_path}",
            file=sys.stderr,
        )

        for page_number, page in enumerate(
            document,
            start=1,
        ):
            print(
                f"  Page {page_number}/{page_count}",
                file=sys.stderr,
                flush=True,
            )

            # Render the complete visual page.
            rendered = render_page(
                page,
                dpi,
            )

            # Convert the rendered snapshot.
            if mode == "grayscale":
                processed = rendered.convert("L")

            elif mode == "bitonal":
                processed = binarize_with_doxa(
                    rendered,
                    window,
                    k,
                )

            else:
                raise ValueError(
                    f"Unknown mode: {mode}"
                )

            png_bytes = image_to_png_bytes(
                processed
            )

            # Put the processed image on top of the
            # existing PDF page.
            #
            # The original PDF page contents are NOT deleted.
            #
            # Therefore:
            #
            #   visible appearance = processed image
            #
            #   searchable text    = original PDF text layer
            #
            page.insert_image(
                page.rect,
                stream=png_bytes,
                keep_proportion=False,
                overlay=True,
            )

        # Save to a temporary file first so that a failed
        # operation doesn't destroy an existing output.
        temporary_path = output_path.with_name(
            output_path.name + ".tmp"
        )

        try:
            document.save(
                temporary_path,
                garbage=4,
                deflate=True,
                clean=True,
            )

            temporary_path.replace(output_path)

        except Exception:
            temporary_path.unlink(
                missing_ok=True
            )
            raise

    print(
        f"Written: {output_path}",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render PDF pages and apply Doxa adaptive "
            "binarization while preserving the original "
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
        help=(
            "output PDF path "
            "(default: INPUT.blackwhite.pdf)"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=("bitonal", "grayscale"),
        default="bitonal",
        help=(
            "processed page image mode "
            "(default: bitonal)"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=(
            f"PDF rendering resolution in DPI "
            f"(default: {DEFAULT_DPI})"
        ),
    )

    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=(
            f"Sauvola local window size "
            f"(default: {DEFAULT_WINDOW})"
        ),
    )

    parser.add_argument(
        "--k",
        type=float,
        default=DEFAULT_K,
        help=(
            f"Sauvola k parameter "
            f"(default: {DEFAULT_K})"
        ),
    )

    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error("--dpi must be greater than zero")

    if args.window <= 0:
        parser.error("--window must be positive")

    if args.window % 2 == 0:
        parser.error("--window must be odd")

    if not 0 < args.k < 1:
        parser.error("--k must be between 0 and 1")

    return args


def main() -> int:
    args = parse_args()

    input_path = args.input

    if not input_path.is_file():
        print(
            f"error: input file does not exist: "
            f"{input_path}",
            file=sys.stderr,
        )
        return 2

    if input_path.suffix.lower() != ".pdf":
        print(
            f"error: input file must have a .pdf extension: "
            f"{input_path}",
            file=sys.stderr,
        )
        return 2

    output_path = (
        args.output
        if args.output is not None
        else default_output_path(input_path)
    )

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
        print(
            f"error: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
