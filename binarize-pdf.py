#!/usr/bin/env python3

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from pdf2image import convert_from_path
import pdf2image
import pymupdf
from PIL import Image
from tqdm import tqdm
import doxapy


def mean_std(im, window_size):
    """Calculate local mean and standard deviation using integral images."""
    print(f"Input image shape: {im.shape}")
    if window_size % 2 == 0:
        window_size += 1

    pad_size = window_size // 2
    im_pad = cv2.copyMakeBorder(
        im, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT
    )
    print(f"Padded image shape: {im_pad.shape}")

    # Compute integral images
    integral = cv2.integral(im_pad)
    integral_sq = cv2.integral(im_pad.astype(np.float32) ** 2)
    print(f"Integral image shape: {integral.shape}")

    # Extract windows
    kerSize = window_size**2

    # Calculate mean
    means = (
        integral[window_size:, window_size:]
        + integral[:-window_size, :-window_size]
        - integral[window_size:, :-window_size]
        - integral[:-window_size, window_size:]
    ) / kerSize

    # Calculate standard deviation
    means_sq = (
        integral_sq[window_size:, window_size:]
        + integral_sq[:-window_size, :-window_size]
        - integral_sq[window_size:, :-window_size]
        - integral_sq[:-window_size, window_size:]
    ) / kerSize

    variances = means_sq - means**2
    stds = np.sqrt(np.maximum(variances, 0))

    # Extract the central region that matches the input dimensions
    h, w = im.shape
    start_h = (means.shape[0] - h) // 2
    start_w = (means.shape[1] - w) // 2
    means = means[start_h : start_h + h, start_w : start_w + w]
    stds = stds[start_h : start_h + h, start_w : start_w + w]

    print(f"Final means shape: {means.shape}")
    print(f"Final stds shape: {stds.shape}")

    return means, stds


def calculate_adaptive_window_size(image_shape):
    """
    Calculate adaptive window size for Sauvola binarization.
    Window size is roughly 1/40th of the smaller image dimension,
    must be odd, and is bounded between 31 and 101 pixels.
    """
    min_dimension = min(image_shape)
    # Calculate base window size as 1/40th of smaller dimension
    window_size = min_dimension // 40
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Bound between 31 and 101
    window_size = max(31, min(101, window_size))
    
    return window_size


def binarize_image(pil_img, args):
    """
    Apply binarization to an image using a combination of methods.
    Uses adaptive window size for Sauvola binarization.
    """
    # Convert PIL to numpy
    np_img = np.array(pil_img)

    # Convert to grayscale if needed
    if pil_img.mode != "L":
        np_img = doxapy.to_grayscale(
            doxapy.GrayscaleAlgorithms.MEAN,
            np_img
        )

    window_size = calculate_adaptive_window_size(np_img.shape)
    # print(f"Using adaptive window size: {window_size}")

    ALGO_MAP = {
        "otsu": doxapy.Binarization.Algorithms.OTSU,
        "niblack": doxapy.Binarization.Algorithms.NIBLACK,
        "sauvola": doxapy.Binarization.Algorithms.SAUVOLA,
        "wolf": doxapy.Binarization.Algorithms.WOLF,
    }

    algo = ALGO_MAP[args.algo]

    options = {
        "window": window_size,
        "k": args.threshold_sensitivity,
    }

    binary = doxapy.to_binary(algo, np_img, options)

    # todo resize?

    # Convert back to PIL
    return Image.fromarray(binary)


def pdf2image_iter_from_path(path, **kwargs):
    """
    stream page images
    minimize memory usage for large input files
    """
    # https://github.com/Belval/pdf2image/issues/197
    info = pdf2image.pdfinfo_from_path(path)
    total_pages = info["Pages"]

    for page_num in range(1, total_pages + 1):
        images = pdf2image.convert_from_path(
            path,
            **kwargs,
            first_page=page_num,
            last_page=page_num,
            thread_count=1,
        )
        yield images[0]


def pymupdf_iter_from_path(path, dpi=600, grayscale=False):
    """
    stream page images
    minimize memory usage for large input files
    """
    doc = pymupdf.open(path)
    colorspace = pymupdf.csGRAY if grayscale else None
    mode = "L" if grayscale else "RGB"
    for page in doc:
        pix = page.get_pixmap(dpi=dpi, colorspace=colorspace)
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        yield img


def main():
    parser = argparse.ArgumentParser(description='Convert PDF to black and white using adaptive thresholding.')
    parser.add_argument('input_pdf', help='Input PDF file')
    parser.add_argument('--threshold-sensitivity', type=float, default=0.2,
                       help='Sauvola threshold sensitivity (k value). Higher values produce darker output. Default: 0.2')
    parser.add_argument(
        "--algo",
        default="sauvola",
        choices=["otsu", "niblack", "sauvola", "wolf"],
        help="image binarization algorithm",
    )
    parser.add_argument(
        "--dpi", # args.dpi
        default=600,
        type=int,
    )
    parser.add_argument(
        "--backend", # args.backend
        default="poppler",
        choices=["poppler", "mupdf"],
        help="PDF rendering backend",
    )
    r'''
    parser.add_argument(
        "--iter-pages", # args.iter_pages
        action="store_true",
    )
    '''
    r'''
    parser.add_argument(
        "--grayscale", # args.grayscale
        action="store_true",
    )
    '''
    args = parser.parse_args()

    input_path = Path(args.input_pdf)
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist")
        sys.exit(1)

    if input_path.suffix.lower() != ".pdf":
        print("Error: Input file must be a PDF")
        sys.exit(1)

    output_path = input_path.parent / f"bw-{input_path.name}"

    try:
        # Convert PDF to images
        print(f"Converting {input_path.name} to images...")
        kwargs = dict(
            dpi=args.dpi,
            # grayscale=args.grayscale,
            grayscale=True,
        )
        if args.backend == "mupdf":
            pages = pymupdf_iter_from_path(input_path, **kwargs)
        else:
            # default backend: poppler
            # pages = convert_from_path(input_path, **kwargs)
            pages = pdf2image_iter_from_path(input_path, **kwargs)

        info = pdf2image.pdfinfo_from_path(input_path)
        total_pages = info["Pages"]

        # Process each page
        processed_pages = []

        for page in tqdm(pages, total=total_pages, unit="page", ncols=80):
            binary = binarize_image(page, args)
            processed_pages.append(binary)

        print(f"Saving binarized PDF to {output_path}...")
        if processed_pages:
            processed_pages[0].save(
                str(output_path),
                "PDF",
                save_all=True,
                append_images=processed_pages[1:],
            )
            print("Done!")
        else:
            print("Error: No pages were processed")
            sys.exit(1)

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise e
        sys.exit(1)


if __name__ == "__main__":
    main()
