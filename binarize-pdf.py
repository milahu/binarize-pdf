#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pdf2image",
#     "opencv-python",
#     "numpy",
#     "Pillow",
#     "tqdm",
# ]
# ///

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm


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


def sauvola(im, k=0.2, window_size=None):
    """
    Sauvola binarization algorithm.
    Adapts to local content while being robust against noise.
    Window size is automatically calculated if not provided.
    """
    assert im.dtype == np.uint8
    
    if window_size is None:
        window_size = calculate_adaptive_window_size(im.shape)
    
    means, stds = mean_std(im, window_size)
    thresh = means * (1 + k * ((stds / 127) - 1))
    return (im > thresh).astype(np.uint8) * 255


def adaptive_otsu(im):
    """
    Adaptive Otsu binarization with background normalization.
    Good for handling uneven illumination.
    """
    im_h, _ = im.shape
    s = (im_h // 200) | 1
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
    background = cv2.morphologyEx(im, cv2.MORPH_DILATE, ellipse)
    bg_float = background.astype(np.float64)

    # Normalize using background estimation
    C = np.percentile(im, 30)
    normalized = np.clip(C / (bg_float + 1e-10) * im, 0, 255).astype(np.uint8)

    # Apply Otsu's method
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def binarize_image(image, sauvola_k=0.2):
    """
    Apply binarization to an image using a combination of methods.
    Uses adaptive window size for Sauvola binarization.
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    window_size = calculate_adaptive_window_size(gray.shape)
    print(f"Using adaptive window size: {window_size}")
    
    sauvola_result = sauvola(gray, k=sauvola_k, window_size=window_size)
    otsu_result = adaptive_otsu(gray)
    h, w = gray.shape
    if sauvola_result.shape != gray.shape:
        print("Resizing sauvola result")
        sauvola_result = cv2.resize(
            sauvola_result, (w, h), interpolation=cv2.INTER_NEAREST
        )
    if otsu_result.shape != gray.shape:
        print("Resizing otsu result")
        otsu_result = cv2.resize(otsu_result, (w, h), interpolation=cv2.INTER_NEAREST)

    print(f"Final shapes - Sauvola: {sauvola_result.shape}, Otsu: {otsu_result.shape}")

    # Combine results - take the more conservative approach
    combined = cv2.bitwise_and(sauvola_result, otsu_result)

    return combined


def main():
    parser = argparse.ArgumentParser(description='Convert PDF to black and white using adaptive thresholding.')
    parser.add_argument('input_pdf', help='Input PDF file')
    parser.add_argument('--threshold-sensitivity', type=float, default=0.2,
                       help='Sauvola threshold sensitivity (k value). Higher values produce darker output. Default: 0.2')
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
        pages = convert_from_path(str(input_path))

        # Process each page
        processed_pages = []

        for page in tqdm(pages):
            np_image = np.array(page)
            binary = binarize_image(np_image, sauvola_k=args.threshold_sensitivity)
            processed_page = Image.fromarray(binary)
            processed_pages.append(processed_page)

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
        sys.exit(1)


if __name__ == "__main__":
    main()
