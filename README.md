I'll create a comprehensive README.md for this PDF binarization tool.



# PDF Binarizer

A robust tool for converting color and grayscale PDFs to clean black and white documents using advanced adaptive thresholding techniques.

## Motivation

Many PDF documents, especially scanned ones, contain text and images with varying contrast, uneven illumination, and grayscale content. This can lead to higher file sizes and poor readability. This tool addresses these issues by:

- Converting documents to pure black and white while preserving text quality
- Handling uneven illumination and background noise effectively
- Reducing file sizes significantly
- Improving OCR accuracy for subsequent text extraction

The script combines two powerful binarization methods:
- Sauvola's adaptive thresholding: Excellent for handling local contrast variations
- Adaptive Otsu with background normalization: Robust against uneven illumination

## Installation

This script uses `uv` for dependency management, eliminating the need for manual pip installations. Simply ensure you have `uv` installed on your system.

If you don't have `uv` installed, you can get it from: https://github.com/astral-sh/uv

Make the script executable:
```bash
chmod +x binarize-pdf.py
```

## Usage

Basic usage:
```bash
./binarize-pdf.py input.pdf
```

Adjust threshold sensitivity:
```bash
./binarize-pdf.py input.pdf --threshold-sensitivity 0.3
```

The script will:
1. Create a new file prefixed with "bw-" (e.g., "bw-input.pdf")
2. Process each page using combined Sauvola and Otsu binarization
3. Show a progress bar during processing

### Parameters

- `--threshold-sensitivity`: Controls the binarization sensitivity (default: 0.2)
  - Higher values (e.g., 0.3) produce darker output
  - Lower values (e.g., 0.1) produce lighter output

## Technical Details

The script employs several advanced techniques:

- Adaptive window sizing based on document dimensions
- Integral image computation for efficient local statistics
- Background estimation for handling uneven illumination
- Combined thresholding approach for robust results

The window size for local adaptive thresholding is automatically calculated as approximately 1/40th of the smaller image dimension, bounded between 31 and 101 pixels for optimal performance.

## Requirements

All dependencies are managed through `uv` and include:
- pdf2image
- opencv-python
- numpy
- Pillow
- tqdm

## Known Limitations

- Processing time increases with page size and document length
- Very large documents may require significant memory
- Some complex color graphics might lose detail in conversion

## Development status

- No plans to develop it further
