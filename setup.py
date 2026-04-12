from setuptools import setup

setup(
    name="binarize-pdf",
    version="0.1.0",
    description="Convert PDF to black and white using adaptive thresholding",
    py_modules=[],
    scripts=[
        "binarize-pdf.py",
    ],
    install_requires=[
        "pdf2image",
        "pymupdf",
        "opencv-python",
        "numpy",
        "Pillow",
        "tqdm",
        "doxapy",
    ],
    python_requires=">=3.8",
)
