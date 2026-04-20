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
import io
from PIL import Image
from tqdm import tqdm
import doxapy
import zlib
import tifffile
import math


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
    # Convert to grayscale if needed
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")

    # Convert PIL to numpy
    np_img = np.array(pil_img)

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
    # binary = Image.fromarray(binary)

    return binary


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


def make_page_filter(spec: str):
    spec = spec.strip()
    if not spec:
        return lambda p: (False, False)  # never continue, never break

    include = set()
    ranges = []

    max_finite = None
    has_infinite_tail = False

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        # "-N"
        if part.startswith("-") and part != "-":
            end = int(part[1:])
            ranges.append(("lte", end))
            max_finite = end if max_finite is None else max(max_finite, end)

        # "N-"
        elif part.endswith("-"):
            start = int(part[:-1])
            ranges.append(("gte", start))
            has_infinite_tail = True

        # "A-B"
        elif "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if a > b:
                raise ValueError(f"Invalid range: {part}")
            ranges.append(("range", a, b))
            max_finite = b if max_finite is None else max(max_finite, b)

        # single number
        else:
            v = int(part)
            include.add(v)
            max_finite = v if max_finite is None else max(max_finite, v)

    def matcher(page: int):
        # does this page match?
        match = False

        if page in include:
            match = True

        if not match:
            for r in ranges:
                if r[0] == "lte" and page <= r[1]:
                    match = True
                elif r[0] == "gte" and page >= r[1]:
                    match = True
                elif r[0] == "range" and r[1] <= page <= r[2]:
                    match = True

        # should_continue = skip page if NOT matched
        should_continue = not match

        # should_break logic
        if has_infinite_tail:
            should_break = False
        elif max_finite is not None and page > max_finite:
            should_break = True
        else:
            should_break = False

        return should_break, should_continue

    return matcher


class StreamingPDFWriter:

    def __init__(self, path):
        self.f = open(path, "wb")
        self.f.write(b"%PDF-1.7\n\n")
        self.offsets = []
        self.obj_id = 0
        self.page_ids = []

    def write_object(self, obj_body: bytes):
        self.obj_id += 1
        offset = self.f.tell()
        self.offsets.append(offset)
        self.f.write(f"{self.obj_id} 0 obj\n".encode("ascii"))
        self.f.write(obj_body)
        self.f.write(b"\nendobj\n\n")
        return self.obj_id

    # TODO pass args
    def add_page_with_image(self, image, width, height, image_depth, image_quality):
        filters = []
        DecodeParms = None

        if image_depth == 1:
            if 1:
                # CCITT Group 4 compression
                pil_img = Image.fromarray(image, mode="L").convert("1")
                tiff_buf = io.BytesIO()
                strip_size = math.ceil(pil_img.width / 8) * pil_img.height
                pil_img.save(
                    tiff_buf,
                    format="TIFF",
                    compression="group4",
                    strip_size=strip_size, # produce single strip
                )
                tiff_buf.seek(0)
                with tifffile.TiffFile(tiff_buf) as tiff:
                    page = tiff.pages[0]
                    assert len(page.dataoffsets) == 1 # require single strip
                    offset = page.dataoffsets[0]
                    size = page.databytecounts[0]
                tiff_buf.seek(offset)
                ccitt_bytes = tiff_buf.read(size)
                del tiff_buf
                image_bytes = ccitt_bytes
                filters.append("/CCITTFaxDecode")
                DecodeParms = (
                    b"<<\n"
                    b"      /K -1\n" +
                    f"      /Columns {width}\n".encode("ascii") +
                    f"      /Rows {height}\n".encode("ascii") +
                    b"      /BlackIs1 true\n"
                    b"   >>"
                )
            elif 0:
                # Deflate compression
                packed = np.packbits(image, axis=1, bitorder="big")
                image_bytes = zlib.compress(packed.tobytes())
                filters.append("/FlateDecode")

        elif image_depth in (8, 24):
            if 1:
                grayscale = image_depth <= 8
                colorspace = pymupdf.csGRAY if grayscale else pymupdf.csRGB
                h, w = image.shape[:2]
                pix = pymupdf.Pixmap(colorspace, w, h, image.tobytes(), False)
                image_bytes = pix.tobytes("jpg", jpg_quality=image_quality)
            elif 0:
                options = [
                    int(cv2.IMWRITE_JPEG_QUALITY), image_quality,
                    # FIXME colorspace
                ]
                success, jpeg = cv2.imencode(".jpeg", image, options)
                if not success:
                    raise RuntimeError("JPEG encoding failed")
                image_bytes = jpeg.tobytes()
            filters.append("/DCTDecode") # jpeg

        else:
            raise ValueError(f"bad image_depth {image_depth}")

        width, height = int(width), int(height)

        BitsPerComponent = 8
        ColorSpace = "DeviceGray"

        if image_depth == 1:
            # ColorSpace = "DeviceGray"
            BitsPerComponent = 1
        elif image_depth == 24:
            ColorSpace = "DeviceRGB"

        img_id = self.write_object(
            b"\n"
            b"<< /Type /XObject\n"
            b"   /Subtype /Image\n" +
            f"   /Width {width}\n".encode("ascii") +
            f"   /Height {height}\n".encode("ascii") +
            f"   /ColorSpace /{ColorSpace}\n".encode("ascii") +
            f"   /BitsPerComponent {BitsPerComponent}\n".encode("ascii") +
            (f"   /Filter {' '.join(filters)}\n".encode("ascii") if filters else b"") +
            ((b"   /DecodeParms " + DecodeParms + b"\n") if DecodeParms else b"") +
            f"   /Length {len(image_bytes)}\n".encode("ascii") +
            b">>\n"
            b"stream\n" +
            image_bytes +
            b"\nendstream\n"
        )

        content = (
            "\n"
            "q\n" +
            f"{width} 0 0 {height} 0 0 cm\n" +
            "/Im1 Do\n"
            "Q\n"
        ).encode("ascii")

        content_id = self.write_object(
            f"\n<< /Length {len(content)} >>\nstream\n".encode("ascii") +
            content +
            b"\nendstream\n"
        )

        page_id = self.write_object(
            b"\n"
            b"<< /Type /Page\n" +
            f"   /MediaBox [0 0 {width} {height}]\n".encode("ascii") +
            b"   /Resources <<\n" +
            f"        /XObject << /Im1 {img_id} 0 R >>\n".encode("ascii") +
            b"   >>\n" +
            f"   /Contents {content_id} 0 R\n".encode("ascii") +
            b">>\n"
        )
        self.page_ids.append(page_id)
        return page_id

    def close(self):

        kids = " ".join(f"{pid} 0 R" for pid in self.page_ids)
        pages_id = self.write_object(
            b"\n"
            b"<< /Type /Pages\n" +
            f"   /Kids [ {kids} ]\n".encode("ascii") +
            b"   /Count {len(self.page_ids)}\n"
            b">>\n"
        )

        catalog_id = self.write_object(
            b"\n"
            b"<< /Type /Catalog\n" +
            f"   /Pages {pages_id} 0 R\n".encode("ascii") +
            b">>\n"
        )

        xref_offset = self.f.tell()

        self.f.write(b"xref\n")
        self.f.write(f"0 {len(self.offsets)+1}\n".encode("ascii"))

        self.f.write(b"0000000000 65535 f \n")

        for off in self.offsets:
            self.f.write(f"{off:010d} 00000 n \n".encode("ascii"))

        self.f.write(
            b"\n"
            b"trailer\n" +
            f"<< /Size {len(self.offsets)+1}\n".encode("ascii") +
            f"    /Root {catalog_id} 0 R >>\n".encode("ascii") +
            b"startxref\n" +
            f"{xref_offset}\n".encode("ascii") +
            b"%EOF\n"
        )

        self.f.close()


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
    parser.add_argument(
        "--singlebit", # args.singlebit
        action="store_true",
    )
    '''
    parser.add_argument(
        "--pages", # args.pages
        default="",
    )
    r'''
    parser.add_argument(
        "--stream", # args.stream
        action="store_true",
        help="stream to the output PDF file, to reduce memory usage on large input files",
    )
    default_jpeg_quality = 95
    parser.add_argument(
        "--quality", # args.quality
        default=default_jpeg_quality,
        type=int,
        help=f"JPEG compression quality in percent. default: {default_jpeg_quality}",
    )
    '''
    parser.add_argument(
        "--output", # args.output
        "-o",
        help="path to output file",
    )
    parser.add_argument(
        "--force", # args.force
        action="store_true",
        help="overwrite existing output file",
    )
    parser.add_argument(
        "--debug", # args.debug
        action="store_true",
    )
    args = parser.parse_args()

    args.grayscale = False
    args.stream = True
    args.quality = 95

    input_path = Path(args.input_pdf)
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist")
        sys.exit(1)

    if input_path.suffix.lower() != ".pdf":
        print("Error: Input file must be a PDF")
        sys.exit(1)

    if not (1 <= args.quality <= 100):
        print("Error: Quality is out of range, must be between 1 and 100")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        output_name = os.path.splitext(input_path.name)[0] + ".blackwhite.pdf"
        output_path = input_path.parent / output_name

    try:
        # todo remove grayscale?
        image_depth = 8 if args.grayscale else 1
        # image_depth = 24 # color # not possible?
        kwargs = dict(
            dpi=args.dpi,
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

        input_doc = pymupdf.open(input_path)

        if os.path.exists(output_path):
            if args.force:
                os.unlink(output_path)
            else:
                print(f"error: output file exists: {str(output_path)!r}. hint: add --force")
                sys.exit(1)

        print(f"writing {str(output_path)!r}")

        if args.stream:
            output_doc = StreamingPDFWriter(output_path)
        else:
            output_doc = pymupdf.open()  # empty PDF

        if args.debug:
            debug_dir = Path("debug_images")
            debug_dir.mkdir(exist_ok=True)

        page_filter = make_page_filter(args.pages)

        page_idx = -1

        for page in tqdm(pages, total=total_pages, unit="page", ncols=80):

            page_idx += 1

            should_break, should_continue = page_filter(page_idx + 1)
            if should_break: break
            if should_continue: continue

            binary = binarize_image(page, args)
            # page: PIL.PpmImagePlugin.PpmImageFile
            # binary: np.array

            if args.debug:
                page.save(debug_dir / f"page{page_idx}_1_original.png")
                Image.fromarray(binary).save(debug_dir / f"page{page_idx}_2_binary.png")

            rect = input_doc[page_idx].rect

            if not args.stream:
                out_page = output_doc.new_page(width=rect.width, height=rect.height)
                # todo remove RGB?
                # colorspace = pymupdf.csGRAY if grayscale else pymupdf.csRGB
                colorspace = pymupdf.csGRAY
                h, w = binary.shape[:2]
                pix = pymupdf.Pixmap(colorspace, w, h, binary.tobytes(), False)

                # FIXME handle image_depth == 1
                # we cannot use pymupdf to insert raw CCITT-G4 images?
                assert image_depth in (8, 24)

                if 1:
                    # TODO pass jpg_quality=args.quality
                    out_page.insert_image(rect, pixmap=pix)

                elif 0:
                    pix_bytes = pix.tobytes("jpg", jpg_quality=args.quality)
                    out_page.insert_image(rect, stream=pix_bytes)

                elif 0:
                    options = [
                        int(cv2.IMWRITE_JPEG_QUALITY), args.quality,
                        # FIXME colorspace
                    ]
                    success, jpeg = cv2.imencode(".jpeg", binary, options)
                    if not success:
                        raise RuntimeError("JPEG encoding failed")
                    out_page.insert_image(rect, stream=jpeg.tobytes())

            if args.stream:
                # binary = binary.astype(np.uint8)
                # binary = np.ascontiguousarray(binary)

                # # if binarized 0/1 then scale to 0/255
                # if binary.max() == 1:
                #     binary = binary * 255

                # TODO preserve the original page size?
                # but then we need to scale the page image
                # height, width = rect.height, rect.width # wrong!
                height, width = binary.shape[:2]

                output_doc.add_page_with_image(
                    binary,
                    width,
                    height,
                    image_depth,
                    args.quality,
                )

        if not args.stream:
            output_doc.save(output_path)

        output_doc.close()
        input_doc.close()

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise e
        sys.exit(1)


if __name__ == "__main__":
    main()
