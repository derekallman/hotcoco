"""Image sizes for the converters' ``images_dir``.

YOLO, DOTA, and Open Images store no pixel size, so ``from_yolo``,
``from_dota``, and ``from_oid`` read each image's ``(width, height)`` from
``images_dir``, keyed by file stem. Pillow, which read them until 1.5, is the
reference here: every format the converters look for, written by Pillow, must
come back at the size Pillow reports.
"""

import pytest
from hotcoco import COCO

Image = pytest.importorskip("PIL.Image")

# Sizes distinct per format and off any power of two, so a swapped width and
# height, or one format's size read for another, cannot pass.
FORMATS = {
    "a.png": (37, 21),
    "b.jpg": (64, 17),
    "c.jpeg": (19, 45),
    "d.bmp": (23, 31),
    "e.tif": (41, 13),
    "f.tiff": (11, 29),
    "g.PNG": (30, 7),
}


def write_images(images_dir):
    images_dir.mkdir()
    for name, size in FORMATS.items():
        Image.new("RGB", size, (200, 10, 10)).save(images_dir / name)
    # Not an image extension: never read.
    (images_dir / "notes.txt").write_text("not an image")
    return {name.rsplit(".", 1)[0]: Image.open(images_dir / name).size for name in FORMATS}


def yolo_dir(tmp_path, stems):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "data.yaml").write_text("names:\n  0: thing\n")
    for stem in stems:
        (labels / f"{stem}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    return labels


def test_from_yolo_reads_every_format_at_pillows_size(tmp_path):
    want = write_images(tmp_path / "images")
    coco = COCO.from_yolo(str(yolo_dir(tmp_path, want)), images_dir=str(tmp_path / "images"))
    got = {img["file_name"].rsplit(".", 1)[0]: (img["width"], img["height"]) for img in coco.dataset["images"]}
    assert got == want
    # The box denormalized by that size: a quarter of each side, centered.
    for img in coco.dataset["images"]:
        (ann,) = coco.load_anns(coco.get_ann_ids(img_ids=[img["id"]]))
        w, h = img["width"], img["height"]
        assert ann["bbox"] == pytest.approx([0.375 * w, 0.375 * h, 0.25 * w, 0.25 * h])


def test_from_dota_reads_image_sizes(tmp_path):
    want = write_images(tmp_path / "images")
    labels = tmp_path / "dota"
    labels.mkdir()
    for stem in want:
        (labels / f"{stem}.txt").write_text("1 1 5 1 5 5 1 5 plane 0\n")
    coco = COCO.from_dota(str(labels), images_dir=str(tmp_path / "images"))
    got = {img["file_name"].rsplit(".", 1)[0]: (img["width"], img["height"]) for img in coco.dataset["images"]}
    assert got == want


def bmp_header(width, height, dib_size=40):
    """A 24-bit BMP's file and DIB headers, with no pixels: enough for a size.
    ``dib_size=12`` is OS/2's BITMAPCOREHEADER, whose sides are 16-bit."""
    import struct

    if dib_size == 12:
        dib = struct.pack("<IHHHH", 12, width, height, 1, 24)
    else:
        dib = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, 0, 2835, 2835, 0, 0)
    return struct.pack("<2sIHHI", b"BM", 14 + len(dib), 0, 0, 14 + len(dib)) + dib


@pytest.mark.parametrize(
    ("header", "size"),
    [
        (bmp_header(5, 3), (5, 3)),
        # Top-down: the height is stored negative. Read unsigned, it was 2**32 - 3.
        (bmp_header(5, -3), (5, 3)),
        (bmp_header(7, 9, dib_size=12), (7, 9)),
    ],
    ids=["bottom-up", "top-down", "os2"],
)
def test_bmp_sizes_are_pillows(tmp_path, header, size):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.bmp").write_bytes(header + b"\x00" * 64)
    coco = COCO.from_yolo(str(yolo_dir(tmp_path, ["a"])), images_dir=str(images))
    (img,) = coco.dataset["images"]
    assert (img["width"], img["height"]) == size


@pytest.mark.parametrize("fmt", ["GIF", "WEBP"])
def test_a_format_pillow_reads_under_another_extension_still_reads(tmp_path, fmt):
    """Pillow detects the format from the content, so a GIF or WebP named
    ``.jpg`` read before 1.5; one such file anywhere in ``images_dir`` must
    not fail the whole import."""
    from PIL import features

    if fmt == "WEBP" and not features.check("webp"):
        pytest.skip("this Pillow has no WebP support")
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (33, 17)).save(images / "a.jpg", format=fmt)
    coco = COCO.from_yolo(str(yolo_dir(tmp_path, ["a"])), images_dir=str(images))
    (img,) = coco.dataset["images"]
    assert (img["width"], img["height"]) == (33, 17)


@pytest.mark.parametrize("content", [b"\x00\x01 not a jpeg", b"\xff\xd8\xff", b""], ids=["junk", "truncated", "empty"])
def test_a_file_that_is_no_image_is_a_value_error_naming_it(tmp_path, content):
    """Bad content, as a label file that does not parse is: a ``ValueError``.
    Pillow raised ``OSError`` here."""
    images = tmp_path / "images"
    images.mkdir()
    (images / "broken.jpg").write_bytes(content)
    with pytest.raises(ValueError, match="broken.jpg"):
        COCO.from_yolo(str(yolo_dir(tmp_path, ["broken"])), images_dir=str(images))


def test_a_missing_images_dir_is_file_not_found_naming_it(tmp_path):
    missing = tmp_path / "nowhere"
    with pytest.raises(FileNotFoundError, match="nowhere"):
        COCO.from_yolo(str(yolo_dir(tmp_path, ["a"])), images_dir=str(missing))


def pillow_jpeg(size=(64, 17)):
    import io

    buf = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.parametrize(
    "edit",
    [
        # 0xFF fill bytes before a marker, which the JPEG standard allows (T.81 B.1.1.2).
        lambda data: data.replace(b"\xff\xc0", b"\xff\xff\xff\xc0", 1),
        lambda data: data[:2] + b"\xff" + data[2:],
        # Bytes between two segments that start no marker, which Pillow skips.
        lambda data: data.replace(b"\xff\xdb", b"\x00\x00\xff\xdb", 1),
    ],
    ids=["fill-before-sof", "fill-after-soi", "junk-before-dqt"],
)
def test_a_jpeg_pillow_reads_has_pillows_size(tmp_path, edit):
    """Pillow walks a JPEG's markers past fill and junk bytes; a reader that
    stops at them failed the whole import on a file 1.4 read."""
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(edit(pillow_jpeg()))
    want = Image.open(images / "a.jpg").size
    coco = COCO.from_yolo(str(yolo_dir(tmp_path, ["a"])), images_dir=str(images))
    (img,) = coco.dataset["images"]
    assert (img["width"], img["height"]) == want == (64, 17)


def test_the_file_an_error_names_is_the_same_every_run(tmp_path):
    """Sizes are read in parallel, but the error is the first failing file's in
    directory order, as a serial loop raises it, not whichever worker failed
    first."""
    images = tmp_path / "images"
    images.mkdir()
    stems = [f"bad{i:02d}" for i in range(32)]
    for stem in stems:
        (images / f"{stem}.jpg").write_bytes(b"\x00\x01 not a jpeg")
    labels = str(yolo_dir(tmp_path, stems))
    messages = set()
    for _ in range(40):
        with pytest.raises(ValueError) as caught:
            COCO.from_yolo(labels, images_dir=str(images))
        messages.add(str(caught.value))
    assert len(messages) == 1, messages


def pillow_bytes(fmt, size=(33, 17)):
    import io

    buf = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buf, format=fmt)
    return buf.getvalue()


def cgbi_png():
    """A PNG with Apple's ``CgBI`` chunk before ``IHDR``, as ``pngcrush -iphone``
    writes them: the size is no longer at a fixed offset."""
    import struct
    import zlib

    data = pillow_bytes("PNG", (37, 21))
    body = b"\x50\x00\x20\x06"
    chunk = struct.pack(">I", len(body)) + b"CgBI" + body + struct.pack(">I", zlib.crc32(b"CgBI" + body))
    return data[:8] + chunk + data[8:]


def gif_with_a_small_screen():
    """A GIF whose logical screen is smaller than its first frame, which Pillow
    grows to the frame."""
    data = bytearray(pillow_bytes("GIF"))
    data[6:10] = (20).to_bytes(2, "little") + (10).to_bytes(2, "little")
    return bytes(data)


def vp8_with_scale_bits():
    """A lossy WebP whose sides carry the two scale bits, which libwebp masks off."""
    from PIL import features

    if not features.check("webp"):
        pytest.skip("this Pillow has no WebP support")
    data = bytearray(pillow_bytes("WEBP"))
    assert data[12:16] == b"VP8 ", "Pillow wrote a WebP that is not simple lossy"
    data[27] |= 0x40
    data[29] |= 0x80
    return bytes(data)


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("a.png", cgbi_png),
        ("a.jpg", gif_with_a_small_screen),
        ("a.jpg", vp8_with_scale_bits),
        ("a.jpg", lambda: pillow_bytes("ICO")),
        ("a.jpg", lambda: pillow_bytes("PPM")),
        ("a.png", lambda: pillow_bytes("TGA")),
    ],
    ids=["png-cgbi-first", "gif-frame-past-screen", "webp-vp8-scale-bits", "ico", "ppm", "tga"],
)
def test_a_file_pillow_reads_has_pillows_size(tmp_path, name, content):
    """Each read at the size Pillow reports: a header field at a fixed offset
    gave another size with no error, and a format that was not built in failed
    the whole import."""
    images = tmp_path / "images"
    images.mkdir()
    (images / name).write_bytes(content())
    want = Image.open(images / name).size
    coco = COCO.from_yolo(str(yolo_dir(tmp_path, ["a"])), images_dir=str(images))
    (img,) = coco.dataset["images"]
    assert (img["width"], img["height"]) == want
