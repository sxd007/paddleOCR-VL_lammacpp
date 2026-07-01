"""
Update clip.vision.image_max_pixels in PaddleOCR-VL mmproj GGUF file.
Default ~1MP → ~1.6MP to preserve more detail on A4@200DPI+ pages.
"""
import struct
import os

MMPROJ = "/home/alpha/.paddlex/official_models/ppocr-vl-gguf/PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
KEY = b"clip.vision.image_max_pixels"
NEW_VALUE = 1605632

GGUF_TYPES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

with open(MMPROJ, "r+b") as f:
    magic = f.read(4)
    assert magic == b"GGUF", f"Not a GGUF file: {magic}"
    f.read(4)  # version
    f.read(8)  # tensor count
    kv_count = struct.unpack("<Q", f.read(8))[0]

    found = False
    for _ in range(kv_count):
        key_len = struct.unpack("<Q", f.read(8))[0]
        key = f.read(key_len)
        val_type = struct.unpack("<I", f.read(4))[0]

        if key == KEY:
            offset = f.tell()
            current = struct.unpack("<I", f.read(4))[0]
            print(f"Found '{key.decode()}' at offset {offset}: current={current}")
            f.seek(offset)
            f.write(struct.pack("<I", NEW_VALUE))
            f.flush()
            os.fsync(f.fileno())
            print(f"Written new value: {NEW_VALUE}")
            found = True
            break
        else:
            size = GGUF_TYPES.get(val_type)
            if size:
                f.read(size)
            elif val_type == 8:
                sl = struct.unpack("<Q", f.read(8))[0]
                f.read(sl)
            elif val_type == 9:
                at = struct.unpack("<I", f.read(4))[0]
                al = struct.unpack("<Q", f.read(8))[0]
                for _ in range(al):
                    if at == 8:
                        sll = struct.unpack("<Q", f.read(8))[0]
                        f.read(sll)
                    else:
                        f.read(GGUF_TYPES.get(at, 4))
            else:
                f.read(4)

    if not found:
        print(f"ERROR: key not found!")
        exit(1)

# Verify
with open(MMPROJ, "rb") as f:
    f.read(4 + 4 + 8 + 8)
    for _ in range(kv_count):
        kl = struct.unpack("<Q", f.read(8))[0]
        k = f.read(kl)
        vt = struct.unpack("<I", f.read(4))[0]
        if k == KEY:
            v = struct.unpack("<I", f.read(4))[0]
            print(f"Verify: {k.decode()} = {v}")
            assert v == NEW_VALUE
            print("OK - verified")
            break
        else:
            size = GGUF_TYPES.get(vt)
            if size:
                f.read(size)
            elif vt == 8:
                sl = struct.unpack("<Q", f.read(8))[0]
                f.read(sl)
            elif vt == 9:
                at = struct.unpack("<I", f.read(4))[0]
                al = struct.unpack("<Q", f.read(8))[0]
                for _ in range(al):
                    if at == 8:
                        sll = struct.unpack("<Q", f.read(8))[0]
                        f.read(sll)
                    else:
                        f.read(GGUF_TYPES.get(at, 4))

print("Done - image_max_pixels updated from ~1MP to ~1.6MP")
