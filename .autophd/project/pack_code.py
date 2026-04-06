#!/usr/bin/env python3
"""Pack train_gpt.py with lzma+base85 for artifact compliance.

Creates a self-extracting Python file that decompresses and exec()s the original.
The packed version is typically 60-65% smaller.

Usage:
    python3 pack_code.py train_gpt.py                  # pack in-place
    python3 pack_code.py train_gpt.py -o packed.py      # pack to separate file
    python3 pack_code.py train_gpt.py --verify          # pack + verify decompression
    python3 pack_code.py train_gpt.py --unpack          # decompress packed file
"""

import argparse
import base64
import lzma
import os
import re
import sys


def pack(input_path, output_path=None, verify=False):
    """Pack a Python file with lzma+base85 self-extraction."""
    with open(input_path, "rb") as f:
        raw = f.read()

    original_size = len(raw)

    # Check if already packed
    if b"b85decode" in raw and b"lzma" in raw:
        print(f"Already packed: {input_path} ({original_size} bytes)")
        return

    # Compress
    compressed = lzma.compress(raw, preset=9)
    encoded = base64.b85encode(compressed).decode("ascii")

    # Build self-extracting wrapper
    packed = f"import lzma,base64;exec(compile(lzma.decompress(base64.b85decode(b'{encoded}')),'train_gpt.py','exec'))\n"
    packed_size = len(packed.encode())

    out = output_path or input_path
    with open(out, "w") as f:
        f.write(packed)

    ratio = 100 * (1 - packed_size / original_size)
    print(f"Packed: {original_size} -> {packed_size} bytes ({ratio:.1f}% reduction)")
    print(f"Output: {out}")

    if verify:
        # Verify decompression produces identical content
        m = re.search(r"b85decode\(b'(.+?)'\)", packed, re.DOTALL)
        if m:
            restored = lzma.decompress(base64.b85decode(m.group(1)))
            if restored == raw:
                print("Verify: OK (decompressed matches original)")
            else:
                print("Verify: FAIL (decompressed does NOT match original!)", file=sys.stderr)
                sys.exit(1)

        # Verify vocab detection works on packed code
        m2 = re.search(r"VOCAB_SIZE.*?,\s*(\d+)", packed)
        if not m2:
            # Try decompressing to find it
            m3 = re.search(r"VOCAB_SIZE.*?,\s*(\d+)", restored.decode())
            if m3:
                print(f"Verify: vocab={m3.group(1)} (detectable via decompression)")
            else:
                print("Verify: WARNING - VOCAB_SIZE not found even after decompression")
        else:
            print(f"Verify: vocab={m2.group(1)} (detectable in header)")


def unpack(input_path, output_path=None):
    """Unpack a lzma+base85 packed Python file."""
    with open(input_path) as f:
        content = f.read()

    m = re.search(r"b85decode\(b'(.+?)'\)", content, re.DOTALL)
    if not m:
        print(f"Not a packed file: {input_path}", file=sys.stderr)
        sys.exit(1)

    raw = lzma.decompress(base64.b85decode(m.group(1)))
    out = output_path or input_path
    with open(out, "wb") as f:
        f.write(raw)

    print(f"Unpacked: {len(content)} -> {len(raw)} bytes")
    print(f"Output: {out}")


def main():
    parser = argparse.ArgumentParser(description="Pack/unpack train_gpt.py for artifact compliance")
    parser.add_argument("input", help="Path to train_gpt.py")
    parser.add_argument("-o", "--output", help="Output path (default: overwrite input)")
    parser.add_argument("--verify", action="store_true", help="Verify decompression after packing")
    parser.add_argument("--unpack", action="store_true", help="Unpack instead of pack")
    args = parser.parse_args()

    if args.unpack:
        unpack(args.input, args.output)
    else:
        pack(args.input, args.output, verify=args.verify)


if __name__ == "__main__":
    main()
