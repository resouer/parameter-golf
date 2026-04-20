#!/usr/bin/env python3
"""Remote helper for evaluate.py — runs on the GPU container.

Handles vocab detection, data setup, and post-training result collection.
This is a standalone script (no shell embedding, no escaping issues).

Usage (called by evaluate.py's remote script):
    python3 remote_helper.py detect-vocab          # prints vocab size
    python3 remote_helper.py collect-results        # prints results_json line
"""

import json
import os
import re
import sys


def detect_vocab():
    """Detect vocab size from train_gpt.py. Prints integer to stdout."""
    try:
        f = open('train_gpt.py').read()
    except FileNotFoundError:
        print('1024')
        return

    # Method 1: regex on raw source (unpacked code or header comments)
    m = re.search(r'VOCAB_SIZE.*?,\s*(\d+)', f)
    if m:
        print(m.group(1))
        return

    # Method 2: decompress lzma+base85 packed code, then regex
    try:
        import lzma
        import base64
        m2 = re.search(r"b85decode\([b]?['\"](.+?)['\"]\)", f, re.DOTALL)
        if m2:
            blob = m2.group(1)
            try:
                code = lzma.decompress(base64.b85decode(blob)).decode()
            except Exception:
                code = lzma.decompress(
                    base64.b85decode(blob),
                    format=lzma.FORMAT_RAW,
                    filters=[{"id": lzma.FILTER_LZMA2}],
                ).decode()
            m3 = re.search(r'VOCAB_SIZE.*?,\s*(\d+)', code)
            if m3:
                print(m3.group(1))
                return
    except Exception:
        pass

    # Method 3: exec sandbox (any packing scheme)
    try:
        import types
        import io
        for mod in ['torch', 'numpy', 'sentencepiece', 'flash_attn_3']:
            if mod not in sys.modules:
                sys.modules[mod] = types.ModuleType(mod)
        ns = {'__builtins__': __builtins__, 'os': os}
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(compile(f, 'train_gpt.py', 'exec'), ns)
        except SystemExit:
            pass
        finally:
            sys.stdout = old_stdout
        for v in ns.values():
            if hasattr(v, 'vocab_size'):
                print(int(v.vocab_size))
                return
    except Exception:
        pass

    print('1024')


def collect_results():
    """Scan training log for results, print results_json line."""
    import glob
    log_files = sorted(glob.glob('logs/*.txt'), key=os.path.getmtime, reverse=True)
    if not log_files:
        return

    log = open(log_files[0]).read()
    r = {}
    for m in re.finditer(r'val_bpb[=: ]+(\d+\.\d+)', log):
        r['val_bpb'] = float(m.group(1))
    for m in re.finditer(r'val_loss[=: ]+(\d+\.\d+)', log):
        r['val_loss'] = float(m.group(1))
    for m in re.finditer(r'Total submission size[^:]*:\s*(\d+)', log):
        r['bytes_total'] = int(m.group(1))
    for m in re.finditer(r'peak memory allocated:\s*(\d+)\s*MiB', log):
        r['peak_memory_mib'] = int(m.group(1))

    if r.get('val_bpb'):
        print('results_json: ' + json.dumps(r))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <detect-vocab|collect-results>", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'detect-vocab':
        detect_vocab()
    elif cmd == 'collect-results':
        collect_results()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)
