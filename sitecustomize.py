"""Minimal diagnosis shim for packed train_gpt.py payloads.

This file is auto-imported by Python at startup. For this diagnosis worker we
patch the packed exec() payload to emit explicit markers around the
quantized_sliding_window evaluation, so we can distinguish:

1. never entered sliding eval
2. entered and returned
3. entered and raised
"""

from __future__ import annotations

import builtins

_ORIG_EXEC = builtins.exec

_TARGET = (
    "if h.sliding_window_enabled:timed_eval('quantized_sliding_window',"
    "eval_val_sliding,h,device,val_data,eval_model)"
)

_REPLACEMENT = (
    "if h.sliding_window_enabled:\n"
    "\tlog('diag:before_quantized_sliding_window')\n"
    "\ttry:\n"
    "\t\ttimed_eval('quantized_sliding_window',eval_val_sliding,h,device,val_data,eval_model)\n"
    "\t\tlog('diag:after_quantized_sliding_window')\n"
    "\texcept Exception as e:\n"
    "\t\tlog(f\"diag:quantized_sliding_window_error:{e!r}\")\n"
    "\t\traise"
)


def _patched_exec(obj, globals=None, locals=None):
    if isinstance(obj, str) and _TARGET in obj and "pre-quantization post-ema" in obj:
        obj = obj.replace(_TARGET, _REPLACEMENT, 1)
        print("sitecustomize:patched_quantized_sliding_window", flush=True)
    return _ORIG_EXEC(obj, globals, locals)


builtins.exec = _patched_exec
