#!/usr/bin/env python3
"""Tests for evaluate.py parsing, detection, and result handling.

Run: python3 test_evaluate.py
"""

import json
import os
import re
import sys
import tempfile
import unittest

# Import evaluate.py functions by exec (it's not a proper module)
EVAL_PATH = os.path.join(os.path.dirname(__file__), "evaluate.py")
_ns = {}
with open(EVAL_PATH) as f:
    code = f.read()
# Only extract the functions we need, skip main() execution
# Parse function definitions from the file
exec(compile(
    "\n".join(line for line in code.split("\n")
              if not line.startswith("_load_env")  # skip env loading side effects
              ),
    EVAL_PATH, "exec"), _ns)

_extract_results = _ns["_extract_results"]
_log_has_results = _ns["_log_has_results"]
_has_final_results_content = _ns["_has_final_results_content"]
_make_job_command = _ns["_make_job_command"]


class TestExtractResults(unittest.TestCase):
    """Test _extract_results() log parsing."""

    def test_roundtrip_only(self):
        log = "final_int6_roundtrip_exact val_loss:2.84280373 val_bpb:1.10053791 eval_time:27711ms"
        r = _extract_results(log)
        self.assertAlmostEqual(r["val_bpb"], 1.10053791)
        self.assertAlmostEqual(r["val_loss"], 2.84280373)

    def test_sliding_overwrites_roundtrip(self):
        """Sliding is run after roundtrip — last match should win."""
        log = """final_int6_roundtrip_exact val_loss:2.84280373 val_bpb:1.10053791 eval_time:27711ms
final_int6_sliding_window val_loss:2.79976457 val_bpb:1.08387612 eval_time:124497ms"""
        r = _extract_results(log)
        self.assertAlmostEqual(r["val_bpb"], 1.08387612, msg="sliding should overwrite roundtrip")

    def test_ttt_overwrites_sliding(self):
        """TTT runs last — should be the final val_bpb."""
        log = """final_int6_roundtrip_exact val_loss:2.84280373 val_bpb:1.10053791 eval_time:27711ms
final_int6_sliding_window val_loss:2.79976457 val_bpb:1.08387612 eval_time:124497ms
legal_ttt_exact val_loss:2.79583345 val_bpb:1.08235425 eval_time:429509ms"""
        r = _extract_results(log)
        self.assertAlmostEqual(r["val_bpb"], 1.08235425, msg="TTT should overwrite sliding")

    def test_ttt_sliding_done_format(self):
        """ttt_sliding:done uses val_bpb= not val_bpb: — must match both."""
        log = """final_int6_sliding_window val_loss:2.79976457 val_bpb:1.08387612 eval_time:124497ms
ttt_sliding:done val_loss=2.795833 val_bpb=1.082354 elapsed=429.3s"""
        r = _extract_results(log)
        self.assertAlmostEqual(r["val_bpb"], 1.082354, msg="ttt_sliding:done with = format")

    def test_training_checkpoints_ignored(self):
        """Training checkpoints (no eval_time) should NOT be parsed as results."""
        log = """0/20000 val_loss: 9.0047 val_bpb: 3.4860
4000/20000 val_loss: 2.9178 val_bpb: 1.1296
5033/20000 val_loss: 2.8152 val_bpb: 1.0898"""
        r = _extract_results(log)
        self.assertNotIn("val_bpb", r, msg="training checkpoints should be ignored")

    def test_pre_quant_ema_with_eval_time(self):
        """pre-quantization post-ema line has eval_time and should match."""
        log = "pre-quantization post-ema val_loss:2.81260634 val_bpb:1.08884756 eval_time:7480ms"
        r = _extract_results(log)
        self.assertAlmostEqual(r["val_bpb"], 1.08884756)

    def test_artifact_size(self):
        log = "Total submission size quantized+brotli: 15985531 bytes"
        r = _extract_results(log)
        self.assertEqual(r["bytes_total"], 15985531)

    def test_peak_memory(self):
        log = "peak memory allocated: 34604 MiB reserved: 34708 MiB"
        r = _extract_results(log)
        self.assertEqual(r["peak_memory_mib"], 34604)

    def test_results_json_format(self):
        """results_json: line should be parsed as structured JSON."""
        log = 'results_json: {"val_bpb": 1.082, "val_loss": 2.796, "bytes_total": 15985531}'
        r = _extract_results(log)
        self.assertAlmostEqual(r["val_bpb"], 1.082)
        self.assertEqual(r["bytes_total"], 15985531)

    def test_results_json_overrides_line_parsing(self):
        """results_json appears after line-based results — JSON should win."""
        log = """final_int6_sliding_window val_loss:2.79976457 val_bpb:1.08387612 eval_time:124497ms
results_json: {"val_bpb": 1.082, "val_loss": 2.796}"""
        r = _extract_results(log)
        self.assertAlmostEqual(r["val_bpb"], 1.082, msg="results_json should override line parsing")

    def test_empty_log(self):
        r = _extract_results("")
        self.assertEqual(r, {})

    def test_unknown_eval_label_with_eval_time(self):
        """Any new eval method with eval_time should be picked up automatically."""
        log = "my_custom_eval_method val_loss:2.80000000 val_bpb:1.08400000 eval_time:500000ms"
        r = _extract_results(log)
        self.assertAlmostEqual(r["val_bpb"], 1.084, msg="unknown eval label with eval_time should match")

    def test_mixed_checkpoint_and_final(self):
        """Full realistic log with checkpoints + final results."""
        log = """0/20000 val_loss: 9.0047 val_bpb: 3.4860
500/20000 train_loss: 3.3964 train_time: 0.9m tok/s: 7669991
4000/20000 val_loss: 2.9178 val_bpb: 1.1297
5048/20000 val_loss: 2.8146 val_bpb: 1.0896
stopping_early: wallclock_cap train_time: 588029ms step: 5048/20000
peak memory allocated: 34604 MiB reserved: 34708 MiB
pre-quantization post-ema val_loss:2.81209012 val_bpb:1.08864772 eval_time:6365ms
Serialized model: 135426937 bytes
Total submission size quantized+brotli: 15972843 bytes
final_int6_roundtrip_exact val_loss:2.84337910 val_bpb:1.10076066 eval_time:26707ms
final_int6_sliding_window val_loss:2.80007025 val_bpb:1.08399445 eval_time:123325ms"""
        r = _extract_results(log)
        self.assertAlmostEqual(r["val_bpb"], 1.08399445, msg="sliding should be final")
        self.assertEqual(r["bytes_total"], 15972843)
        self.assertEqual(r["peak_memory_mib"], 34604)

    def test_full_log_with_ttt(self):
        """Full log including TTT — TTT result must be final."""
        log = """5033/20000 val_loss: 2.8152 val_bpb: 1.0898
stopping_early: wallclock_cap train_time: 588124ms step: 5033/20000
pre-quantization post-ema val_loss:2.81260634 val_bpb:1.08884756 eval_time:7480ms
Serialized model: 135426937 bytes
Total submission size quantized+brotli: 15972056 bytes
final_int6_roundtrip_exact val_loss:2.84280373 val_bpb:1.10053791 eval_time:27711ms
final_int6_sliding_window val_loss:2.79976457 val_bpb:1.08387612 eval_time:124497ms
ttt_sliding:start chunks=1238 chunk_tokens=32768 total_windows=633409 stride=64
ttt_sliding:params unfrozen=35943512 frozen=0
ttt_sliding:done val_loss=2.795833 val_bpb=1.082354 elapsed=429.3s
legal_ttt_exact val_loss:2.79583345 val_bpb:1.08235425 eval_time:429509ms"""
        r = _extract_results(log)
        self.assertAlmostEqual(r["val_bpb"], 1.08235425, msg="legal_ttt should be final (last match)")
        self.assertEqual(r["bytes_total"], 15972056)


class TestLogHasResults(unittest.TestCase):
    """Test _log_has_results() detection."""

    def test_has_final_int6(self):
        self.assertTrue(_log_has_results_str("final_int6_roundtrip_exact val_bpb:1.10 eval_time:27000ms"))

    def test_has_ttt_done(self):
        self.assertTrue(_log_has_results_str("ttt_sliding:done val_bpb=1.082 elapsed=429s eval_time:429000ms"))

    def test_has_results_json(self):
        self.assertTrue(_log_has_results_str('results_json: {"val_bpb": 1.08}'))

    def test_training_checkpoint_only(self):
        self.assertFalse(_log_has_results_str("4000/20000 val_loss: 2.9178 val_bpb: 1.1297"))

    def test_empty(self):
        self.assertFalse(_log_has_results_str(""))

    def test_partial_log_no_eval(self):
        self.assertFalse(_log_has_results_str(
            "500/20000 train_loss: 3.39 train_time: 0.9m\n"
            "1000/20000 train_loss: 3.20 train_time: 1.7m"
        ))

    def test_ttt_quantized_only_is_not_final(self):
        content = """ttt_enabled: True
pre-quantization post-ema val_loss:2.78 val_bpb:1.077 eval_time:9709ms
quantized val_loss:2.81 val_bpb:1.088 eval_time:68979ms
ttt_lora:warming up compile"""
        self.assertFalse(_log_has_results_str(content))

    def test_ttt_final_marker_is_final(self):
        content = """ttt_enabled: True
quantized val_loss:2.81 val_bpb:1.088 eval_time:68979ms
quantized_ttt_lora val_loss:2.79 val_bpb:1.077 eval_time:429509ms"""
        self.assertTrue(_log_has_results_str(content))

    def test_slot_roundtrip_is_not_final(self):
        content = """slot_enabled: True
final_int6_roundtrip_exact val_loss:1.88 val_bpb:1.114 eval_time:22797ms"""
        self.assertFalse(_log_has_results_str(content))

    def test_slot_final_marker_is_final(self):
        content = """slot_enabled: True
final_causal_slot val_loss:1.70 val_bpb:1.0069 time:474604ms"""
        self.assertTrue(_log_has_results_str(content))

    def test_sliding_roundtrip_only_is_not_final(self):
        content = """sliding_window_enabled: True
final_int6_roundtrip_exact val_loss:2.84 val_bpb:1.10 eval_time:27711ms"""
        self.assertFalse(_log_has_results_str(content))

    def test_sliding_marker_is_final(self):
        content = """sliding_window_enabled: True
final_int6_sliding_window val_loss:2.80 val_bpb:1.083 eval_time:124497ms"""
        self.assertTrue(_log_has_results_str(content))


class TestFinalMarkerDetection(unittest.TestCase):
    def test_results_json_short_circuits(self):
        self.assertTrue(_has_final_results_content('results_json: {"val_bpb": 1.08}'))


class TestJobCommand(unittest.TestCase):
    def test_requirements_filter_skips_runtime_owned_packages(self):
        cmd = _make_job_command("deadbeef", branch="exp/test")
        self.assertIn("requirements_filter: skip", cmd)
        self.assertIn("'torch'", cmd)
        self.assertIn("'flash_attn_3'", cmd)
        self.assertIn('"$PYBIN" -m pip install --no-deps -r "$FILTERED_REQ"', cmd)

    def test_job_command_selects_python_with_torch(self):
        cmd = _make_job_command("deadbeef", branch="exp/test")
        self.assertIn('python_exec=$PYBIN', cmd)
        self.assertIn('import torch', cmd)
        self.assertIn('"$PYBIN" -m torch.distributed.run --nproc_per_node=8 train_gpt.py', cmd)


def _log_has_results_str(content):
    """Helper: test _log_has_results with a string instead of a file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(content)
        f.flush()
        result = _log_has_results(f.name)
    os.unlink(f.name)
    return result


class TestVocabDetection(unittest.TestCase):
    """Test vocab size detection from train_gpt.py (Method 1, 2, 3)."""

    def _detect_vocab(self, code_content):
        """Run the exact vocab detection logic from evaluate.py."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
            f.write(code_content)
            f.flush()
            import subprocess
            # Exact same logic as evaluate.py data_setup section
            detect_script = """
import re,sys,io
f=open('%s').read()
found=None
m=re.search(r'VOCAB_SIZE.*?,\\s*(\\d+)',f)
if m: found=m.group(1)
if not found:
  try:
    import types
    for mod in ['torch','numpy','sentencepiece','flash_attn_3']:
      if mod not in sys.modules: sys.modules[mod]=types.ModuleType(mod)
    ns={'__builtins__':__builtins__,'os':__import__('os')}
    old=sys.stdout; sys.stdout=io.StringIO()
    try: exec(compile(f,'train_gpt.py','exec'),ns)
    except SystemExit: pass
    finally: sys.stdout=old
    for v in ns.values():
      if hasattr(v,'vocab_size'): found=str(int(v.vocab_size)); break
  except Exception: pass
if not found:
  try:
    import lzma,base64
    m2=re.search(r"b85decode\\(b'(.+?)'\\)",f,re.DOTALL)
    if m2:
      code=lzma.decompress(base64.b85decode(m2.group(1))).decode()
      m3=re.search(r'VOCAB_SIZE.*?,\\s*(\\d+)',code)
      if m3: found=m3.group(1)
  except Exception: pass
print(found or '1024')
""" % f.name
            r = subprocess.run([sys.executable, "-c", detect_script],
                             capture_output=True, text=True, timeout=10)
        os.unlink(f.name)
        return r.stdout.strip()

    def test_unpacked_sp8192(self):
        code = "VOCAB_SIZE', 8192)\nrest of code..."
        self.assertEqual(self._detect_vocab(code), "8192")

    def test_unpacked_sp4096(self):
        code = "VOCAB_SIZE', 4096)\nrest of code..."
        self.assertEqual(self._detect_vocab(code), "4096")

    def test_unpacked_sp1024(self):
        code = "VOCAB_SIZE', 1024)\nrest of code..."
        self.assertEqual(self._detect_vocab(code), "1024")

    def test_packed_lzma_base85(self):
        """Packed code should be decompressed to find VOCAB_SIZE."""
        import lzma, base64
        inner = b"class Hyperparameters:\n  vocab_size=int(os.environ.get('VOCAB_SIZE', 8192))\n"
        blob = base64.b85encode(lzma.compress(inner, preset=9)).decode()
        packed = f"import lzma,base64;exec(compile(lzma.decompress(base64.b85decode(b'{blob}')),'t.py','exec'))"
        self.assertEqual(self._detect_vocab(packed), "8192")

    def test_packed_with_header_comment(self):
        """Packed code with VOCAB_SIZE header should match via Method 1."""
        code = "# VOCAB_SIZE, 8192\nimport lzma,base64;exec(...)"
        self.assertEqual(self._detect_vocab(code), "8192")

    def test_no_vocab_defaults_1024(self):
        code = "print('hello world')"
        self.assertEqual(self._detect_vocab(code), "1024")


class TestOutputResultFile(unittest.TestCase):
    """Test that _output() writes last_result.json."""

    def test_result_file_written(self):
        """_output() should write result to WORKSPACE/last_result.json before exiting."""
        workspace = _ns.get("WORKSPACE", os.path.expanduser("~/autoresearch/pgolf"))
        result_file = os.path.join(workspace, "last_result.json")

        # Remove existing file
        if os.path.exists(result_file):
            os.unlink(result_file)

        # _output calls sys.exit, so we catch SystemExit
        try:
            _ns["_output"](True, score=-1.084, details={"val_bpb": 1.084})
        except SystemExit:
            pass

        self.assertTrue(os.path.exists(result_file), "last_result.json should exist")
        with open(result_file) as f:
            data = json.load(f)
        self.assertTrue(data["pass"])
        self.assertAlmostEqual(data["score"], -1.084)
        self.assertAlmostEqual(data["details"]["val_bpb"], 1.084)


class TestRemoteScript(unittest.TestCase):
    """Test that _make_job_command generates valid shell."""

    def _get_script(self):
        fn = _ns.get("_make_job_command")
        if fn is None:
            self.skipTest("_make_job_command not found")
        return fn("abc1234", branch="exp/test")

    def test_script_has_torchrun(self):
        script = self._get_script()
        self.assertIn('"$PYBIN" -m torch.distributed.run --nproc_per_node=8 train_gpt.py', script)

    def test_script_has_results_json_collector(self):
        script = self._get_script()
        self.assertIn("results_json", script, "post-training result collector should be in script")

    def test_script_has_vocab_detection(self):
        script = self._get_script()
        self.assertIn("VOCAB", script)

    def test_script_vocab_detection_complete(self):
        """Vocab detection block must produce a VOCAB variable."""
        script = self._get_script()
        self.assertIn("VOCAB=", script, "Script must set VOCAB variable")
        # No duplicate closing ") (from old python3 -c pattern)
        self.assertNotIn('")\n")', script, "Duplicate closing paren in script")

    def test_script_no_fstring_braces_error(self):
        """Embedded Python must not have unescaped {} inside f-strings."""
        try:
            script = self._get_script()
        except (KeyError, ValueError) as e:
            self.fail(f"f-string escaping error in _make_job_command: {e}")

    def test_script_shell_syntax_valid(self):
        """Generated script should pass bash -n syntax check."""
        script = self._get_script()
        import subprocess
        r = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, f"Shell syntax error: {r.stderr[:500]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
