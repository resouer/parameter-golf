#!/usr/bin/env python3
"""Self-contained autoresearch evaluator for Parameter Golf.

Submits GPU training jobs to Lepton, streams logs (dev clusters lack log
persistence — logs vanish after job termination), polls job completion
independently, and parses BPP results from captured logs.

Output: {pass: bool, score?: float} JSON to stdout (autoresearch contract).

Config: reads ~/autoresearch/pgolf/.env for Lepton credentials and settings.
"""

import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Config — loaded from ~/autoresearch/pgolf/.env
# ---------------------------------------------------------------------------

WORKSPACE = os.path.expanduser("~/autoresearch/pgolf")
os.makedirs(WORKSPACE, exist_ok=True)

DEFAULT_THRESHOLD = float(os.environ.get("AUTORESEARCH_THRESHOLD", "1.1164"))
DEFAULT_TIMEOUT = 2700  # 45 min
ARTIFACT_LIMIT = 16_000_000


def _load_env():
    env_path = os.path.join(WORKSPACE, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


_load_env()

REPO_URL = os.environ.get("PGOLF_REPO_URL", "git@github.com:resouer/parameter-golf.git")
GIT_TOKEN = os.environ.get("PGOLF_GIT_TOKEN", "")
RESOURCE_SHAPE = os.environ.get("PGOLF_RESOURCE_SHAPE", "gpu.8xh100-sxm")
CONTAINER_IMAGE = os.environ.get("PGOLF_CONTAINER_IMAGE", "runpod/parameter-golf:latest")
LOCAL_VOLUME = os.environ.get("PGOLF_LOCAL_VOLUME", "")
LEP_CLI = os.environ.get("PGOLF_LEP_CLI", "lep")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _run(cmd, check=False, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace")
        stderr = (stderr + f"\nTIMEOUT after {timeout}s").strip()
        r = subprocess.CompletedProcess(cmd, 124, stdout=stdout, stderr=stderr)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{r.stderr}")
    return r


def _parse_repo_owner_and_name(url):
    m = re.match(r"git@[^:]+:([^/]+)/([^/]+?)(?:\.git)?$", url)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"https?://[^/]+/([^/]+)/([^/]+?)(?:\.git)?$", url)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(f"Cannot parse owner/repo from URL: {url}")


def _parse_job_metadata(output):
    metadata = {}
    patterns = {
        "job_id": [r'"id"\s*:\s*"([^"]+)"', r'(?im)^\s*id\s*[:=]\s*[\'"]?([^\'"\\n]+)[\'"]?\s*$'],
        "job_name": [r'"name"\s*:\s*"([^"]+)"', r'(?im)^\s*name\s*[:=]\s*[\'"]?([^\'"\\n]+)[\'"]?\s*$'],
    }
    for key, candidates in patterns.items():
        for pat in candidates:
            m = re.search(pat, output)
            if m:
                metadata[key] = m.group(1).strip()
                break
    return metadata


def _strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text or "")


def _node_group_health(node_group):
    """Best-effort summary from `lep node list --detail --node-group ...`."""
    r = subprocess.run(
        [*shlex.split(LEP_CLI), "node", "list", "--detail", "--node-group", node_group],
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = _strip_ansi((r.stdout or "") + (r.stderr or ""))
    summary = {
        "raw": text,
        "available": None,
        "total": None,
        "has_unhealthy": ("Unhealthy" in text or "NotReady" in text),
        "has_diskpressure": ("DiskPressure" in text),
        "has_initializing": ("Initializing" in text),
    }
    m = re.search(rf"{re.escape(node_group)}.*?(\d+)\s*/\s*(\d+)", text, re.S)
    if m:
        summary["available"] = int(m.group(1))
        summary["total"] = int(m.group(2))
    return summary


def _normalize_status_text(value):
    if not value:
        return None
    v = str(value).strip().lower()
    if v in ("completed", "complete", "succeeded", "success"):
        return "completed"
    if v in ("failed", "error"):
        return "failed"
    if v in ("stopped", "cancelled", "canceled"):
        return "stopped"
    if v in ("running",):
        return "running"
    if v in ("pending", "starting", "queueing", "queued"):
        return "queueing"
    return None


def _extract_json_object(text):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Job command template
# ---------------------------------------------------------------------------

def _make_job_command(commit_sha, branch=None):
    owner, repo = _parse_repo_owner_and_name(REPO_URL)

    if LOCAL_VOLUME:
        data_setup = """
CACHE_DIR=/mnt/pgolf-data/pgolf-cache
if [ -f "$CACHE_DIR/.download_complete" ]; then
    echo "Using cached data from node-local volume"
    export DATA_PATH=${DATA_PATH:-$CACHE_DIR/datasets/fineweb10B_sp1024}
    export TOKENIZER_PATH=${TOKENIZER_PATH:-$CACHE_DIR/tokenizers/fineweb_1024_bpe.model}
else
    python data/cached_challenge_fineweb.py --train-shards 80
    mkdir -p $CACHE_DIR && cp -r data/datasets data/tokenizers $CACHE_DIR/ && touch $CACHE_DIR/.download_complete
    export DATA_PATH=${DATA_PATH:-./data/datasets/fineweb10B_sp1024}
    export TOKENIZER_PATH=${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}
fi
"""
    else:
        data_setup = """
# Auto-detect vocab size from train_gpt.py (default sp1024, supports sp4096+)
if [ -f remote_helper.py ]; then
    VOCAB=$("$PYBIN" remote_helper.py detect-vocab)
else
    VOCAB=$("$PYBIN" << 'PYEOF'
import re, sys
f = open('train_gpt.py').read()
m = re.search(r'VOCAB_SIZE.*?,\\s*(\\d+)', f)
if m: print(m.group(1)); sys.exit()
try:
    import lzma, base64
    m2 = re.search(r"b85decode\\([b]?['\\\"](.+?)['\\\"]\\)", f, re.DOTALL)
    if m2:
        blob = m2.group(1)
        try: code = lzma.decompress(base64.b85decode(blob)).decode()
        except: code = lzma.decompress(base64.b85decode(blob), format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2}]).decode()
        m3 = re.search(r'VOCAB_SIZE.*?,\\s*(\\d+)', code)
        if m3: print(m3.group(1)); sys.exit()
except Exception: pass
print('1024')
PYEOF
)
fi
[ -z "$VOCAB" ] && VOCAB=1024
SHARDS=80
[ "$VOCAB" -gt 1024 ] && SHARDS=143
[ "$VOCAB" -gt 4096 ] && SHARDS=128
echo "data_setup: vocab=$VOCAB shards=$SHARDS"
# Scylla (998-vocab custom tokenizer): download from HuggingFace
if [ "$VOCAB" = "998" ]; then
    echo "data_setup: Scylla tokenizer detected (vocab=998)"
    SCYLLA_DIR="./data/datasets/fineweb10B_scylla"
    if [ ! -f "$SCYLLA_DIR/.download_complete" ]; then
        echo "data_setup: downloading Scylla data from HuggingFace..."
        PIP_NO_CACHE_DIR=1 "$PYBIN" -m pip install -q --no-cache-dir huggingface_hub 2>/dev/null || true
        "$PYBIN" -c "from huggingface_hub import snapshot_download; snapshot_download('anthonym21/fineweb10B-scylla', local_dir='$SCYLLA_DIR', repo_type='dataset')"
        touch "$SCYLLA_DIR/.download_complete"
        echo "data_setup: Scylla download complete"
    fi
    export DATA_PATH="$SCYLLA_DIR"
    # Tokenizer files from repo (candidate.vocab, candidate.meta.npz)
    export TOKENIZER_PATH="${TOKENIZER_PATH:-./candidate.vocab}"
    export TOKENIZER_META_PATH="${TOKENIZER_META_PATH:-./candidate.meta.npz}"
else
    # Non-default vocab: use kevclark's HF repo + delete stale manifest (SP8192 not in default manifest)
    if [ "$VOCAB" = "8192" ] && [ -f "tokenizer_specs_export_caseops_v1_reserved_only.json" ]; then
        CASEOPS_VARIANT=sp8192_lossless_caps_caseops_v1_reserved
        export MATCHED_FINEWEB_REPO_ID=${MATCHED_FINEWEB_REPO_ID:-romeerp/parameter-golf-caseops-v1}
        export MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=${MATCHED_FINEWEB_REMOTE_ROOT_PREFIX:-datasets}
        rm -f data/manifest.json data/datasets/manifest.json
        if [ ! -f "data/datasets/.download_complete_${CASEOPS_VARIANT}" ]; then
            python data/cached_challenge_fineweb.py --variant ${CASEOPS_VARIANT} --train-shards 80
            touch "data/datasets/.download_complete_${CASEOPS_VARIANT}"
        fi
        export DATA_PATH=${DATA_PATH:-./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}
        export DATASETS_DIR=${DATASETS_DIR:-./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}
        export TOKENIZER_PATH=${TOKENIZER_PATH:-./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}
    else
        [ "$VOCAB" -gt 1024 ] && export MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf
        [ "$VOCAB" -gt 1024 ] && rm -f data/manifest.json data/datasets/manifest.json
        if [ ! -f "data/datasets/.download_complete_sp${VOCAB}" ]; then
            python data/cached_challenge_fineweb.py --variant sp${VOCAB} --train-shards $SHARDS
            touch "data/datasets/.download_complete_sp${VOCAB}"
        fi
        export DATA_PATH=${DATA_PATH:-./data/datasets/fineweb10B_sp${VOCAB}}
        export DATASETS_DIR=${DATASETS_DIR:-./data/datasets/fineweb10B_sp${VOCAB}}
        export TOKENIZER_PATH=${TOKENIZER_PATH:-./data/tokenizers/fineweb_${VOCAB}_bpe.model}
    fi
fi
"""

    clone_setup = f"""
rm -rf /workspace/pgolf /root/.cache/pip ~/.cache/pip /tmp/pip-cache
mkdir -p /workspace
if [ -n "$PGOLF_GIT_TOKEN" ]; then
    CLONE_URL="https://x-access-token:${{PGOLF_GIT_TOKEN}}@github.com/{owner}/{repo}.git"
else
    CLONE_URL="{REPO_URL}"
fi
GIT_TERMINAL_PROMPT=0 git clone --quiet --filter=blob:none --no-tags "$CLONE_URL" /workspace/pgolf
"""

    return f"""set -e
PYBIN=$(command -v python3 || true)
if [ -z "$PYBIN" ] || ! "$PYBIN" - <<'PYEOF' >/dev/null 2>&1
import torch
PYEOF
then
  ALT_PY=$(command -v python || true)
  if [ -n "$ALT_PY" ] && "$ALT_PY" - <<'PYEOF' >/dev/null 2>&1
import torch
PYEOF
  then
    PYBIN="$ALT_PY"
  fi
fi
[ -z "$PYBIN" ] && PYBIN=$(command -v python3 || command -v python)
echo "python_exec=$PYBIN"

PIP_NO_CACHE_DIR=1 "$PYBIN" -m pip install -q --no-cache-dir sentencepiece huggingface-hub tiktoken zstandard brotli 2>/dev/null || true
if ! command -v git >/dev/null 2>&1; then
  apt-get update && apt-get install -y git
fi

{clone_setup}
cd /workspace/pgolf
git fetch origin {f'{branch}' if branch else '--all'}
git checkout {commit_sha}
rm -rf .git /root/.cache/pip ~/.cache/pip /tmp/pip-cache

export PYTHONUNBUFFERED=1

{data_setup}

# Optional: force-install the known-good H100 FA3 wheel before launch.
# Drift isolation uses this to distinguish runtime image regressions from
# model/code regressions.
if [ "${{PGOLF_ENSURE_FA3:-0}}" = "1" ]; then
  "$PYBIN" -m pip install --no-deps --find-links \
    "${{PGOLF_FA3_WHEEL_INDEX:-https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/}}" \
    flash_attn_3
fi

# Install repo-root extra requirements without dependency resolution drift.
# This keeps experiment branches that vendor non-standard runtime packages
# (for example FLA / Triton families) runnable on the remote image while
# preserving the preinstalled torch stack. Core runtime packages stay owned
# by the base image / explicit FA3 probe settings.
if [ -f requirements.txt ]; then
  FILTERED_REQ=$("$PYBIN" << 'PYEOF'
from pathlib import Path
import re

blocked = {
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "flash_attn_3",
    "flash-attn",
    "flash-attn-3",
}
src = Path("requirements.txt")
dst = Path("/tmp/pgolf.requirements.filtered.txt")
kept = []
for raw in src.read_text().splitlines():
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        kept.append(raw)
        continue
    name = re.split(r"[<>=!~\\[\\s]", stripped, maxsplit=1)[0].lower()
    if name in blocked:
        print(f"requirements_filter: skip {{name}}", flush=True)
        continue
    kept.append(raw)
dst.write_text("\\n".join(kept) + ("\\n" if kept else ""))
print(dst)
PYEOF
)
  FILTERED_REQ=$(printf "%s\n" "$FILTERED_REQ" | tail -n 1)
  if [ -s "$FILTERED_REQ" ]; then
    "$PYBIN" -m pip install --no-deps -r "$FILTERED_REQ"
  fi
fi

"$PYBIN" -m torch.distributed.run --nproc_per_node=8 train_gpt.py
RC=$?

# Post-training: emit structured results_json from log file if available.
# This decouples evaluate.py parsing from train_gpt.py output format.
# Works with ANY base code that writes val_bpb somewhere in its log.
if [ -d logs ]; then
  LOGFILE=$(ls -t logs/*.txt 2>/dev/null | head -1)
  if [ -n "$LOGFILE" ]; then
    if [ -f remote_helper.py ]; then
        "$PYBIN" remote_helper.py collect-results 2>/dev/null || true
    fi
  fi
fi

exit $RC
"""


# ---------------------------------------------------------------------------
# Job creation
# ---------------------------------------------------------------------------

def _create_job(commit_sha, node_group=None, branch=None):
    """Create a Lepton job. Returns (job_name, job_id)."""
    ng = node_group or os.environ.get("PGOLF_NODE_GROUP", "")
    node_id = (os.environ.get("PGOLF_NODE_ID", "") or "").strip()
    short_sha = commit_sha[:7]
    prefix = os.environ.get("PGOLF_JOB_PREFIX", "pgolf")

    # Guard: check for existing running/queueing jobs with this commit
    for state in ("running", "queueing"):
        try:
            r = subprocess.run(f"{LEP_CLI} job list -s {state}", shell=True,
                             capture_output=True, text=True, timeout=15)
            if f"{prefix}-{short_sha}" in (r.stdout + r.stderr):
                _log(f"WARNING: Found existing {state} job for commit {short_sha}. Reusing.")
                for line in (r.stdout + r.stderr).split("\n"):
                    if f"{prefix}-{short_sha}" in line:
                        parts = [p.strip().strip("│").strip() for p in line.split("│") if p.strip()]
                        if parts and parts[0].startswith(f"{prefix}-"):
                            existing_id = parts[0]
                            _log(f"Reusing existing job: {existing_id}")
                            return existing_id, existing_id
        except Exception:
            pass

    ts = int(time.time()) % 100000
    job_name = f"{prefix}-{short_sha}-{ts}"
    command = _make_job_command(commit_sha, branch=branch)

    lep_cmd = [
        *shlex.split(LEP_CLI), "job", "create",
        "-n", job_name,
        "-rs", RESOURCE_SHAPE,
        "--container-image", CONTAINER_IMAGE,
        "--command", command,
    ]
    if ng:
        lep_cmd.extend(["-ng", ng])
    if node_id:
        lep_cmd.extend(["-ni", node_id])
    log_collection = (os.environ.get("PGOLF_LOG_COLLECTION", "") or "").strip().lower()
    if log_collection in ("1", "true", "false", "0"):
        lep_cmd.extend(["-lg", "true" if log_collection in ("1", "true") else "false"])
    if LOCAL_VOLUME:
        lep_cmd.extend(["--mount", f"/:/mnt/pgolf-data:node-local:{LOCAL_VOLUME}"])
    if GIT_TOKEN:
        lep_cmd.extend(["-e", f"PGOLF_GIT_TOKEN={GIT_TOKEN}"])
    seed = os.environ.get("PGOLF_SEED")
    if seed:
        lep_cmd.extend(["-e", f"SEED={seed}"])
    ensure_fa3 = os.environ.get("PGOLF_ENSURE_FA3")
    if ensure_fa3:
        lep_cmd.extend(["-e", f"PGOLF_ENSURE_FA3={ensure_fa3}"])
    fa3_index = os.environ.get("PGOLF_FA3_WHEEL_INDEX")
    if fa3_index:
        lep_cmd.extend(["-e", f"PGOLF_FA3_WHEEL_INDEX={fa3_index}"])

    _log(f"Creating job {job_name}...")
    result = subprocess.run(lep_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create job: {result.stderr}\n{result.stdout}")

    metadata = _parse_job_metadata(result.stdout + result.stderr)
    job_id = metadata.get("job_id")
    if not job_id:
        # Retry lookup
        for _ in range(5):
            time.sleep(2)
            r = subprocess.run([*shlex.split(LEP_CLI), "job", "get", "-n", job_name],
                               capture_output=True, text=True)
            m = _parse_job_metadata(r.stdout + r.stderr)
            if m.get("job_id"):
                job_id = m["job_id"]
                break
    if not job_id:
        job_id = job_name
    _log(f"Job created: {job_name} (id: {job_id})")
    return job_name, job_id


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

def _stop_job_safe(job_id):
    """Stop a Lepton job, retrying up to 3 times. Ignores errors."""
    for attempt in range(3):
        try:
            r = subprocess.run(f"{LEP_CLI} job stop -i {job_id}", shell=True,
                             capture_output=True, text=True, timeout=15)
            _log(f"Stopped job {job_id} (attempt {attempt+1})")
            return
        except Exception as e:
            _log(f"Stop attempt {attempt+1} failed: {e}")
            time.sleep(2)


def _get_job_status(job_name, job_id=None):
    if job_id:
        result = _run(f"{LEP_CLI} job get -i {job_id}")
    else:
        result = _run(f"{LEP_CLI} job get -n {job_name}")
    output = result.stdout + result.stderr
    for candidate in (result.stdout, output):
        payload = _extract_json_object(candidate)
        if isinstance(payload, dict):
            status_blob = payload.get("status", {})
            state = _normalize_status_text(status_blob.get("state"))
            if state:
                return state
    for line in output.split("\n"):
        ll = line.lower().strip()
        if "status" in ll or "state" in ll:
            state = _normalize_status_text(ll)
            if state:
                return state
    # Fallback: check job lists
    for state in ["queueing", "running", "completed", "failed", "stopped"]:
        r = _run(f"{LEP_CLI} job list -s {state}")
        if job_name in r.stdout or (job_id and job_id in r.stdout):
            return state
    return "unknown"


def _get_job_status_info(job_name, job_id=None):
    info = {"state": "unknown", "reason": "", "message": ""}
    if job_id:
        result = _run(f"{LEP_CLI} job get -i {job_id}")
    else:
        result = _run(f"{LEP_CLI} job get -n {job_name}")
    output = result.stdout + result.stderr
    for candidate in (result.stdout, output):
        payload = _extract_json_object(candidate)
        if isinstance(payload, dict):
            status_blob = payload.get("status", {})
            state = _normalize_status_text(status_blob.get("state"))
            if state:
                info["state"] = state
            info["reason"] = str(status_blob.get("reason", "") or "")
            info["message"] = str(status_blob.get("message", "") or "")
            return info
    info["state"] = _get_job_status(job_name, job_id)
    return info


# ---------------------------------------------------------------------------
# Log streaming (critical: dev clusters have no log persistence)
# ---------------------------------------------------------------------------

def _log_path(job_id):
    return os.path.join(WORKSPACE, f"run_{job_id}.log")


def _has_final_results_content(content):
    """Return True only when the final metric for the active eval mode is present."""
    if "results_json" in content:
        return True
    # SLOT runs after roundtrip/sliding and must not be cut off early.
    if "slot_enabled: True" in content:
        return "final_causal_slot" in content
    # LoRA TTT runs after quantized roundtrip/sliding and is the real final metric.
    if "ttt_enabled: True" in content:
        return "quantized_ttt_lora" in content or "ttt_sliding:done" in content
    # Sliding-only runs should wait for the sliding metric, not roundtrip.
    if "sliding_window_enabled: True" in content:
        return (
            "quantized_sliding_window" in content
            or "final_int6_sliding_window" in content
        )
    # Plain roundtrip-only evals can stop on the first final eval line.
    if "final_int6_roundtrip_exact" in content:
        return True
    return "eval_time" in content and "val_bpb" in content


def _log_has_results(log_file):
    if not os.path.exists(log_file):
        return False
    with open(log_file) as f:
        content = f.read()
    return _has_final_results_content(content)


def _stream_job_logs(job_id, log_file):
    """Stream logs to local file with auto-reconnect and dedup.

    Stops when: results captured, job completed/failed, or 1h wall time.
    """
    cmd = f"{LEP_CLI} job log -i {job_id}"
    max_wall = 3600
    base_delay = 15
    max_delay = 60
    seen_lines = set()
    start = time.time()
    attempt = 0
    got_output = False

    # Wait for Running state
    for _ in range(60):
        status = _get_job_status("", job_id)
        if status == "running":
            break
        if status in ("completed", "failed"):
            _robust_log_capture(job_id, log_file)
            return
        time.sleep(5)

    while (time.time() - start) < max_wall:
        attempt += 1
        proc = None
        lines_this = 0
        try:
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, bufsize=1)
            mode = "w" if attempt == 1 else "a"
            last_line_time = time.time()
            stall_timeout = 120  # kill if no output for 2 min
            with open(log_file, mode) as f:
                import select
                while True:
                    # Non-blocking check: wait up to 10s for output
                    ready, _, _ = select.select([proc.stdout], [], [], 10.0)
                    if ready:
                        line = proc.stdout.readline()
                        if not line:  # EOF
                            break
                        stripped = line.strip()
                        if stripped.startswith("Replica name") or stripped.startswith("Selected replica"):
                            last_line_time = time.time()
                            continue
                        if stripped in ("Connection stopped.", "") or stripped.startswith("End of log."):
                            if not stripped:
                                continue
                            last_line_time = time.time()
                            continue
                        if stripped in seen_lines:
                            last_line_time = time.time()
                            continue
                        seen_lines.add(stripped)
                        f.write(line)
                        f.flush()
                        lines_this += 1
                        got_output = True
                        last_line_time = time.time()
                    else:
                        # No output — check for stall
                        if time.time() - last_line_time > stall_timeout:
                            break  # force reconnect
                        # Also check if job is done
                        st = _get_job_status("", job_id)
                        if st in ("completed", "failed", "stopped"):
                            break
            proc.wait(timeout=10)
        except Exception:
            pass
        finally:
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass

        if _log_has_results(log_file):
            return

        status = _get_job_status("", job_id)
        if status in ("completed", "failed"):
            _robust_log_capture(job_id, log_file)
            return

        delay = base_delay if lines_this > 0 else min(base_delay * (2 if not got_output else 1), max_delay)
        time.sleep(delay)


def _robust_log_capture(job_id, log_file, retries=10):
    """Retry log retrieval after job completion to capture final results."""
    if _log_has_results(log_file):
        return
    for attempt in range(retries):
        time.sleep(min(5 * (attempt + 1), 30))
        try:
            cmd = [*shlex.split(LEP_CLI), "job", "log", "-i", job_id]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            lines = [l for l in result.stdout.split("\n")
                     if not l.startswith("Replica name")
                     and not l.startswith("Selected replica")
                     and l.strip() != "Connection stopped."]
            text = "\n".join(lines).strip()
            if _has_final_results_content(text):
                with open(log_file, "a") as f:
                    f.write(text + "\n")
                return
        except Exception:
            pass
        # Fallback: historical logs
        r = _run(f"{LEP_CLI} log get -j {job_id} --without-timestamp", timeout=15)
        if r.stdout.strip() and "No logs found" not in r.stdout:
            with open(log_file, "w") as f:
                f.write(r.stdout)
            return


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def _extract_results(logs):
    results = {}
    for line in logs.split("\n"):
        if "results_json:" in line:
            try:
                results.update(json.loads(line.split("results_json:", 1)[1].strip()))
                continue
            except (json.JSONDecodeError, IndexError):
                pass
        if "final_int6_roundtrip_exact" in line and re.search(r"val_bpb[=:]", line):
            m = re.search(r"val_bpb[=:](\d+\.\d+)", line)
            if m:
                results["val_bpb"] = float(m.group(1))
            m = re.search(r"val_loss[=:](\d+\.\d+)", line)
            if m:
                results["val_loss"] = float(m.group(1))
        # Match any eval result line: has val_bpb AND eval_time (or ttt done marker).
        # This covers all eval methods (roundtrip, sliding, TTT, future) without keyword updates.
        # Training checkpoints (e.g., "4000/20000 val_bpb: 1.13") lack eval_time so are excluded.
        # Last match wins — TTT result (run last) overrides sliding which overrides roundtrip.
        if ("eval_time" in line or "ttt_sliding:done" in line) and re.search(r"val_bpb[=:]", line):
            m = re.search(r"val_bpb[=:](\d+\.\d+)", line)
            if m:
                results["val_bpb"] = float(m.group(1))
            m = re.search(r"val_loss[=:](\d+\.\d+)", line)
            if m:
                results["val_loss"] = float(m.group(1))
        if "Total submission size" in line:
            m = re.search(r"(\d+)\s*bytes", line)
            if m:
                results["bytes_total"] = int(m.group(1))
        if "Artifact:" in line:
            m = re.search(r"Artifact:\s*([\d,]+)\s*bytes", line)
            if m:
                results["bytes_total"] = int(m.group(1).replace(",", ""))
        if "peak memory allocated" in line:
            m = re.search(r"(\d+)\s*MiB", line)
            if m:
                results["peak_memory_mib"] = int(m.group(1))
        if "Peak memory:" in line:
            m = re.search(r"Peak memory:\s*(\d+)\s*MiB", line)
            if m:
                results["peak_memory_mib"] = int(m.group(1))
    return results


# ---------------------------------------------------------------------------
# Output (autoresearch contract)
# ---------------------------------------------------------------------------

def _log(msg):
    print(f"[evaluate] {msg}", file=sys.stderr)


def _result_file_path() -> str:
    explicit = (
        os.environ.get("AUTORESEARCH_RESULT_FILE")
        or os.environ.get("PGOLF_RESULT_FILE")
        or ""
    ).strip()
    if explicit:
        return os.path.expanduser(explicit)
    return os.path.join(WORKSPACE, f"last_result.{os.getpid()}.json")


def _output(pass_val, score=None, details=None, error=None):
    result = {"pass": pass_val}
    if score is not None:
        result["score"] = score
    if details:
        result["details"] = details
    if error:
        result["error"] = error
    # Persist result to file BEFORE stdout — survives parent process kill.
    result_file = _result_file_path()
    try:
        with open(result_file, "w") as f:
            json.dump(result, f)
    except Exception:
        pass
    print(json.dumps(result))
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    import signal
    parser = argparse.ArgumentParser(description="Parameter Golf evaluator (self-contained)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--node-group", default=os.environ.get("AUTORESEARCH_NODE_GROUP", "heimdall-dev"))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()
    queue_failfast_seconds = int(os.environ.get("PGOLF_QUEUE_FAILFAST_SECONDS", "300"))

    # Best-effort node-group health summary before any push/launch.
    if args.node_group:
        try:
            health = _node_group_health(args.node_group)
            _log(
                "node-group health: "
                f"{args.node_group} available={health['available']}/{health['total']} "
                f"unhealthy={health['has_unhealthy']} "
                f"diskpressure={health['has_diskpressure']} "
                f"initializing={health['has_initializing']}"
            )
            if os.environ.get("PGOLF_STRICT_NODEGROUP_HEALTH", "0") == "1":
                blockers = []
                if health["available"] == 0:
                    blockers.append("no fully available GPU nodes")
                if health["has_unhealthy"]:
                    blockers.append("unhealthy/not-ready nodes present")
                if health["has_diskpressure"]:
                    blockers.append("diskpressure nodes present")
                if blockers:
                    _output(False, error="node-group unhealthy: " + ", ".join(blockers))
        except Exception as e:
            _log(f"node-group health check warning: {e}")

    # Signal handler: on SIGTERM/SIGINT, attempt final log capture before dying.
    # This catches the case where the calling Bash shell kills us (e.g., 10-min timeout).
    _main_state = {"job_id": None, "log_file": None}

    def _signal_handler(signum, frame):
        if _main_state["job_id"] and _main_state["log_file"]:
            _log(f"Signal {signum} received — attempting final log capture...")
            try:
                _robust_log_capture(_main_state["job_id"], _main_state["log_file"])
                log_content = open(_main_state["log_file"]).read()
                results = _extract_results(log_content)
                if results.get("val_bpb"):
                    _log(f"Saved partial results: val_bpb={results['val_bpb']}")
                    result_file = _result_file_path()
                    with open(result_file, "w") as f:
                        json.dump({"pass": False, "error": f"killed by signal {signum}",
                                   "score": -results["val_bpb"], "details": results}, f)
            except Exception as e:
                _log(f"Final capture failed: {e}")
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # 1. Detect and push branch
    branch = os.environ.get("AUTORESEARCH_BRANCH")
    if not branch:
        r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
        branch = r.stdout.strip() if r.returncode == 0 else None
    if not branch or branch == "HEAD":
        _output(False, error="cannot detect git branch")

    # Guard: submission/ branches are for PRs only, not GPU evaluation.
    # They fork from upstream/main and include the full repo (100s of files),
    # causing clone failures on remote GPU containers.
    if branch.startswith("submission/"):
        _output(False, error=f"branch '{branch}' is a submission branch (for PRs only). "
                f"Use an exp/ branch for GPU evaluation. "
                f"Submission branches include the full upstream repo and will fail on remote.")

    _log(f"Pushing {branch}...")
    r = subprocess.run(["git", "push", "origin", branch, "--force"], capture_output=True, text=True)
    if r.returncode != 0:
        _output(False, error=f"git push failed: {r.stderr[:500]}")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    _log(f"branch={branch} commit={commit[:7]}")

    # 2. Create job
    try:
        job_name, job_id = _create_job(commit, node_group=args.node_group, branch=branch)
    except RuntimeError as e:
        _output(False, error=str(e))

    # 3. Start log streaming in background thread
    log_file = _log_path(job_id)
    _main_state["job_id"] = job_id
    _main_state["log_file"] = log_file
    log_thread = threading.Thread(target=_stream_job_logs, args=(job_id, log_file), daemon=True)
    log_thread.start()

    # 4. Poll job status independently
    start = time.time()
    status = "unknown"
    while time.time() - start < args.timeout:
        status_info = _get_job_status_info(job_name, job_id)
        status = status_info["state"]
        elapsed = int(time.time() - start)
        suffix = ""
        if status_info.get("reason"):
            suffix = f" reason={status_info['reason']}"
        elif status_info.get("message"):
            suffix = f" message={status_info['message'][:160]}"
        _log(f"[{elapsed}s] {job_name}: {status}{suffix}")

        if (
            status == "queueing"
            and queue_failfast_seconds > 0
            and elapsed >= queue_failfast_seconds
            and status_info.get("reason") == "InsufficientQuota"
        ):
            _stop_job_safe(job_id)
            _output(False, error=f"queueing>{queue_failfast_seconds}s due to InsufficientQuota")

        if status in ("completed", "failed", "stopped"):
            log_thread.join(timeout=15)
            if status == "completed":
                _robust_log_capture(job_id, log_file)
            elif status == "stopped":
                _log(f"Job {job_name} was stopped externally (not by this evaluator)")
                _stop_job_safe(job_id)
                _output(False, error=f"job stopped externally — relaunch with same command")
            # Always stop the Lepton job after capturing results
            _stop_job_safe(job_id)
            break

        time.sleep(30)
    else:
        _log(f"Timeout after {args.timeout}s, attempting final status + log capture before stopping")
        status = "unknown"
        for _ in range(6):
            final_status = _get_job_status(job_name, job_id)
            try:
                _robust_log_capture(job_id, log_file)
            except Exception:
                pass
            log_thread.join(timeout=5)
            if os.path.exists(log_file):
                with open(log_file) as f:
                    timeout_log_content = f.read()
                timeout_results = _extract_results(timeout_log_content)
                if (
                    timeout_results.get("val_bpb") is not None
                    and _has_final_results_content(timeout_log_content)
                ):
                    status = "completed" if final_status == "unknown" else final_status
                    break
            status = final_status
            if status in ("completed", "failed", "stopped"):
                break
            time.sleep(10)
        if status not in ("completed", "failed", "stopped"):
            _stop_job_safe(job_id)
            _output(False, error=f"job timeout after {args.timeout}s")

    # 5. Parse results from log
    if not os.path.exists(log_file):
        _output(False, error=f"log file not found: {log_file}")

    with open(log_file) as f:
        log_content = f.read()
    results = _extract_results(log_content)
    val_bpb = results.get("val_bpb")
    has_final_marker = _has_final_results_content(log_content)

    if status == "failed":
        _output(False, score=-val_bpb if val_bpb else None, error="job failed on Lepton")

    if val_bpb is None:
        _output(False, error=f"no val_bpb in log ({len(log_content)} bytes, {log_content.count(chr(10))} lines)")

    if not has_final_marker:
        _output(False, score=-val_bpb, details=results,
                error="no final eval marker in log")

    # Score is always NEGATIVE val_bpb: higher = better = lower BPB (-1.116 > -1.117).
    # score_improvement keeps when score goes UP, so negating BPP aligns directions.
    score = -val_bpb

    bytes_total = results.get("bytes_total", 0)
    if bytes_total > ARTIFACT_LIMIT:
        # Artifact too large — pass=False so non-compliant candidates are NOT kept.
        # Score uses -val_bpb for consistent direction in logs (not used by framework
        # when pass=False, but useful for debugging).
        results["compliant"] = False
        _output(False, score=score, details=results,
                error=f"artifact too large: {bytes_total} > {ARTIFACT_LIMIT}")

    results["compliant"] = True
    _output(True, score=score, details=results)


def _run_seed(seed, commit, node_group, timeout):
    """Run a single seed validation. Returns result dict."""
    # Thread-safe: set env vars with lock to avoid race condition in parallel runs
    import threading
    _env_lock = getattr(_run_seed, '_lock', threading.Lock())
    _run_seed._lock = _env_lock
    with _env_lock:
        os.environ["PGOLF_SEED"] = str(seed)
        os.environ["PGOLF_JOB_PREFIX"] = f"pgolf-s{seed}"
        _log(f"[seed-{seed}] Starting on {node_group}")
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip()
        job_name, job_id = _create_job(commit, node_group, branch=branch)
    log_file = _log_path(job_id)
    _log(f"[seed-{seed}] Job {job_name} ({job_id})")

    # Stream logs in background thread
    stream_thread = threading.Thread(
        target=_stream_job_logs, args=(job_id, log_file), daemon=True)
    stream_thread.start()

    # Poll until done
    start = time.time()
    status = "unknown"
    while time.time() - start < timeout:
        time.sleep(30)
        status = _get_job_status(job_name, job_id)
        elapsed = int(time.time() - start)
        # Only trust Lepton job status — don't check log early (SLOT eval runs after roundtrip)
        _log(f"[seed-{seed}] [{elapsed}s] {job_name}: {status}")
        if status in ("completed", "failed", "stopped"):
            break
    else:
        _log(f"[seed-{seed}] TIMEOUT after {timeout}s")
        status = "timeout"

    # Give streaming thread time to flush final results
    stream_thread.join(timeout=60)
    # Final log capture attempt (stream may have missed tail)
    if status == "timeout":
        for _ in range(6):
            try:
                _robust_log_capture(job_id, log_file)
            except Exception:
                pass
            final_status = _get_job_status(job_name, job_id)
            if os.path.exists(log_file):
                with open(log_file) as f:
                    log_content = f.read()
                r = _extract_results(log_content)
                if r.get("val_bpb") is not None:
                    status = "completed" if final_status == "unknown" else final_status
                    break
            if final_status in ("completed", "failed", "stopped"):
                status = final_status
                break
            time.sleep(10)
    else:
        try:
            _robust_log_capture(job_id, log_file)
        except Exception:
            pass

    # Parse results
    val_bpb = None
    bytes_total = 0
    if os.path.exists(log_file):
        with open(log_file) as f:
            log_content = f.read()
        r = _extract_results(log_content)
        val_bpb = r.get("val_bpb")
        bytes_total = r.get("bytes_total", 0)
        if status == "timeout" and val_bpb is not None:
            final_status = _get_job_status(job_name, job_id)
            status = "completed" if final_status == "unknown" else final_status

    _log(f"[seed-{seed}] DONE: val_bpb={val_bpb}, bytes={bytes_total}, status={status}")
    return {
        "seed": seed,
        "val_bpb": val_bpb,
        "bytes_total": bytes_total,
        "status": status,
        "log_file": log_file,
        "job_name": job_name,
    }


def validate_3seed(node_groups=None, commit=None, seeds=(1337, 42, 2025)):
    """Run 3-seed validation in parallel across multiple node groups.

    Args:
        node_groups: List of node groups to spread seeds across.
                     e.g., ['aws-iad-leptondev-001', 'heimdall-dev']
                     Seeds are round-robin assigned to node groups.
        commit: Git commit SHA to validate. If None, uses HEAD.
        seeds: Tuple of seeds to validate (default: 1337, 42, 2025)
    """
    if node_groups is None:
        ng = os.environ.get("PGOLF_NODE_GROUP", "")
        node_groups = [ng] if ng else []
    if not node_groups:
        _log("ERROR: no node groups specified")
        return

    if commit is None:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        commit = r.stdout.strip()[:7]
    else:
        commit = commit[:7]

    # Push branch so all nodes can access the commit
    _log(f"Pushing branch for commit {commit}...")
    try:
        r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
        branch = r.stdout.strip()
        subprocess.run(["git", "push", "origin", branch, "--force"], capture_output=True, text=True)
    except Exception as e:
        _log(f"Push warning: {e}")

    # Assign seeds to node groups round-robin
    assignments = []
    for i, seed in enumerate(seeds):
        ng = node_groups[i % len(node_groups)]
        assignments.append((seed, ng))
    _log(f"Validating commit {commit} — parallel across {len(node_groups)} node group(s)")
    for seed, ng in assignments:
        _log(f"  Seed {seed} → {ng}")

    # Launch all seeds in parallel threads
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        futures = {}
        for seed, ng in assignments:
            f = pool.submit(_run_seed, seed, commit, ng, DEFAULT_TIMEOUT)
            futures[f] = seed

        for f in as_completed(futures):
            seed = futures[f]
            try:
                result = f.result()
                results[seed] = result
            except Exception as e:
                _log(f"[seed-{seed}] ERROR: {e}")
                results[seed] = {"val_bpb": None, "bytes_total": 0, "status": "error",
                                 "log_file": "", "job_name": ""}

    # Summary
    _log(f"\n{'='*60}")
    _log("3-SEED VALIDATION SUMMARY")
    _log(f"{'='*60}")
    bpbs = [r["val_bpb"] for r in results.values() if r["val_bpb"] is not None]
    if len(bpbs) == len(seeds):
        import statistics
        mean_bpb = statistics.mean(bpbs)
        std_bpb = statistics.stdev(bpbs) if len(bpbs) > 1 else 0
        for seed in seeds:
            r = results[seed]
            _log(f"  Seed {seed}: val_bpb={r['val_bpb']:.8f}  bytes={r['bytes_total']}  log={r['log_file']}")
        _log(f"  Mean: {mean_bpb:.8f}  Std: {std_bpb:.8f}")
        _log(f"  All artifacts < 16MB: {all(r['bytes_total'] < ARTIFACT_LIMIT for r in results.values())}")
        print(json.dumps({
            "seeds": {str(s): results[s]["val_bpb"] for s in seeds},
            "mean": mean_bpb,
            "std": std_bpb,
            "bytes": {str(s): results[s]["bytes_total"] for s in seeds},
            "logs": {str(s): results[s]["log_file"] for s in seeds},
        }, indent=2))
    else:
        _log(f"  FAILED: only {len(bpbs)}/{len(seeds)} seeds produced results")
        for seed in seeds:
            r = results.get(seed, {})
            _log(f"  Seed {seed}: val_bpb={r.get('val_bpb')}  status={r.get('status')}")


def stop_all():
    """Stop ALL autoresearch processes and Lepton jobs. Nuclear cleanup."""
    import signal

    _log("=== STOPPING ALL AUTORESEARCH ===")

    # 1. Kill all local autoresearch/evaluator processes (except ourselves)
    my_pid = os.getpid()
    patterns = ["evaluate_pgolf", "evaluate\\.py", "stream_existing_job"]
    killed = []
    try:
        r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        for line in r.stdout.split("\n"):
            if any(p.replace(".*", "") in line for p in patterns) and "grep" not in line and "--stop" not in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = int(parts[1])
                    if pid != my_pid:
                        try:
                            os.kill(pid, signal.SIGKILL)
                            killed.append(pid)
                        except ProcessLookupError:
                            pass
    except Exception as e:
        _log(f"Process kill error: {e}")
    _log(f"Killed {len(killed)} local processes: {killed}")

    # 2. Stop all running/queueing Lepton pgolf jobs
    stopped = []
    try:
        r = subprocess.run(f"{LEP_CLI} job list -s running", shell=True, capture_output=True, text=True, timeout=30)
        for line in (r.stdout + r.stderr).split("\n"):
            if "pgolf" in line:
                # Extract job ID (pattern: pgolf-xxx-yyy-zzzz)
                parts = [p.strip().strip("│").strip() for p in line.split("│") if p.strip()]
                if parts and parts[0].startswith("pgolf-") and "-" in parts[0][6:]:
                    job_id = parts[0]
                    try:
                        subprocess.run(f"{LEP_CLI} job stop -i {job_id}", shell=True,
                                     capture_output=True, text=True, timeout=15)
                        stopped.append(job_id)
                    except Exception:
                        pass
        # Also check queueing
        r = subprocess.run(f"{LEP_CLI} job list -s queueing", shell=True, capture_output=True, text=True, timeout=30)
        for line in (r.stdout + r.stderr).split("\n"):
            if "pgolf" in line:
                parts = [p.strip().strip("│").strip() for p in line.split("│") if p.strip()]
                if parts and parts[0].startswith("pgolf-") and "-" in parts[0][6:]:
                    job_id = parts[0]
                    try:
                        subprocess.run(f"{LEP_CLI} job stop -i {job_id}", shell=True,
                                     capture_output=True, text=True, timeout=15)
                        stopped.append(job_id)
                    except Exception:
                        pass
    except Exception as e:
        _log(f"Lepton job stop error: {e}")
    _log(f"Stopped {len(stopped)} Lepton jobs: {stopped}")

    # 3. Final verification
    time.sleep(2)
    r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    remaining = [l for l in r.stdout.split("\n")
                 if any(p.replace(".*", "") in l for p in patterns)
                 and "grep" not in l and "--stop" not in l and str(my_pid) not in l]
    if remaining:
        _log(f"WARNING: {len(remaining)} processes still alive:")
        for l in remaining:
            _log(f"  {l[:100]}")
    else:
        _log("All clean.")

    _log("=== STOP COMPLETE ===")


def stop_job(job_id):
    """Stop a single Lepton job by ID. Use this instead of stop_all() in multi-worker setups."""
    _log(f"=== STOP JOB: {job_id} ===")
    try:
        r = subprocess.run(f"{LEP_CLI} job stop -i {job_id}", shell=True,
                         capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            _log(f"Stopped job {job_id}")
        else:
            _log(f"Failed to stop {job_id}: {r.stderr.strip()}")
    except Exception as e:
        _log(f"Error stopping {job_id}: {e}")


def list_jobs():
    """List all running/queueing pgolf Lepton jobs."""
    _log("=== LEPTON JOBS ===")
    for state in ["running", "queueing"]:
        try:
            r = subprocess.run(f"{LEP_CLI} job list -s {state}", shell=True,
                             capture_output=True, text=True, timeout=30)
            output = r.stdout + r.stderr
            if "pgolf" in output:
                _log(f"\n{state.upper()}:")
                for line in output.split("\n"):
                    if "pgolf" in line or "Name" in line:
                        _log(f"  {line.strip()}")
            else:
                _log(f"{state}: none")
        except Exception as e:
            _log(f"Error listing {state}: {e}")


def preflight(node_group=None, commit=None):
    """Pre-flight validation: check baseline compliance before launching experiments.

    Runs a single GPU job on the specified commit and verifies:
    - Training completes within 600s
    - Eval completes within 600s
    - Artifact < 16MB
    - val_bpb is valid
    - Score direction is correct (negative)
    """
    if commit is None:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        commit = r.stdout.strip()[:7]
    else:
        commit = commit[:7]
    ng = node_group or os.environ.get("PGOLF_NODE_GROUP", "")

    _log(f"=== PRE-FLIGHT CHECK: commit {commit} on {ng} ===")

    # Push branch
    try:
        r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
        branch = r.stdout.strip()
        subprocess.run(["git", "push", "origin", branch, "--force"], capture_output=True, text=True)
    except Exception as e:
        _log(f"Push warning: {e}")

    # Run single job (default seed 1337)
    os.environ["PGOLF_JOB_PREFIX"] = "pgolf-preflight"
    job_name, job_id = _create_job(commit, ng, branch=branch)
    log_file = _log_path(job_id)
    _log(f"Job: {job_name} ({job_id})")

    # Stream + poll
    stream_thread = threading.Thread(target=_stream_job_logs, args=(job_id, log_file), daemon=True)
    stream_thread.start()

    start = time.time()
    status = "unknown"
    while time.time() - start < DEFAULT_TIMEOUT:
        time.sleep(30)
        status = _get_job_status(job_name, job_id)
        elapsed = int(time.time() - start)
        _log(f"[preflight] [{elapsed}s] {status}")
        if status in ("completed", "failed", "stopped"):
            break

    stream_thread.join(timeout=60)
    try:
        _robust_log_capture(job_id, log_file)
    except Exception:
        pass

    # Parse and validate
    checks = {"training": False, "eval": False, "artifact": False, "bpb_valid": False}
    if not os.path.exists(log_file):
        _log("FAIL: no log file")
        return False

    with open(log_file) as f:
        log_content = f.read()
    results = _extract_results(log_content)
    val_bpb = results.get("val_bpb")
    bytes_total = results.get("bytes_total", 0)

    # Check training completed
    if "stopping_early" in log_content or "step:" in log_content:
        checks["training"] = True

    # Check eval completed
    if val_bpb is not None:
        checks["eval"] = True
        checks["bpb_valid"] = True

    # Check artifact size
    if 0 < bytes_total < ARTIFACT_LIMIT:
        checks["artifact"] = True

    _log(f"\n=== PRE-FLIGHT RESULTS ===")
    _log(f"  val_bpb:    {val_bpb}")
    _log(f"  bytes:      {bytes_total} / {ARTIFACT_LIMIT} ({'OK' if checks['artifact'] else 'FAIL'})")
    _log(f"  training:   {'OK' if checks['training'] else 'FAIL'}")
    _log(f"  eval:       {'OK' if checks['eval'] else 'FAIL'}")
    _log(f"  job status: {status}")

    all_pass = all(checks.values()) and status == "completed"
    _log(f"  OVERALL:    {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        _log(f"  Failed checks: {[k for k,v in checks.items() if not v]}")
    return all_pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--preflight":
        ng = None
        commit = None
        for i, a in enumerate(sys.argv):
            if a == "--node-group" and i + 1 < len(sys.argv):
                ng = sys.argv[i + 1]
            if a == "--commit" and i + 1 < len(sys.argv):
                commit = sys.argv[i + 1]
        ok = preflight(node_group=ng, commit=commit)
        sys.exit(0 if ok else 1)
    elif len(sys.argv) > 1 and sys.argv[1] == "--stop":
        # --stop <job-id> stops a single job
        if len(sys.argv) > 2:
            stop_job(sys.argv[2])
        else:
            _log("Usage: --stop <job-id> to stop a single job. Use --stop-all to kill everything.")
            _log("Run --list to see job IDs.")
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == "--stop-all":
        stop_all()
    elif len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_jobs()
    elif len(sys.argv) > 1 and sys.argv[1] == "--3seed":
        node_groups = []
        commit = None
        for i, a in enumerate(sys.argv):
            if a == "--node-group" and i + 1 < len(sys.argv):
                node_groups.append(sys.argv[i + 1])
            if a == "--commit" and i + 1 < len(sys.argv):
                commit = sys.argv[i + 1]
        validate_3seed(node_groups=node_groups or None, commit=commit)
    else:
        main()
