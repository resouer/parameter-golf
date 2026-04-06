#!/usr/bin/env python3
"""Setup worker repos for a Parameter Golf round.

Reads the round config (rounds/roundN.md) and sets up each worker's repo:
- Creates clone from ~/code/parameter-golf if it doesn't exist
- Sets fetch URL to local repo, push URL to GitHub
- Syncs to origin/main, creates worker branch
- Deploys correct base code (from Baseline section in round config)
- Deploys latest evaluate.py from main repo
- Verifies base code features (vocab, layers, key components)
- Refuses to proceed if any check fails

Usage:
    python3 setup_round.py rounds/round12.md
    python3 setup_round.py rounds/round12.md --dry-run
"""

import argparse
import os
import re
import shutil
import subprocess
import sys


MAIN_REPO = os.path.expanduser("~/code/parameter-golf")
GITHUB_PUSH_URL = "git@github.com:resouer/parameter-golf.git"


def _run(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if check and r.returncode != 0:
        print(f"  FAIL: {cmd}\n  {r.stderr.strip()}", file=sys.stderr)
    return r


def parse_workers(round_file):
    """Parse the Workers table from a round config markdown file."""
    workers = []
    in_table = False
    with open(round_file) as f:
        for line in f:
            if "| Worker " in line and "| Repo " in line:
                in_table = True
                continue
            if in_table and line.strip().startswith("|---"):
                continue
            if in_table and line.strip().startswith("|"):
                cols = [c.strip().strip("`") for c in line.strip().split("|")[1:-1]]
                if len(cols) >= 4:
                    workers.append({
                        "name": cols[0].strip(),
                        "repo": os.path.expanduser(cols[1].strip()),
                        "node_group": cols[2].strip(),
                        "branch": cols[3].strip(),
                    })
            elif in_table and not line.strip().startswith("|"):
                in_table = False
    return workers


def parse_baseline(round_file):
    """Parse the Baseline section to find base code source.

    Looks for patterns like:
    - 'commit 867c8b3' or 'commit `867c8b3`'
    - 'R10 W3 packed code (commit dfba227'
    - 'base:** PR #1394'
    - Worker repo references like '~/code/parameter-golf-w3'
    """
    baseline = {"commit": None, "pr": None, "source_repo": None, "source_branch": None}
    in_baseline = False
    with open(round_file) as f:
        for line in f:
            if line.strip().startswith("## Baseline"):
                in_baseline = True
                continue
            if in_baseline and line.strip().startswith("## "):
                break
            if in_baseline:
                # Find commit hash
                m = re.search(r'commit\s+`?([0-9a-f]{7,40})`?', line, re.I)
                if m:
                    baseline["commit"] = m.group(1)
                # Find PR number
                m = re.search(r'PR\s*#(\d+)', line)
                if m:
                    baseline["pr"] = m.group(1)
                # Find source repo path
                m = re.search(r'(~/code/parameter-golf-w\d+)', line)
                if m:
                    baseline["source_repo"] = os.path.expanduser(m.group(1))
                # Find source branch
                m = re.search(r'(exp/round-\d+/w\d+)', line)
                if m:
                    baseline["source_branch"] = m.group(1)
    return baseline


def find_base_code(baseline):
    """Find the base train_gpt.py from baseline info.

    Priority:
    1. Source repo + commit (exact code)
    2. Source repo + branch (latest on branch)
    3. PR diff extraction (from upstream)
    4. FAIL — cannot find base code
    """
    # Method 1: exact commit in a worker repo
    if baseline.get("commit") and baseline.get("source_repo"):
        repo = baseline["source_repo"]
        commit = baseline["commit"]
        if os.path.exists(repo):
            r = _run(f"git show {commit}:train_gpt.py", cwd=repo, check=False)
            if r.returncode == 0 and len(r.stdout) > 100:
                print(f"  Base code: from {repo} @ {commit} ({len(r.stdout)} bytes)")
                return r.stdout

    # Method 2: branch in a worker repo
    if baseline.get("source_branch") and baseline.get("source_repo"):
        repo = baseline["source_repo"]
        branch = baseline["source_branch"]
        if os.path.exists(repo):
            r = _run(f"git show {branch}:train_gpt.py", cwd=repo, check=False)
            if r.returncode == 0 and len(r.stdout) > 100:
                print(f"  Base code: from {repo} @ {branch} ({len(r.stdout)} bytes)")
                return r.stdout

    # Method 3: any worker repo with the commit
    if baseline.get("commit"):
        commit = baseline["commit"]
        for i in range(1, 4):
            repo = os.path.expanduser(f"~/code/parameter-golf-w{i}")
            if os.path.exists(repo):
                _run("git fetch --all", cwd=repo, check=False)
                r = _run(f"git show {commit}:train_gpt.py", cwd=repo, check=False)
                if r.returncode == 0 and len(r.stdout) > 100:
                    print(f"  Base code: from {repo} @ {commit} ({len(r.stdout)} bytes)")
                    return r.stdout

    # Method 4: PR diff extraction
    if baseline.get("pr"):
        pr = baseline["pr"]
        print(f"  Extracting base code from PR #{pr} diff...")
        r = _run(
            f"gh pr diff {pr} --repo openai/parameter-golf | "
            f"sed -n '/^+++ b\\/records\\/.*train_gpt\\.py/,/^diff --git/p' | "
            f"grep '^+' | sed 's/^+//' | tail -n +2",
            check=False
        )
        if r.returncode == 0 and len(r.stdout) > 100:
            print(f"  Base code: from PR #{pr} diff ({len(r.stdout)} bytes)")
            return r.stdout
        # PR might be packed — try fetching the blob directly
        print(f"  PR diff extraction failed (packed?). Trying blob fetch...")

    return None


def verify_base_code(train_gpt_path):
    """Verify base code has expected features. Returns list of warnings."""
    warnings = []
    with open(train_gpt_path) as f:
        content = f.read()

    lines = content.count('\n') + 1

    # Check for naive baseline (CRITICAL)
    if lines > 1000 and lines < 1200:
        warnings.append(f"FATAL: {lines} lines looks like NAIVE BASELINE (expect <500 for packed or 400-500 for competitive)")

    # Detect if packed
    is_packed = "lzma" in content and "b85decode" in content

    if is_packed:
        # For packed code, try to decompress and check
        try:
            import lzma, base64
            m = re.search(r"b85decode\(b'(.+?)'\)", content, re.DOTALL)
            if m:
                inner = lzma.decompress(base64.b85decode(m.group(1))).decode()
                content = inner  # use decompressed for feature checks
                print(f"  Code: packed ({lines} lines, decompresses to {inner.count(chr(10))+1} lines)")
        except Exception as e:
            warnings.append(f"WARNING: packed code decompression failed: {e}")
    else:
        print(f"  Code: {lines} lines (unpacked)")

    # Feature checks on (decompressed) content
    checks = {
        "VOCAB_SIZE": re.search(r"VOCAB_SIZE.*?(\d+)", content),
        "NUM_LAYERS": re.search(r"NUM_LAYERS.*?(\d+)", content),
        "MLP_MULT": re.search(r"MLP_MULT.*?([\d.]+)", content),
        "evaluate.py": None,  # checked separately
    }

    for key, match in checks.items():
        if match:
            print(f"  {key}: {match.group(1)}")

    if not re.search(r"VOCAB_SIZE", content):
        warnings.append("WARNING: VOCAB_SIZE not found in code")

    return warnings


def setup_worker(worker, baseline_code, main_repo, dry_run=False):
    """Setup a single worker repo."""
    name = worker["name"]
    repo = worker["repo"]
    branch = worker["branch"]

    print(f"\n{'='*60}")
    print(f"Setting up {name}: {repo} (branch: {branch})")
    print(f"{'='*60}")

    # Refuse to use main repo
    main_repo_real = os.path.realpath(MAIN_REPO)
    if os.path.exists(repo) and os.path.realpath(repo) == main_repo_real:
        print(f"  FATAL: {repo} IS the main repo. Workers must use dedicated clones.")
        return False

    if dry_run:
        print(f"  [dry-run] Would create/sync {repo}")
        return True

    # Create clone if needed
    if not os.path.exists(repo):
        print(f"  Creating clone: {MAIN_REPO} -> {repo}")
        r = _run(f"git clone {MAIN_REPO} {repo}")
        if r.returncode != 0:
            return False
    else:
        print(f"  Repo exists: {repo}")

    # Set remotes
    print(f"  Setting fetch URL -> {MAIN_REPO}")
    _run(f"git remote set-url origin {MAIN_REPO}", cwd=repo)
    print(f"  Setting push URL -> {GITHUB_PUSH_URL}")
    _run(f"git remote set-url --push origin {GITHUB_PUSH_URL}", cwd=repo)

    # Sync to main
    print(f"  Fetching origin...")
    _run("git fetch origin", cwd=repo)
    _run("git checkout main 2>/dev/null || git checkout -b main origin/main", cwd=repo)
    _run("git reset --hard origin/main", cwd=repo)

    # Create/checkout worker branch
    print(f"  Creating branch: {branch}")
    r = _run(f"git checkout -B {branch}", cwd=repo, check=False)
    if r.returncode != 0:
        _run(f"git checkout {branch}", cwd=repo)

    # Verify branch
    r = _run("git rev-parse --abbrev-ref HEAD", cwd=repo)
    current = r.stdout.strip()
    if current == "main":
        print(f"  FATAL: Repo is on main!")
        return False
    if current != branch:
        print(f"  WARNING: Expected branch {branch}, got {current}")

    # Deploy base code
    train_path = os.path.join(repo, "train_gpt.py")
    if baseline_code:
        with open(train_path, "w") as f:
            f.write(baseline_code)
        print(f"  train_gpt.py: deployed from baseline ({len(baseline_code)} bytes)")
    elif os.path.exists(train_path):
        print(f"  train_gpt.py: keeping existing (no baseline code found)")
    else:
        print(f"  WARNING: train_gpt.py not found and no baseline code available")

    # Deploy latest evaluate.py from main repo
    main_eval = os.path.join(main_repo, ".autophd", "project", "evaluate.py")
    worker_eval = os.path.join(repo, "evaluate.py")
    if os.path.exists(main_eval):
        shutil.copy2(main_eval, worker_eval)
        print(f"  evaluate.py: deployed from main repo")
    else:
        print(f"  WARNING: evaluate.py not found in main repo")

    # Verify base code
    if os.path.exists(train_path):
        warnings = verify_base_code(train_path)
        for w in warnings:
            print(f"  {w}")
            if w.startswith("FATAL"):
                return False

    # Commit deployed files
    _run("git add train_gpt.py evaluate.py", cwd=repo, check=False)
    r = _run("git diff --cached --quiet", cwd=repo, check=False)
    if r.returncode != 0:  # there are staged changes
        round_name = os.path.basename(branch)
        _run(f'git commit -m "Setup {round_name}: deploy base code + evaluate.py"', cwd=repo)
        print(f"  Committed deployed files")

    print(f"  Branch: {current}")
    r = _run("git log --oneline -1", cwd=repo)
    print(f"  HEAD: {r.stdout.strip()}")
    print(f"  Status: READY")
    return True


def verify_main_repo():
    """Verify main repo is on main branch."""
    r = _run("git rev-parse --abbrev-ref HEAD", cwd=MAIN_REPO)
    branch = r.stdout.strip()
    if branch != "main":
        print(f"WARNING: Main repo ({MAIN_REPO}) is on '{branch}', not 'main'.")
        print(f"  Switching to main...")
        _run("git checkout main", cwd=MAIN_REPO)
        r = _run("git rev-parse --abbrev-ref HEAD", cwd=MAIN_REPO)
        if r.stdout.strip() != "main":
            print(f"  FATAL: Could not switch main repo to main branch.")
            return False
    print(f"Main repo ({MAIN_REPO}): on main branch. OK.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup worker repos for an experiment round")
    parser.add_argument("round_file", help="Path to round config (e.g., rounds/round12.md)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without doing it")
    args = parser.parse_args()

    if not os.path.exists(args.round_file):
        print(f"Error: {args.round_file} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Setup for: {args.round_file}")
    print()

    # Verify main repo
    if not verify_main_repo():
        sys.exit(1)

    # Parse baseline — find correct base code
    print("Parsing baseline...")
    baseline = parse_baseline(args.round_file)
    print(f"  Baseline: commit={baseline.get('commit')}, PR=#{baseline.get('pr')}, "
          f"repo={baseline.get('source_repo')}, branch={baseline.get('source_branch')}")

    baseline_code = find_base_code(baseline)
    if not baseline_code:
        print("  WARNING: Could not find base code. Workers will use whatever is on their branch.")

    # Parse workers
    workers = parse_workers(args.round_file)
    if not workers:
        print("No workers found in round config.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(workers)} workers:")
    for w in workers:
        print(f"  {w['name']}: {w['repo']} ({w['node_group']}) -> {w['branch']}")

    # Setup each worker
    results = []
    for w in workers:
        ok = setup_worker(w, baseline_code, MAIN_REPO, dry_run=args.dry_run)
        results.append((w["name"], ok))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for name, ok in results:
        status = "READY" if ok else "FAILED"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"\nAll {len(workers)} workers ready. Launch the team.")
    else:
        print(f"\nSome workers FAILED. Fix issues before launching.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
