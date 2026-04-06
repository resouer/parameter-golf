#!/usr/bin/env python3
"""Setup worker repos for a Parameter Golf round.

Reads the round config (rounds/roundN.md) and sets up each worker's repo:
- Creates clone from ~/code/parameter-golf if it doesn't exist
- Sets fetch URL to local repo, push URL to GitHub
- Syncs to origin/main, creates worker branch
- Verifies evaluator exists and branch is correct
- Refuses to proceed if any repo is on main

Usage:
    python3 setup_round.py rounds/round3.md
    python3 setup_round.py rounds/round3.md --dry-run
"""

import argparse
import os
import re
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


def setup_worker(worker, dry_run=False):
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
        print(f"  FATAL: {repo} IS the main repo ({MAIN_REPO}). Workers must use dedicated clones.")
        print(f"  Fix: Change the repo path in the round config to a separate directory.")
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
        print(f"  FATAL: Repo is on main! Something went wrong.")
        return False
    if current != branch:
        print(f"  WARNING: Expected branch {branch}, got {current}")

    # Always deploy latest evaluate.py from main repo
    main_eval = os.path.join(main_repo, ".autophd", "project", "evaluate.py")
    worker_eval = os.path.join(repo, "evaluate.py")
    if os.path.exists(main_eval):
        import shutil
        shutil.copy2(main_eval, worker_eval)
        print(f"  evaluate.py: deployed from main repo")
    else:
        print(f"  WARNING: .autophd/project/evaluate.py not found in main repo")

    # Verify train_gpt.py exists
    train_path = os.path.join(repo, "train_gpt.py")
    if os.path.exists(train_path):
        print(f"  train_gpt.py: OK")
    else:
        print(f"  WARNING: train_gpt.py not found in {repo}")

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
    parser.add_argument("round_file", help="Path to round config (e.g., rounds/round3.md)")
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
        ok = setup_worker(w, dry_run=args.dry_run)
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
