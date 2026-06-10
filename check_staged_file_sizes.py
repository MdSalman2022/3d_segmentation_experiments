from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


BYTES_PER_MIB = 1024 * 1024


def run_git(repo_root: str, *args: str) -> str:
    safe_directory = Path(repo_root).resolve().as_posix()
    result = subprocess.run(
        ["git", "-c", f"safe.directory={safe_directory}", *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return result.stdout


def staged_paths(repo_root: str) -> list[str]:
    output = run_git(repo_root, "diff", "--cached", "--name-only", "-z", "--diff-filter=AM", "--")
    return [path for path in output.split("\0") if path]


def staged_size(repo_root: str, path: str) -> int:
    return int(run_git(repo_root, "cat-file", "-s", f":{path}").strip())


def format_size(num_bytes: int) -> str:
    return f"{num_bytes / BYTES_PER_MIB:.2f} MiB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Block commits when staged files exceed the configured size limit.",
    )
    parser.add_argument("--limit-mb", type=int, default=30, help="Maximum allowed staged file size in MiB.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to the Git repository root.",
    )
    args = parser.parse_args()

    repo_root = str(Path(args.repo_root).resolve())
    limit_bytes = args.limit_mb * BYTES_PER_MIB
    offenders: list[tuple[str, int]] = []

    for path in staged_paths(repo_root):
        size = staged_size(repo_root, path)
        if size > limit_bytes:
            offenders.append((path, size))

    if not offenders:
        return 0

    print(f"Commit blocked: staged files larger than {args.limit_mb} MiB are not allowed.", file=sys.stderr)
    print("", file=sys.stderr)
    for path, size in offenders:
        print(f"  {format_size(size):>10}  {path}", file=sys.stderr)

    print("", file=sys.stderr)
    print("Unstage or remove these files before committing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
