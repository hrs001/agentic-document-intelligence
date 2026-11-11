#!/usr/bin/env python3
"""Publish local git history to GitHub via Git Data REST API (no git push)."""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OWNER = "hrs001"
REPO = "agentic-document-intelligence"
BRANCH = "main"
API = "https://api.github.com"
REPO_ROOT = Path(__file__).resolve().parent

# Paths never uploaded when mirroring working tree (git replay uses commit trees only).
EXCLUDE_PATH_PARTS = {
    ".git",
    ".env",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".aws-sam",
    ".next",
    ".npm-cache",
}


@dataclass
class ApiError(Exception):
    status: int
    body: str
    method: str
    url: str

    def __str__(self) -> str:
        return (
            f"{self.method} {self.url} -> HTTP {self.status}: "
            f"{self.body[:2000]}"
        )


class GitHubClient:
    def __init__(self, token: str) -> None:
        self.token = token

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        tolerate: set[int] | None = None,
    ) -> tuple[int, Any]:
        url = f"{API}{path}"
        data = None
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "adi-publish-via-api",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return resp.status, None
                return resp.status, json.loads(raw)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if tolerate and e.code in tolerate:
                return e.code, json.loads(body) if body else None
            raise ApiError(e.code, body, method, url) from e


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_commit(sha: str) -> dict[str, str]:
    raw = run_git("cat-file", "-p", sha)
    info: dict[str, str] = {"sha": sha, "parents": ""}
    lines = raw.splitlines()
    msg_start = 0
    for i, line in enumerate(lines):
        if line.startswith("tree "):
            info["tree"] = line.split()[1]
        elif line.startswith("parent "):
            info["parents"] = (info.get("parents", "") + " " + line.split()[1]).strip()
        elif line.startswith("author "):
            info["author"] = line[len("author ") :]
        elif line.startswith("committer "):
            info["committer"] = line[len("committer ") :]
        elif line == "":
            msg_start = i + 1
            break
    info["message"] = "\n".join(lines[msg_start:])
    return info


def git_ls_tree(tree_sha: str) -> list[tuple[str, str, str, str]]:
    """Return (mode, type, object_sha, name) for one tree level."""
    out = run_git("ls-tree", tree_sha)
    rows: list[tuple[str, str, str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        mode, obj_type, obj_sha, name = line.split(maxsplit=3)
        rows.append((mode, obj_type, obj_sha, name))
    return rows


def flatten_tree(local_tree_sha: str, prefix: str = "") -> list[tuple[str, str, str, str]]:
    items: list[tuple[str, str, str, str]] = []
    for mode, obj_type, obj_sha, name in git_ls_tree(local_tree_sha):
        path = f"{prefix}/{name}" if prefix else name
        if any(part in EXCLUDE_PATH_PARTS for part in Path(path).parts):
            continue
        if obj_type == "blob":
            items.append((mode, obj_type, obj_sha, path))
        elif obj_type == "tree":
            items.extend(flatten_tree(obj_sha, path))
    return items


def blob_bytes(sha: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "cat-file", "blob", sha])


def author_line_to_api(line: str) -> dict[str, str]:
    # "Name <email> timestamp tz"
    m = re.match(r"^(.*) <([^>]+)> (\d+) ([+-]\d{4})$", line.strip())
    if not m:
        return {"name": "unknown", "email": "unknown@users.noreply.github.com"}
    name, email, when, tz = m.groups()
    return {
        "name": name,
        "email": email,
        "date": f"{when} {tz}",
    }


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print(
            "Set GITHUB_TOKEN (hrs001 PAT with repo scope) in your environment, then re-run.",
            file=sys.stderr,
        )
        return 2

    client = GitHubClient(token)

    status, user = client.request("GET", "/user")
    login = user.get("login", "")
    print(f"Authenticated as: {login}")
    if login != OWNER:
        print(
            f"WARNING: token user is {login!r}, expected {OWNER!r}. Aborting.",
            file=sys.stderr,
        )
        return 3

    status, repo = client.request("GET", f"/repos/{OWNER}/{REPO}")
    default_branch = repo.get("default_branch") or BRANCH
    print(f"Destination: {OWNER}/{REPO} default_branch={default_branch!r}")

    code, ref = client.request(
        "GET",
        f"/repos/{OWNER}/{REPO}/git/ref/heads/{default_branch}",
        tolerate={404, 409},
    )
    if code == 200 and ref and ref.get("object", {}).get("sha"):
        remote_sha = ref["object"]["sha"]
        print(
            f"Remote {default_branch} already at {remote_sha[:12]} — refusing to overwrite.",
            file=sys.stderr,
        )
        return 4
    print(f"Remote {default_branch} has no commits (safe to publish).")

    local_commits = run_git("rev-list", "--reverse", "main").splitlines()
    print(f"Replaying {len(local_commits)} local commits from main…")

    blob_map: dict[str, str] = {}
    tree_map: dict[str, str] = {}
    github_commit_map: dict[str, str] = {}
    uploaded_blobs = 0

    for idx, commit_sha in enumerate(local_commits, 1):
        meta = parse_commit(commit_sha)
        local_tree = meta["tree"]

        if local_tree in tree_map:
            github_tree = tree_map[local_tree]
        else:
            tree_entries = []
            for mode, _obj_type, blob_sha, path in flatten_tree(local_tree):
                if blob_sha not in blob_map:
                    content = blob_bytes(blob_sha)
                    if len(content) > 100 * 1024 * 1024:
                        raise RuntimeError(f"Blob too large for GitHub API: {path} ({len(content)} bytes)")
                    payload = {
                        "content": base64.b64encode(content).decode("ascii"),
                        "encoding": "base64",
                    }
                    _s, blob_resp = client.request(
                        "POST",
                        f"/repos/{OWNER}/{REPO}/git/blobs",
                        payload,
                    )
                    blob_map[blob_sha] = blob_resp["sha"]
                    uploaded_blobs += 1
                tree_entries.append(
                    {
                        "path": path,
                        "mode": mode,
                        "type": "blob",
                        "sha": blob_map[blob_sha],
                    }
                )
            _s, tree_resp = client.request(
                "POST",
                f"/repos/{OWNER}/{REPO}/git/trees",
                {"tree": tree_entries},
            )
            github_tree = tree_resp["sha"]
            tree_map[local_tree] = github_tree

        parents = []
        for p in meta.get("parents", "").split():
            if p in github_commit_map:
                parents.append(github_commit_map[p])

        commit_payload: dict[str, Any] = {
            "message": meta["message"],
            "tree": github_tree,
            "parents": parents,
        }
        if meta.get("author"):
            commit_payload["author"] = author_line_to_api(meta["author"])
        if meta.get("committer"):
            commit_payload["committer"] = author_line_to_api(meta["committer"])

        _s, commit_resp = client.request(
            "POST",
            f"/repos/{OWNER}/{REPO}/git/commits",
            commit_payload,
        )
        github_commit_map[commit_sha] = commit_resp["sha"]
        local_tip = run_git("rev-parse", commit_sha).strip()[:12]
        remote_tip = commit_resp["sha"][:12]
        print(
            f"  [{idx}/{len(local_commits)}] {local_tip} -> {remote_tip} "
            f"({meta['message'].splitlines()[0][:60]})"
        )

    tip = github_commit_map[local_commits[-1]]
    if code in (404, 409):
        client.request(
            "POST",
            f"/repos/{OWNER}/{REPO}/git/refs",
            {"ref": f"refs/heads/{default_branch}", "sha": tip},
        )
        print(f"Created refs/heads/{default_branch} -> {tip[:12]}")
    else:
        print("Unexpected ref state; not updating.", file=sys.stderr)
        return 5

    # Verification
    _s, remote_commit = client.request("GET", f"/repos/{OWNER}/{REPO}/commits/{tip}")
    local_head_msg = parse_commit(local_commits[-1])["message"]
    remote_msg = remote_commit["commit"]["message"]
    if local_head_msg != remote_msg:
        print("VERIFY FAIL: commit message mismatch", file=sys.stderr)
        return 6

    local_files = set(run_git("ls-tree", "-r", "--name-only", "HEAD").splitlines())
    remote_tree_sha = remote_commit["commit"]["tree"]["sha"]
    _s, remote_tree = client.request(
        "GET",
        f"/repos/{OWNER}/{REPO}/git/trees/{remote_tree_sha}?recursive=1",
    )
    remote_files = {
        e["path"]
        for e in remote_tree.get("tree", [])
        if e.get("type") == "blob"
    }
    missing = local_files - remote_files
    extra = remote_files - local_files
    if missing or extra:
        print(f"VERIFY FAIL: file set mismatch missing={len(missing)} extra={len(extra)}", file=sys.stderr)
        if missing:
            print("  missing sample:", sorted(missing)[:10], file=sys.stderr)
        if extra:
            print("  extra sample:", sorted(extra)[:10], file=sys.stderr)
        return 7

    print(
        f"VERIFY OK: {len(local_files)} files, tip {tip}, "
        f"uploaded_blobs={uploaded_blobs}, unique_trees={len(tree_map)}"
    )
    print(f"Repository: https://github.com/{OWNER}/{REPO}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ApiError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
