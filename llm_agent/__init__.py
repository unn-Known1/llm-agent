#!/usr/bin/env python3
"""
File Agent v1.1.0 - Autonomous file-system agent using LLM API
Supports persistent named profiles (api key, base url, models).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from difflib import unified_diff
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)
VERSION = "1.0.0"


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIG_DIR  = Path.home() / ".config" / "file_agent"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history.txt"

# ---------------------------------------------------------------------------
# Config / Profile Manager
# ---------------------------------------------------------------------------

DEFAULT_PROFILE: dict = {
    "api_key":      "",
    "base_url":     "https://api.openai.com/v1",
    "models":       [],
    "active_model": "",
    "max_iter":     100,
    "read_only":    False,
    "dry_run":      False,
}

_EMPTY_CONFIG: dict = {
    "active_profile":   "",
    "last_sandbox_root": "",
    "profiles":         {},
}


class ConfigManager:
    """Persistent named profiles stored in ~/.config/file_agent/config.json."""

    def __init__(self) -> None:
        self._data: dict = {**_EMPTY_CONFIG}
        self.load()

    # ---- persistence ----

    def load(self) -> None:
        if not CONFIG_FILE.exists():
            return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self._data.update(loaded)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not load config: %s", e)

    def save(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            log.warning("Could not save config: %s", e)

    # ---- global props ----

    @property
    def active_profile_name(self) -> str:
        return self._data.get("active_profile", "")

    @active_profile_name.setter
    def active_profile_name(self, name: str) -> None:
        self._data["active_profile"] = name

    @property
    def last_sandbox_root(self) -> str:
        return self._data.get("last_sandbox_root", "")

    @last_sandbox_root.setter
    def last_sandbox_root(self, path: str) -> None:
        self._data["last_sandbox_root"] = path
        self.save()

    def has_profiles(self) -> bool:
        return bool(self._data.get("profiles"))

    def profile_names(self) -> list[str]:
        return list(self._data.get("profiles", {}).keys())

    # ---- profile CRUD ----

    def get_profile(self, name: Optional[str] = None) -> dict:
        name = name or self.active_profile_name
        raw = self._data.get("profiles", {}).get(name, {})
        return {**DEFAULT_PROFILE, **raw}

    def active_profile(self) -> dict:
        return self.get_profile(self.active_profile_name)

    def save_profile(self, name: str, profile: dict) -> None:
        self._data.setdefault("profiles", {})[name] = {**DEFAULT_PROFILE, **profile}
        self.save()

    def delete_profile(self, name: str) -> bool:
        profiles = self._data.get("profiles", {})
        if name not in profiles:
            return False
        del profiles[name]
        if self.active_profile_name == name:
            self._data["active_profile"] = next(iter(profiles), "")
        self.save()
        return True

    def use_profile(self, name: str) -> bool:
        if name not in self._data.get("profiles", {}):
            return False
        self._data["active_profile"] = name
        self.save()
        return True

    # ---- convenience mutators on active profile ----

    def update_active(self, **kwargs) -> None:
        name = self.active_profile_name
        profile = self._data.setdefault("profiles", {}).setdefault(name, {**DEFAULT_PROFILE})
        profile.update(kwargs)
        self.save()

    def add_model(self, model: str) -> None:
        p = self.active_profile()
        models: list = p.get("models", [])
        if model not in models:
            models.append(model)
        p["models"] = models
        if not p.get("active_model"):
            p["active_model"] = model
        self.save_profile(self.active_profile_name, p)

    def remove_model(self, model: str) -> bool:
        p = self.active_profile()
        models: list = p.get("models", [])
        if model not in models:
            return False
        models.remove(model)
        p["models"] = models
        if p.get("active_model") == model:
            p["active_model"] = models[0] if models else ""
        self.save_profile(self.active_profile_name, p)
        return True

    def set_active_model(self, model: str) -> None:
        p = self.active_profile()
        models: list = p.get("models", [])
        if model not in models:
            models.append(model)
        p["models"] = models
        p["active_model"] = model
        self.save_profile(self.active_profile_name, p)


# ---------------------------------------------------------------------------
# First-time setup wizard
# ---------------------------------------------------------------------------

PRESET_URLS: dict[str, tuple[str, str]] = {
    "1": ("OpenAI",       "https://api.openai.com/v1"),
    "2": ("Groq",         "https://api.groq.com/openai/v1"),
    "3": ("Together AI",  "https://api.together.xyz/v1"),
    "4": ("Ollama (local)", "http://localhost:11434/v1"),
    "5": ("Custom",       ""),
}


def _prompt_inline(label: str, default: str = "", secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    sys.stderr.write(f"  {_cyan(label)}{_grey(suffix)}: ")
    sys.stderr.flush()
    try:
        val = input().strip()
    except EOFError:
        val = ""
    return val if val else default


def run_wizard(cfg: ConfigManager) -> bool:
    """Interactive first-time setup. Returns True on success."""
    print(_bold(_cyan("\n  ╔══ File Agent Setup ══╗")), file=sys.stderr)
    print(_grey("  No profiles found. Let's create your first one.\n"), file=sys.stderr)

    name = _prompt_inline("Profile name", "default")

    print("\n  Select API provider:", file=sys.stderr)
    for k, (label, url) in PRESET_URLS.items():
        url_hint = f"  {_grey(url)}" if url else ""
        print(f"    {k}. {label}{url_hint}", file=sys.stderr)

    choice = _prompt_inline("Choice", "1")
    if choice in PRESET_URLS and PRESET_URLS[choice][1]:
        base_url = PRESET_URLS[choice][1]
        print(f"  → {_grey(base_url)}", file=sys.stderr)
    else:
        base_url = _prompt_inline("Base URL", "https://api.openai.com/v1")

    api_key = _prompt_inline("API key")
    if not api_key:
        print(_yellow("  ⚠  No API key — set later with /set key <value>"), file=sys.stderr)

    model = _prompt_inline("Default model ID", "gpt-4o")

    profile = {
        **DEFAULT_PROFILE,
        "api_key":      api_key,
        "base_url":     base_url,
        "models":       [model] if model else [],
        "active_model": model,
    }
    cfg.save_profile(name, profile)
    cfg.active_profile_name = name
    cfg.save()
    print(_green(f"\n  ✓ Profile '{name}' saved to {CONFIG_FILE}\n"), file=sys.stderr)
    return True


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

class SandboxError(Exception):
    pass


class Sandbox:
    MAX_DIR_ENTRIES   = 10_000
    MAX_READ_BYTES    = 10_000_000
    MAX_WRITE_BYTES   = 10_000_000
    DEFAULT_READ_LINES = 400
    SKIP_DIRS = {
        ".git", ".svn", ".hg", "__pycache__", ".pytest_cache",
        "node_modules", ".venv", "venv", ".tox", ".mypy_cache",
        ".ruff_cache", ".coverage", ".htmlcov",
    }
    BINARY_EXTS = {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
        ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",
        ".exe", ".dll", ".so", ".dylib", ".a", ".o", ".obj",
        ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv",
        ".wav", ".flac", ".aac", ".ogg",
        ".ttf", ".otf", ".woff", ".woff2", ".eot",
        ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".db", ".sqlite", ".mdb",
        ".pyc", ".pyo", ".pyd",
    }
    DANGEROUS_ROOTS = {Path("/"), Path("/home"), Path("/Users"), Path("/tmp")}

    def __init__(self, root: str | Path, *,
                 read_only: bool = False, dry_run: bool = False,
                 allow_dangerous: bool = False) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise SandboxError(f"root is not a directory: {self.root}")
        if not allow_dangerous and self.root in self.DANGEROUS_ROOTS:
            raise SandboxError(f"refusing to use {self.root} as sandbox root")
        self.read_only = read_only
        self.dry_run   = dry_run
        self._writes: list[dict] = []

    def _resolve(self, path: str, must_exist: bool = False) -> Path:
        p = (self.root / path.lstrip("/")).resolve()
        if not str(p).startswith(str(self.root)):
            raise SandboxError(f"path outside sandbox: {path}")
        if must_exist and not p.exists():
            raise SandboxError(f"does not exist: {path}")
        return p

    def _check_write(self) -> None:
        if self.read_only:
            raise SandboxError("write tools disabled (read-only mode)")

    def _audit(self, op: str, path: str, **extra) -> None:
        self._writes.append({"op": op, "path": path, "ts": time.time(), **extra})

    def writes_audit(self) -> list[dict]:
        return list(self._writes)

    # ---- read tools ----

    def list_dir(self, path: str = ".", *, show_hidden: bool = False) -> dict:
        full = self._resolve(path, must_exist=True)
        if not full.is_dir():
            raise SandboxError(f"not a directory: {path}")
        entries = []
        truncated = False
        try:
            children = sorted(full.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as e:
            raise SandboxError(f"cannot list: {e}")
        for child in children:
            if not show_hidden and child.name.startswith("."):
                continue
            if child.is_dir() and child.name in self.SKIP_DIRS:
                continue
            try:
                stat = child.stat()
                entries.append({
                    "name": child.name,
                    "type": "dir" if child.is_dir() else "file",
                    "size": stat.st_size if child.is_file() else None,
                })
            except OSError:
                continue
            if len(entries) >= self.MAX_DIR_ENTRIES:
                truncated = True
                break
        rel = full.relative_to(self.root).as_posix() or "."
        return {"path": rel, "entries": entries, "count": len(entries), "truncated": truncated}

    def read_file(self, path: str, start_line: int = 1, end_line: Optional[int] = None) -> dict:
        full = self._resolve(path, must_exist=True)
        if not full.is_file():
            raise SandboxError(f"not a file: {path}")
        if full.suffix.lower() in self.BINARY_EXTS:
            raise SandboxError(f"binary file refused: {path}")
        size = full.stat().st_size
        if size > self.MAX_READ_BYTES:
            raise SandboxError(f"file too large: {size:,} bytes. Use line ranges or search.")
        try:
            start_line = max(1, int(start_line))
            end_line_arg = int(end_line) if end_line is not None else start_line + self.DEFAULT_READ_LINES - 1
            end_line_arg = max(start_line, end_line_arg)
        except (TypeError, ValueError):
            raise SandboxError("start_line/end_line must be integers")
        try:
            with open(full, "rb") as fb:
                if b"\x00" in fb.read(8192):
                    raise SandboxError(f"binary file refused (NUL bytes): {path}")
        except OSError as e:
            raise SandboxError(f"cannot open file: {e}")
        slice_lines: list[str] = []
        total = 0
        try:
            with open(full, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    total = i
                    if start_line <= i <= end_line_arg:
                        slice_lines.append(line.rstrip("\n"))
                    elif i > end_line_arg:
                        total += sum(1 for _ in fh)
                        break
        except UnicodeDecodeError:
            raise SandboxError(f"file is not valid UTF-8: {path}")
        except OSError as e:
            raise SandboxError(f"cannot read file: {e}")
        end_eff = min(total, end_line_arg)
        return {
            "path": path, "start_line": start_line, "end_line": end_eff,
            "total_lines": total, "content": "\n".join(slice_lines),
            "truncated": end_eff < total,
        }

    def glob(self, pattern: str, max_results: int = 200) -> dict:
        if not isinstance(pattern, str) or not pattern:
            raise SandboxError("pattern must be a non-empty string")
        max_results = max(1, min(int(max_results) if str(max_results).isdigit() else 200, 1000))
        match_path = ("/" in pattern) or ("**" in pattern)
        match_name = not match_path or pattern.startswith("**/")
        results: list[str] = []
        root_str = str(self.root)
        for dirpath, dirnames, filenames in os.walk(root_str):
            dirnames[:] = [d for d in dirnames if d not in self.SKIP_DIRS]
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                rel  = os.path.relpath(full, root_str).replace(os.sep, "/")
                if (match_path and fnmatch.fnmatch(rel, pattern)) or \
                   (match_name and fnmatch.fnmatch(fname, pattern)):
                    results.append(rel)
                    if len(results) >= max_results:
                        return {"pattern": pattern, "matches": results, "count": len(results), "truncated": True}
        return {"pattern": pattern, "matches": results, "count": len(results), "truncated": False}

    def search(self, pattern: str, path: str = ".", max_results: int = 100) -> dict:
        if not isinstance(pattern, str) or not pattern:
            raise SandboxError("pattern must be a non-empty string")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise SandboxError(f"invalid regex: {e}")
        max_results = max(1, min(int(max_results) if str(max_results).isdigit() else 100, 500))
        base = self._resolve(path, must_exist=True)
        hits: list[dict] = []
        root_str   = str(self.root)
        binary_exts = self.BINARY_EXTS
        max_bytes   = self.MAX_READ_BYTES

        def _scan(full: str, rel: str) -> bool:
            try:
                if os.path.getsize(full) > max_bytes:
                    return False
                with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            hits.append({"path": rel, "line": i, "text": line.rstrip("\n")[:200]})
                            if len(hits) >= max_results:
                                return True
            except OSError:
                pass
            return False

        if base.is_file():
            rel = os.path.relpath(str(base), root_str).replace(os.sep, "/")
            if base.suffix.lower() not in binary_exts:
                _scan(str(base), rel)
        else:
            for dirpath, dirnames, filenames in os.walk(str(base)):
                dirnames[:] = [d for d in dirnames if d not in self.SKIP_DIRS]
                for fname in filenames:
                    if fname.rfind(".") != -1 and fname[fname.rfind("."):].lower() in binary_exts:
                        continue
                    full = os.path.join(dirpath, fname)
                    rel  = os.path.relpath(full, root_str).replace(os.sep, "/")
                    if _scan(full, rel):
                        return {"pattern": pattern, "hits": hits, "count": len(hits), "truncated": True}
        return {"pattern": pattern, "hits": hits, "count": len(hits), "truncated": len(hits) >= max_results}

    def file_info(self, path: str) -> dict:
        full = self._resolve(path, must_exist=True)
        try:
            st = full.stat()
        except OSError as e:
            raise SandboxError(f"cannot stat: {e}")
        info: dict = {
            "path": path,
            "type": "dir" if full.is_dir() else ("symlink" if full.is_symlink() else "file"),
            "size": st.st_size if full.is_file() else None,
            "mtime": st.st_mtime, "ctime": st.st_ctime,
            "mode": oct(st.st_mode & 0o777),
            "is_binary": full.suffix.lower() in self.BINARY_EXTS if full.is_file() else False,
        }
        if full.is_file() and not info["is_binary"] and st.st_size <= self.MAX_READ_BYTES:
            try:
                with open(full, "rb") as fh:
                    info["line_count"] = sum(1 for _ in fh)
            except OSError:
                info["line_count"] = None
        return info

    def tree(self, path: str = ".", max_depth: int = 3,
             show_hidden: bool = False, max_entries: int = 500) -> dict:
        max_depth   = max(1, min(int(max_depth)   if str(max_depth).isdigit()   else 3,   10))
        max_entries = max(1, min(int(max_entries)  if str(max_entries).isdigit() else 500, 2000))
        base = self._resolve(path, must_exist=True)
        if not base.is_dir():
            raise SandboxError(f"not a directory: {path}")
        count     = [0]
        truncated = [False]

        def _walk(node: Path, depth: int) -> dict:
            entry: dict = {"name": node.name or ".", "type": "dir", "children": []}
            if depth >= max_depth:
                entry["truncated"] = True
                return entry
            try:
                children = sorted(node.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                return entry
            for child in children:
                if not show_hidden and child.name.startswith("."):
                    continue
                if child.is_dir() and child.name in self.SKIP_DIRS:
                    continue
                if count[0] >= max_entries:
                    truncated[0] = True
                    break
                count[0] += 1
                if child.is_dir():
                    entry["children"].append(_walk(child, depth + 1))
                else:
                    try:
                        size = child.stat().st_size
                    except OSError:
                        size = None
                    entry["children"].append({"name": child.name, "type": "file", "size": size})
            return entry

        rel = base.relative_to(self.root).as_posix() or "."
        root_entry = _walk(base, 0)
        root_entry["name"] = rel
        return {"path": rel, "max_depth": max_depth, "count": count[0],
                "truncated": truncated[0], "tree": root_entry}

    def head(self, path: str, lines: int = 20) -> dict:
        lines = max(1, min(int(lines) if str(lines).isdigit() else 20, 1000))
        full  = self._resolve(path, must_exist=True)
        if not full.is_file():
            raise SandboxError(f"not a file: {path}")
        if full.suffix.lower() in self.BINARY_EXTS:
            raise SandboxError(f"binary file refused: {path}")
        out: list[str] = []
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if i > lines:
                        break
                    out.append(line.rstrip("\n"))
        except OSError as e:
            raise SandboxError(f"cannot read file: {e}")
        return {"path": path, "lines": len(out), "content": "\n".join(out)}

    def tail(self, path: str, lines: int = 20) -> dict:
        lines = max(1, min(int(lines) if str(lines).isdigit() else 20, 1000))
        full  = self._resolve(path, must_exist=True)
        if not full.is_file():
            raise SandboxError(f"not a file: {path}")
        if full.suffix.lower() in self.BINARY_EXTS:
            raise SandboxError(f"binary file refused: {path}")
        buf: deque = deque(maxlen=lines)
        total = 0
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    total = i
                    buf.append(line.rstrip("\n"))
        except OSError as e:
            raise SandboxError(f"cannot read file: {e}")
        return {"path": path, "lines": len(buf), "total_lines": total, "content": "\n".join(buf)}

    def hash_file(self, path: str, algo: str = "sha256") -> dict:
        algo = (algo or "sha256").lower()
        if algo not in ("md5", "sha1", "sha256", "sha512"):
            raise SandboxError(f"unsupported algo: {algo}")
        full = self._resolve(path, must_exist=True)
        if not full.is_file():
            raise SandboxError(f"not a file: {path}")
        h    = hashlib.new(algo)
        size = 0
        try:
            with open(full, "rb") as fh:
                while True:
                    chunk = fh.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
                    size += len(chunk)
        except OSError as e:
            raise SandboxError(f"cannot read file: {e}")
        return {"path": path, "algo": algo, "hash": h.hexdigest(), "bytes": size}

    def word_count(self, path: str) -> dict:
        full = self._resolve(path, must_exist=True)
        if not full.is_file():
            raise SandboxError(f"not a file: {path}")
        if full.suffix.lower() in self.BINARY_EXTS:
            raise SandboxError(f"binary file refused: {path}")
        lines = words = chars = nbytes = 0
        try:
            with open(full, "rb") as fh:
                for raw in fh:
                    nbytes += len(raw)
                    lines  += 1
                    decoded = raw.decode("utf-8", errors="replace")
                    chars  += len(decoded)
                    words  += len(decoded.split())
        except OSError as e:
            raise SandboxError(f"cannot read file: {e}")
        return {"path": path, "lines": lines, "words": words, "chars": chars, "bytes": nbytes}

    def diff_files(self, path_a: str, path_b: str, context: int = 3) -> dict:
        context = max(0, min(int(context) if str(context).isdigit() else 3, 10))
        full_a  = self._resolve(path_a, must_exist=True)
        full_b  = self._resolve(path_b, must_exist=True)
        if not full_a.is_file() or not full_b.is_file():
            raise SandboxError("both paths must be files")
        for p, raw in ((full_a, path_a), (full_b, path_b)):
            if p.suffix.lower() in self.BINARY_EXTS:
                raise SandboxError(f"binary file refused: {raw}")
            if p.stat().st_size > self.MAX_READ_BYTES:
                raise SandboxError(f"file too large for diff: {raw}")
        try:
            a_lines = full_a.read_text(encoding="utf-8").splitlines(keepends=True)
            b_lines = full_b.read_text(encoding="utf-8").splitlines(keepends=True)
        except UnicodeDecodeError as e:
            raise SandboxError(f"file is not valid UTF-8: {e}")
        diff = list(unified_diff(a_lines, b_lines, fromfile=path_a, tofile=path_b, n=context))
        text = "".join(diff)
        if len(text) > self.MAX_READ_BYTES:
            text = text[:self.MAX_READ_BYTES] + "\n[...diff truncated]"
        return {
            "path_a": path_a, "path_b": path_b,
            "identical": not diff, "diff": text,
            "hunks": sum(1 for ln in diff if ln.startswith("@@")),
        }

    def grep(self, pattern: str, path: str = ".", max_results: int = 100) -> dict:
        if not isinstance(pattern, str) or not pattern:
            raise SandboxError("pattern must be a non-empty string")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise SandboxError(f"invalid regex: {e}")
        try:
            max_results = max(1, min(int(max_results) if str(max_results).isdigit() else 100, 500))
        except (TypeError, ValueError):
            max_results = 100
        base = self._resolve(path, must_exist=True)
        matches: list[str] = []
        root_str = str(self.root)
        for dirpath, dirnames, filenames in os.walk(str(base)):
            dirnames[:] = [d for d in dirnames if d not in self.SKIP_DIRS]
            for fname in filenames:
                if fname.rfind(".") != -1 and fname[fname.rfind("."):].lower() in self.BINARY_EXTS:
                    continue
                full = os.path.join(dirpath, fname)
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                    if rx.search(content):
                        rel = os.path.relpath(full, root_str).replace(os.sep, "/")
                        matches.append(rel)
                        if len(matches) >= max_results:
                            return {"pattern": pattern, "matches": matches, "count": len(matches), "truncated": True}
                except OSError:
                    continue
        return {"pattern": pattern, "matches": matches, "count": len(matches), "truncated": False}

    def count_matches(self, path: str, pattern: str) -> dict:
        if not isinstance(pattern, str) or not pattern:
            raise SandboxError("pattern must be a non-empty string")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise SandboxError(f"invalid regex: {e}")
        full = self._resolve(path, must_exist=True)
        if not full.is_file():
            raise SandboxError(f"not a file: {path}")
        if full.suffix.lower() in self.BINARY_EXTS:
            raise SandboxError(f"binary file refused: {path}")
        count = 0
        lines_with_match = 0
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    matches = rx.findall(line)
                    if matches:
                        count += len(matches)
                        lines_with_match += 1
        except OSError as e:
            raise SandboxError(f"cannot read file: {e}")
        return {"path": path, "pattern": pattern, "count": count, "lines_with_match": lines_with_match}

    def file_permissions(self, path: str, mode: str = "") -> dict:
        full = self._resolve(path, must_exist=True)
        current_mode = oct(full.stat().st_mode & 0o777)
        if not mode:
            return {"path": path, "mode": current_mode, "changed": False}
        self._check_write()
        try:
            if mode.startswith("0"):
                new_mode = int(mode, 8)
            if mode.startswith("-"):
                change = mode[1:]
                who_map = {"u": 0o700, "g": 0o070, "o": 0o007, "a": 0o777}
                mask = 0o777
                for c in change:
                    if c in "rwx":
                        continue
                    if c in "augo" and len(change) > 1:
                        mask = who_map.get(c, 0o777)
                        change = change.replace(c, "", 1)
                if not change:
                    new_mode = None
                else:
                    cur = full.stat().st_mode & 0o777
                    for c in change:
                        if c == "+":
                            continue
                        perms = {"r": 4, "w": 2, "x": 1}
                        bit = perms.get(c)
                        if bit:
                            new_mode = (cur | bit) & mask
                        else:
                            new_mode = None
                            break
            else:
                new_mode = int(mode, 8)
            if new_mode is not None:
                os.chmod(full, new_mode)
                return {"path": path, "mode": oct(full.stat().st_mode & 0o777), "changed": True}
        except (ValueError, OSError) as e:
            raise SandboxError(f"cannot change permissions: {e}")
        return {"path": path, "mode": current_mode, "changed": False}

    def file_type(self, path: str) -> dict:
        full = self._resolve(path, must_exist=True)
        suffix = full.suffix.lower()
        mime_types = {
            ".py": "text/x-python", ".js": "text/javascript", ".ts": "text/typescript",
            ".json": "application/json", ".xml": "application/xml", ".html": "text/html",
            ".css": "text/css", ".md": "text/markdown", ".txt": "text/plain",
            ".sh": "application/x-sh", ".bash": "application/x-sh",
            ".c": "text/x-c", ".cpp": "text/x-c++", ".h": "text/x-c",
            ".java": "text/x-java", ".go": "text/x-go", ".rs": "text/x-rust",
            ".rb": "text/x-ruby", ".php": "text/x-php",
            ".sql": "application/x-sql", ".yaml": "application/x-yaml",
            ".yml": "application/x-yaml", ".toml": "application/x-toml",
            ".png": "image/png", ".jpg": "image/jpeg", ".gif": "image/gif",
            ".pdf": "application/pdf", ".zip": "application/zip",
        }
        mime = mime_types.get(suffix, "application/octet-stream")
        is_binary = full.suffix.lower() in self.BINARY_EXTS
        return {"path": path, "suffix": suffix, "mime": mime, "is_binary": is_binary}

    def run_command(self, cmd: str, cwd: str = ".", timeout: int = 30) -> dict:
        self._check_write()
        import subprocess
        import platform
        try:
            timeout = max(1, min(int(timeout), 300))
        except (TypeError, ValueError):
            timeout = 30
        full_cwd = self._resolve(cwd, must_exist=True) if cwd != "." else self.root
        system = platform.system()
        if system == "Windows":
            shell_cmd = ["cmd", "/c", cmd]
        else:
            shell_cmd = ["/bin/sh", "-c", cmd]
        try:
            result = subprocess.run(
                shell_cmd, cwd=str(full_cwd),
                capture_output=True, text=True, timeout=timeout
            )
            return {
                "cmd": cmd, "cwd": str(full_cwd),
                "returncode": result.returncode,
                "stdout": result.stdout[:5000],
                "stderr": result.stderr[:5000],
                "ok": result.returncode == 0,
                "platform": system,
            }
        except subprocess.TimeoutExpired:
            return {"cmd": cmd, "cwd": str(full_cwd), "returncode": -1,
                    "stdout": "", "stderr": f"Command timed out after {timeout}s", "ok": False}
        except OSError as e:
            raise SandboxError(f"command failed: {e}")

    # ---- write tools ----

    def write_file(self, path: str, content: str, overwrite: bool = False) -> dict:
        self._check_write()
        if not isinstance(content, str):
            raise SandboxError("content must be a string")
        full    = self._resolve(path)
        existed = full.exists()
        if existed and not overwrite:
            raise SandboxError(f"file exists, set overwrite:true to replace: {path}")
        if existed and full.is_dir():
            raise SandboxError(f"refusing to overwrite directory: {path}")
        nbytes = len(content.encode("utf-8"))
        if nbytes > self.MAX_WRITE_BYTES:
            raise SandboxError(f"content exceeds {self.MAX_WRITE_BYTES:,}-byte limit ({nbytes:,} bytes)")
        if self.dry_run:
            self._audit("write_file", path, bytes=nbytes, overwrote=existed, dry_run=True)
            return {"path": path, "bytes": nbytes, "created": not existed, "dry_run": True}
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        self._audit("write_file", path, bytes=nbytes, overwrote=existed)
        return {"path": path, "bytes": nbytes, "created": not existed}

    def edit_file(self, path: str, old_string: str, new_string: str) -> dict:
        self._check_write()
        if not isinstance(old_string, str) or not isinstance(new_string, str):
            raise SandboxError("old_string and new_string must be strings")
        if not old_string:
            raise SandboxError("old_string cannot be empty")
        if old_string == new_string:
            raise SandboxError("old_string and new_string are identical")
        full = self._resolve(path, must_exist=True)
        if not full.is_file():
            raise SandboxError(f"not a file: {path}")
        try:
            text = full.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise SandboxError(f"file is not valid UTF-8: {path}")
        count = text.count(old_string)
        if count == 0:
            stripped = old_string.strip()
            hint = " (hint: matched without leading/trailing whitespace)" \
                   if stripped and stripped != old_string and stripped in text else ""
            raise SandboxError(f"old_string not found in file{hint}")
        if count > 1:
            raise SandboxError(f"old_string appears {count} times — make it unique")
        new_text  = text.replace(old_string, new_string, 1)
        new_bytes = len(new_text.encode("utf-8"))
        if new_bytes > self.MAX_WRITE_BYTES:
            raise SandboxError(f"resulting file exceeds {self.MAX_WRITE_BYTES:,}-byte limit")
        if self.dry_run:
            self._audit("edit_file", path, old_len=len(old_string), new_len=len(new_string), dry_run=True)
            return {"path": path, "old_len": len(old_string), "new_len": len(new_string),
                    "delta_bytes": new_bytes - len(text.encode()), "dry_run": True}
        full.write_text(new_text, encoding="utf-8")
        self._audit("edit_file", path, old_len=len(old_string), new_len=len(new_string))
        return {"path": path, "old_len": len(old_string), "new_len": len(new_string),
                "delta_bytes": new_bytes - len(text.encode())}

    def append_file(self, path: str, content: str, add_newline: bool = False) -> dict:
        self._check_write()
        if not isinstance(content, str):
            raise SandboxError("content must be a string")
        full    = self._resolve(path)
        if full.exists() and full.is_dir():
            raise SandboxError(f"path is a directory: {path}")
        existed = full.exists()
        prior   = full.stat().st_size if existed else 0
        payload = content + ("\n" if add_newline and not content.endswith("\n") else "")
        nbytes  = len(payload.encode("utf-8"))
        if prior + nbytes > self.MAX_WRITE_BYTES:
            raise SandboxError(f"resulting file would exceed {self.MAX_WRITE_BYTES:,}-byte limit")
        if self.dry_run:
            self._audit("append_file", path, bytes=nbytes, created=not existed, dry_run=True)
            return {"path": path, "appended_bytes": nbytes, "created": not existed, "dry_run": True}
        full.parent.mkdir(parents=True, exist_ok=True)
        with open(full, "a", encoding="utf-8") as fh:
            fh.write(payload)
        self._audit("append_file", path, bytes=nbytes, created=not existed)
        return {"path": path, "appended_bytes": nbytes, "created": not existed}

    def delete_file(self, path: str, confirm: bool = False) -> dict:
        self._check_write()
        if not confirm:
            raise SandboxError("delete requires confirm:true")
        full = self._resolve(path, must_exist=True)
        if full.is_dir():
            raise SandboxError(f"refusing to delete directory: {path}")
        if self.dry_run:
            self._audit("delete_file", path, dry_run=True)
            return {"path": path, "deleted": True, "dry_run": True}
        full.unlink()
        self._audit("delete_file", path)
        return {"path": path, "deleted": True}

    def move_file(self, src: str, dst: str, overwrite: bool = False) -> dict:
        self._check_write()
        src_full = self._resolve(src, must_exist=True)
        dst_full = self._resolve(dst)
        if src_full.is_dir():
            raise SandboxError(f"refusing to move directory: {src}")
        if dst_full.exists():
            if dst_full.is_dir():
                raise SandboxError(f"destination is a directory: {dst}")
            if not overwrite:
                raise SandboxError(f"destination exists, set overwrite:true to replace: {dst}")
        if self.dry_run:
            self._audit("move_file", src, dst=dst, dry_run=True)
            return {"src": src, "dst": dst, "moved": True, "dry_run": True}
        dst_full.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src_full, dst_full)
        self._audit("move_file", src, dst=dst)
        return {"src": src, "dst": dst, "moved": True}

    def copy_file(self, src: str, dst: str, overwrite: bool = False) -> dict:
        self._check_write()
        src_full = self._resolve(src, must_exist=True)
        dst_full = self._resolve(dst)
        if src_full.is_dir():
            raise SandboxError(f"refusing to copy directory: {src}")
        if dst_full.exists():
            if dst_full.is_dir():
                raise SandboxError(f"destination is a directory: {dst}")
            if not overwrite:
                raise SandboxError(f"destination exists, set overwrite:true to replace: {dst}")
        size = src_full.stat().st_size
        if size > self.MAX_WRITE_BYTES:
            raise SandboxError(f"source exceeds {self.MAX_WRITE_BYTES:,}-byte copy limit")
        if self.dry_run:
            self._audit("copy_file", src, dst=dst, bytes=size, dry_run=True)
            return {"src": src, "dst": dst, "bytes": size, "copied": True, "dry_run": True}
        dst_full.parent.mkdir(parents=True, exist_ok=True)
        with open(src_full, "rb") as fin, open(dst_full, "wb") as fout:
            while True:
                chunk = fin.read(65536)
                if not chunk:
                    break
                fout.write(chunk)
        with contextlib.suppress(OSError):
            os.chmod(dst_full, src_full.stat().st_mode)
        self._audit("copy_file", src, dst=dst, bytes=size)
        return {"src": src, "dst": dst, "bytes": size, "copied": True}

    def make_dir(self, path: str, exist_ok: bool = True, parents: bool = True) -> dict:
        self._check_write()
        full    = self._resolve(path)
        existed = full.exists()
        if existed:
            if not full.is_dir():
                raise SandboxError(f"path exists and is not a directory: {path}")
            if not exist_ok:
                raise SandboxError(f"directory already exists: {path}")
        if self.dry_run:
            self._audit("make_dir", path, dry_run=True)
            return {"path": path, "created": not existed, "dry_run": True}
        try:
            full.mkdir(parents=parents, exist_ok=exist_ok)
        except OSError as e:
            raise SandboxError(f"cannot create directory: {e}")
        self._audit("make_dir", path)
        return {"path": path, "created": not existed}

    def remove_dir(self, path: str, recursive: bool = False, confirm: bool = False) -> dict:
        self._check_write()
        if not confirm:
            raise SandboxError("remove_dir requires confirm:true")
        full = self._resolve(path, must_exist=True)
        if not full.is_dir():
            raise SandboxError(f"not a directory: {path}")
        if full == self.root:
            raise SandboxError("refusing to remove sandbox root")
        try:
            entries = list(full.iterdir())
        except OSError as e:
            raise SandboxError(f"cannot list directory: {e}")
        if entries and not recursive:
            raise SandboxError(f"directory not empty ({len(entries)} entries) — set recursive:true")
        if self.dry_run:
            self._audit("remove_dir", path, recursive=recursive, dry_run=True)
            return {"path": path, "removed": True, "recursive": recursive, "dry_run": True}
        try:
            if recursive:
                shutil.rmtree(full)
            else:
                full.rmdir()
        except OSError as e:
            raise SandboxError(f"cannot remove directory: {e}")
        self._audit("remove_dir", path, recursive=recursive)
        return {"path": path, "removed": True, "recursive": recursive}

    def touch_file(self, path: str, update_mtime: bool = True) -> dict:
        self._check_write()
        full    = self._resolve(path)
        if full.exists() and full.is_dir():
            raise SandboxError(f"path is a directory: {path}")
        existed = full.exists()
        if self.dry_run:
            self._audit("touch_file", path, created=not existed, dry_run=True)
            return {"path": path, "created": not existed, "dry_run": True}
        full.parent.mkdir(parents=True, exist_ok=True)
        if not existed:
            full.touch()
        elif update_mtime:
            os.utime(full, None)
        self._audit("touch_file", path, created=not existed)
        return {"path": path, "created": not existed, "mtime": full.stat().st_mtime}

    def replace_in_file(self, path: str, pattern: str, replacement: str, count: int = 0) -> dict:
        self._check_write()
        if not isinstance(pattern, str) or not pattern:
            raise SandboxError("pattern must be a non-empty string")
        if not isinstance(replacement, str):
            raise SandboxError("replacement must be a string")
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise SandboxError(f"invalid regex: {e}")
        try:
            count = int(count)
        except (ValueError, TypeError):
            count = 0
        full     = self._resolve(path, must_exist=True)
        if not full.is_file():
            raise SandboxError(f"not a file: {path}")
        try:
            text = full.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise SandboxError(f"file is not valid UTF-8: {path}")
        new_text, n = rx.subn(replacement, text, count=max(0, count))
        if n == 0:
            raise SandboxError("pattern did not match")
        new_bytes = len(new_text.encode("utf-8"))
        if new_bytes > self.MAX_WRITE_BYTES:
            raise SandboxError(f"resulting file exceeds {self.MAX_WRITE_BYTES:,}-byte limit")
        if self.dry_run:
            self._audit("replace_in_file", path, replacements=n, dry_run=True)
            return {"path": path, "replacements": n, "delta_bytes": new_bytes - len(text.encode()), "dry_run": True}
        full.write_text(new_text, encoding="utf-8")
        self._audit("replace_in_file", path, replacements=n)
        return {"path": path, "replacements": n, "delta_bytes": new_bytes - len(text.encode())}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_SCHEMA: dict[str, tuple[str, ...]] = {
    "list_dir":        ("path", "show_hidden"),
    "read_file":       ("path", "start_line", "end_line"),
    "write_file":      ("path", "content", "overwrite"),
    "edit_file":       ("path", "old_string", "new_string"),
    "delete_file":     ("path", "confirm"),
    "glob":            ("pattern", "max_results"),
    "search":          ("pattern", "path", "max_results"),
    "move_file":       ("src", "dst", "overwrite"),
    "copy_file":       ("src", "dst", "overwrite"),
    "make_dir":        ("path", "exist_ok", "parents"),
    "file_info":       ("path",),
    "tree":            ("path", "max_depth", "show_hidden", "max_entries"),
    "append_file":     ("path", "content", "add_newline"),
    "head":            ("path", "lines"),
    "tail":            ("path", "lines"),
    "hash_file":       ("path", "algo"),
    "replace_in_file": ("path", "pattern", "replacement", "count"),
    "remove_dir":      ("path", "recursive", "confirm"),
    "touch_file":      ("path", "update_mtime"),
    "diff_files":      ("path_a", "path_b", "context"),
    "word_count":      ("path",),
    "done":            ("summary",),
    "grep":            ("pattern", "path", "max_results"),
    "count_matches":   ("path", "pattern"),
    "file_permissions": ("path", "mode"),
    "file_type":       ("path",),
    "run_command":     ("cmd", "cwd", "timeout"),
}


def dispatch_tool(sandbox: Sandbox, name: str, args: dict) -> dict:
    if not isinstance(args, dict):
        raise SandboxError(f"tool args must be an object, got {type(args).__name__}")
    if name == "done":
        return {"_done": True, "summary": str(args.get("summary", ""))}
    if name not in TOOL_SCHEMA:
        raise SandboxError(f"unknown tool: {name}. Available: {sorted(TOOL_SCHEMA)}")
    fn       = getattr(sandbox, name)
    allowed  = set(TOOL_SCHEMA[name])
    safe_args = {k: v for k, v in args.items() if k in allowed}
    return fn(**safe_args)


# ---------------------------------------------------------------------------
# LLM Client  (fixed: true async I/O, conversation history)
# ---------------------------------------------------------------------------

class LLMError(Exception):
    pass


class LLMClient:
    """Async OpenAI-compatible client. Sends full conversation history each turn."""

    def __init__(self, api_key: str, base_url: str, model: str,
                 timeout: int = 120) -> None:
        self.api_key  = api_key
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.timeout  = timeout

    async def complete(self, messages: list[dict], system: str = "",
                       model: Optional[str] = None,
                       stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """Returns assistant reply text. Raises LLMError on failure.
        If stream_callback is provided, enables streaming mode."""
        model = model or self.model
        payload_messages: list[dict] = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(messages)

        payload = {
            "model":       model,
            "messages":    payload_messages,
            "temperature": 0.2,
            "stream":      stream_callback is not None,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        url     = f"{self.base_url}/chat/completions"
        timeout = self.timeout

        if stream_callback:
            return await self._streaming_complete(url, payload, headers, timeout, stream_callback)

        def _blocking() -> str:
            import urllib.request
            import urllib.error
            data = json.dumps(payload).encode("utf-8")
            req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                try:
                    err = json.loads(body).get("error", {})
                    msg = err.get("message", str(e)) if isinstance(err, dict) else str(e)
                except Exception:
                    msg = f"HTTP {e.code}: {body[:300]}"
                raise LLMError(msg)
            except Exception as e:
                raise LLMError(f"Request failed: {e}")

            if "error" in result:
                err = result["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                raise LLMError(msg)

            choices = result.get("choices", [])
            if not choices:
                raise LLMError("Empty choices in response")
            return choices[0].get("message", {}).get("content", "") or ""

        return await asyncio.to_thread(_blocking)

    async def _streaming_complete(self, url: str, payload: dict,
                                   headers: dict, timeout: int,
                                   stream_callback: Callable[[str], None]) -> str:
        """Handle streaming response from OpenAI-compatible API."""
        import urllib.request
        import urllib.error

        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                full_text = ""
                for line in resp:
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        break
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            full_text += delta
                            stream_callback(delta)
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                err = json.loads(body).get("error", {})
                msg = err.get("message", str(e)) if isinstance(err, dict) else str(e)
            except Exception:
                msg = f"HTTP {e.code}: {body[:300]}"
            raise LLMError(msg)
        except Exception as e:
            raise LLMError(f"Streaming request failed: {e}")

        return full_text


# ---------------------------------------------------------------------------
# Tool call parser
# ---------------------------------------------------------------------------

_TOOL_BLOCK_RE = re.compile(
    r"```(?:tool|tool_call|tool_use)\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def _try_repair_json(raw: str) -> Optional[Any]:
    s = raw.strip()
    if not s or not s.startswith("{"):
        return None
    in_str = escape = False
    stack: list[str] = []
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    if in_str or not stack:
        return None
    try:
        return json.loads(s.rstrip().rstrip(",") + "".join(reversed(stack)))
    except json.JSONDecodeError:
        return None


def parse_tool_calls(text: str) -> list[dict]:
    if not text:
        return []
    calls: list[dict] = []
    for match in _TOOL_BLOCK_RE.finditer(text):
        raw       = match.group(1).strip()
        raw_clean = re.sub(r",(\s*[}\]])", r"\1", raw)
        try:
            obj = json.loads(raw_clean)
        except json.JSONDecodeError as e:
            repaired = _try_repair_json(raw_clean)
            if repaired is None:
                calls.append({"_parse_error": str(e), "raw": raw[:300]})
                continue
            obj = repaired
        if not isinstance(obj, dict):
            calls.append({"_parse_error": "not a JSON object", "raw": raw[:300]})
            continue
        if "name" not in obj:
            calls.append({"_parse_error": "missing 'name' field", "raw": raw[:300]})
            continue
        obj.setdefault("args", {})
        if not isinstance(obj["args"], dict):
            calls.append({"_parse_error": "'args' must be an object", "raw": raw[:300]})
            continue
        calls.append(obj)
    return calls


# ---------------------------------------------------------------------------
# No-call reprompt
# ---------------------------------------------------------------------------

NO_CALL_SOFT_LIMIT = 6
NO_CALL_HARD_LIMIT = 3


def _build_no_call_reprompt(*, attempt: int, last_tool: Optional[str],
                             had_prose: bool) -> str:
    fmt = '```tool\n{"name": "TOOL_NAME", "args": {...}}\n```'
    if attempt == 1:
        prose_hint = (
            " If you wrote prose, just append a tool block at the end."
            if had_prose else ""
        )
        return (
            "No ```tool``` block found in your response." + prose_hint +
            "\nEmit EXACTLY ONE tool call:\n" + fmt +
            "\nUnsure? Call list_dir with {\"path\": \".\"}."
        )
    if attempt == 2:
        suggestions = {
            "list_dir":  "Try read_file, glob, or tree next.",
            "read_file": "Try edit_file, search, or another read_file.",
            "search":    "Try read_file on one of the hits.",
            "glob":      "Try read_file on one of the matches.",
        }
        hint = suggestions.get(last_tool or "", "") if last_tool else ""
        return f"Still no tool block. {hint}\nYou MUST emit a fenced ```tool``` block:\n{fmt}"
    if attempt <= NO_CALL_SOFT_LIMIT:
        return (
            f"Reminder ({attempt}/{NO_CALL_SOFT_LIMIT}): no tool block in last "
            f"{attempt} response(s). Reply with ONLY a tool block:\n{fmt}"
        )
    return f"FINAL WARNING — {attempt} responses without a tool call. Emit NOW:\n{fmt}"


# ---------------------------------------------------------------------------
# Agent dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    iteration:   int
    model_text:  str = ""
    tool_name:   Optional[str] = None
    tool_args:   dict = field(default_factory=dict)
    tool_result: Any = None
    tool_result_summary: Any = None   # lightweight copy for callers
    error:       Optional[str] = None
    duration_ms: int = 0


@dataclass
class AgentResult:
    success:      bool
    summary:      str
    steps:        list[AgentStep]
    iterations:   int
    writes_audit: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success":      self.success,
            "summary":      self.summary,
            "iterations":   self.iterations,
            "writes_audit": self.writes_audit,
            "steps":        [asdict(s) for s in self.steps],
        }


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are an autonomous file-system agent operating inside a sandboxed folder.
You have NO direct OS access — every action goes through the tools below.
ROOT:    {root}
MODE:    {mode}
SESSION: {session_id}

═══════════════════════════════════════════════════════════════
TOOL PROTOCOL — MANDATORY RULES
═══════════════════════════════════════════════════════════════
1. Emit EXACTLY ONE fenced ```tool block per response.
2. Valid JSON only: double-quoted keys, no trailing commas.
3. "args" is always an object: "args": {{}}.
4. Call `done` ONLY after the task is FULLY complete.

```tool
{{"name": "TOOL_NAME", "args": {{"key": "value"}}}}
```

═══════════════════════════════════════════════════════════════
READ TOOLS (use these to investigate first)
═══════════════════════════════════════════════════════════════
list_dir(path=".", show_hidden=false)
    → Lists directory contents. Returns entries with name, type, size.
    Best for: exploring folder structure, finding files.

read_file(path, start_line=1, end_line=400)
    → Reads file content by line range. Returns total_lines, content, truncated flag.
    Best for: examining file contents, code review. Use start_line/end_line to limit output.

head(path, lines=20) / tail(path, lines=20)
    → Quick preview of first or last N lines.
    Best for: checking file headers, log endings without loading full file.

glob(pattern, max_results=200)
    → fnmatch-style file matching (e.g. "*.py", "**/*.js", "src/**/*.ts").
    Best for: finding files by name pattern across the entire tree.

search(pattern, path=".", max_results=100)
    → Regex search across files. Searches both filenames and line content.
    Best for: finding code patterns, function definitions, TODO comments.

file_info(path)
    → Returns file metadata: type, size, mtime, ctime, mode, is_binary, line_count.
    Best for: checking file properties without reading content.

tree(path=".", max_depth=3, show_hidden=false, max_entries=500)
    → Recursive directory tree. Returns hierarchical structure.
    Best for: understanding folder layout before exploring deeper.

word_count(path)
    → Returns lines, words, chars, bytes counts.
    Best for: analyzing file size, getting quick statistics.

diff_files(path_a, path_b, context=3)
    → Unified diff between two files. Returns identical flag, diff text, hunks.
    Best for: comparing file versions, checking changes.

hash_file(path, algo="sha256")
    → Computes cryptographic hash (md5, sha1, sha256, sha512).
    Best for: verifying file integrity, checksums.

grep(pattern, path=".", max_results=100)
    → Finds files containing a regex pattern. Returns list of filenames.
    Different from search: grep returns FILENAMES, search returns line content.
    Best for: finding which files contain a string, class, or function.

count_matches(path, pattern)
    → Counts regex matches in a file without modifying it.
    Returns total count and lines_with_match.
    Best for: quick analysis before using replace_in_file.

file_type(path)
    → Returns MIME type, file suffix, and binary flag.
    Best for: identifying file types without reading content.

file_permissions(path, mode="")
    → Get or set file permissions (octal mode like "644", "755").
    Without mode: returns current permissions. With mode: sets new permissions.
    Best for: making scripts executable, fixing permission issues.

run_command(cmd, cwd=".", timeout=30)
    → Executes a shell command in the sandbox directory.
    Cross-platform: uses cmd /c on Windows, /bin/sh -c on Linux/macOS.
    Returns returncode, stdout, stderr, ok, and platform flag.
    Best for: running tests, linters, git commands, build scripts.
    WARNING: Use only when tools above cannot accomplish the task.
    timeout max: 300 seconds.

═══════════════════════════════════════════════════════════════
WRITE TOOLS (use these to modify files)
═══════════════════════════════════════════════════════════════
write_file(path, content, overwrite=false)
    → Creates a new file. Set overwrite=true to replace existing file.
    Best for: creating new files, writing content from scratch.

edit_file(path, old_string, new_string)
    → Replaces ONE occurrence of old_string with new_string.
    Best for: targeted text changes, bug fixes, small modifications.
    IMPORTANT: old_string must match EXACTLY — including whitespace.

append_file(path, content, add_newline=false)
    → Appends text to end of file. Creates file if missing.
    Best for: adding logs, extending config files.

replace_in_file(path, pattern, replacement, count=0)
    → Regex find-and-replace. count=0 replaces ALL matches.
    Best for: bulk replacements, renaming patterns across files.

touch_file(path, update_mtime=true)
    → Creates empty file or updates modification time.
    Best for: creating placeholder files, updating timestamps.

delete_file(path, confirm=true)
    → Deletes a single file. Requires confirm=true.
    Best for: removing unwanted files.
    DANGER: Cannot undo — verify before deleting.

move_file(src, dst, overwrite=false)
    → Moves or renames a file.
    Best for: organizing files, renaming without copying.

copy_file(src, dst, overwrite=false)
    → Copies file contents (preserves permissions).
    Best for: duplicating files, backups.

make_dir(path, exist_ok=true, parents=true)
    → Creates directory. parents=true creates intermediate dirs.
    Best for: creating folder structures.

remove_dir(path, recursive=false, confirm=true)
    → Removes directory. recursive=true deletes contents first.
    Best for: cleaning up empty folders, removing entire directories.
    WARNING: recursive=true is irreversible — use with caution.

═══════════════════════════════════════════════════════════════
WORKFLOW PATTERNS
═══════════════════════════════════════════════════════════════
EXPLORE → INVESTIGATE → MODIFY → VERIFY → DONE

Explore a new project:
  1. list_dir(".") to see top-level structure
  2. tree(max_depth=2) to understand layout
  3. glob("*.py") to find all source files

Read and modify a file:
  1. read_file(path) to see current content
  2. edit_file(path, old, new) to make targeted change
  3. read_file(path) to verify the change

Bulk operations:
  1. search("pattern") to find all occurrences
  2. read_file() on each hit to verify context
  3. replace_in_file(path, pattern, replacement, count=N) to replace N matches

Create new file:
  1. write_file(path, content) to create
  2. read_file(path) to verify content

Find and understand:
  1. glob("**/*.py") to find relevant files
  2. read_file() on each to understand structure
  3. search("function_name") across matches

Quick code overview (FAST):
  1. grep("^(class |def |async def )", path=".") → list all classes/functions
  2. grep("import |from .* import", path=".") → find all imports
  3. run_command("wc -l *.py") → get file sizes at a glance

Analyze before modifying:
  1. grep("function_name") to find all files containing it
  2. count_matches("pattern") to see how many occurrences
  3. read_file() to examine specific files (use line ranges)
  4. replace_in_file() with count=N to replace specific matches

Understand new codebase:
  1. tree(".", max_depth=2) → get overview
  2. grep("^(class |def |async def )", path=".") → find all definitions
  3. grep("TODO|FIXME|XXX", path=".") → find notes
  4. read_file() specific files only as needed

Run tests/builds:
  1. run_command("pytest") to run tests
  2. run_command("npm build") to build project
  3. run_command("git status") to check git state
  4. run_command("ls -la")  # dir on Windows, ls -la on Linux/macOS

Fix permissions:
  1. file_permissions("script.sh") to see current mode
  2. file_permissions("script.sh", "755") to make executable

═══════════════════════════════════════════════════════════════
EFFICIENCY GUIDELINES (follow these for faster results)
══════════════════════════════════════════════════════════════
• Use grep/glob for code exploration instead of reading entire files
• Pattern: For understanding codebase → grep("^(class |def |async def )", path=".")
• Pattern: For finding files with patterns → glob("**/*.py") then grep on specific files
• Avoid sequential chunked reads — use search/grep for overview first
• Batch related operations: use run_command("grep -r 'pattern' .") instead of multiple reads
• Re-read files only if content changed; avoid redundant reads for same data

═══════════════════════════════════════════════════════════════
IMPORTANT REMINDERS
══════════════════════════════════════════════════════════════
• ALWAYS start by exploring with list_dir or tree before diving into files.
• Use line ranges in read_file to avoid overwhelming output.
• For edit_file: include surrounding context in old_string to ensure uniqueness.
• For replace_in_file: test with count=1 first, then use count=0 for all.
• When done, call done(summary="...") with a clear summary of what was done.
• NEVER call done before completing the task — models calling done early will be rejected.
"""


# ---------------------------------------------------------------------------
# File Agent  (fixed: conversation history, async I/O)
# ---------------------------------------------------------------------------

class FileAgent:
    DEFAULT_MAX_RESULT_CHARS = 50_000

    def __init__(self, client: LLMClient, root: str | Path, *,
                 max_iterations: int = 100,
                 read_only: bool = False,
                 dry_run: bool = False,
                 allow_dangerous: bool = False,
                 max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
                 require_investigation: bool = True) -> None:
        self.client  = client
        self.sandbox = Sandbox(root, read_only=read_only, dry_run=dry_run,
                               allow_dangerous=allow_dangerous)
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.max_iterations     = max_iterations
        self.max_result_chars   = max(1000, int(max_result_chars))
        self.require_investigation = require_investigation

    def _format_result(self, result: Any) -> str:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            text = json.dumps({"_serialize_error": str(e)}, separators=(",", ":"))
        if len(text) > self.max_result_chars:
            text = text[:self.max_result_chars - 50] + '..."[TRUNCATED]"'
        return text

    async def run(self, task: str,
                  on_step: Optional[Callable[[AgentStep], None]] = None,
                  stream_callback: Optional[Callable[[str], None]] = None
                  ) -> AgentResult:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        session_id = uuid.uuid4().hex[:12]
        mode_parts = []
        if self.sandbox.read_only: mode_parts.append("READ_ONLY")
        if self.sandbox.dry_run:   mode_parts.append("DRY_RUN")
        mode_str = "+".join(mode_parts) or "READ_WRITE"

        system = SYSTEM_PROMPT_TEMPLATE.format(
            root=self.sandbox.root,
            mode=mode_str,
            session_id=session_id,
        )

        caps_line = (
            "READ + WRITE (write_file, edit_file, delete_file, …)"
            if not self.sandbox.read_only else
            "READ ONLY (list_dir, read_file, glob, search, …)"
        )
        first_user_msg = (
            f"=== NEW TASK — session {session_id} ===\n\n"
            f"TASK: {task}\n\n"
            f"TOOLS: {caps_line}\n\n"
            "Begin. Emit your first tool call."
        )

        # Conversation history (accumulated across turns)
        # Fixed bug: original sent only the latest message each turn,
        # giving the model zero memory of prior tool calls/results.
        messages: list[dict] = [{"role": "user", "content": first_user_msg}]

        steps: list[AgentStep] = []
        consecutive_no_call    = 0
        investigation_done     = 0
        last_successful_tool: Optional[str] = None

        for i in range(1, self.max_iterations + 1):
            log.info("iteration %d/%d", i, self.max_iterations)
            t0   = time.time()
            step = AgentStep(iteration=i)

            try:
                reply = await self.client.complete(messages, system=system,
                                                    stream_callback=stream_callback)
            except LLMError as e:
                step.error       = f"LLM error: {e}"
                step.duration_ms = int((time.time() - t0) * 1000)
                steps.append(step)
                if on_step:
                    with contextlib.suppress(Exception):
                        on_step(step)
                return AgentResult(
                    success=False, summary=f"LLM failure: {e}",
                    steps=steps, iterations=i,
                    writes_audit=self.sandbox.writes_audit(),
                )

            step.model_text = reply
            # Add assistant reply to history
            messages.append({"role": "assistant", "content": reply})

            calls       = parse_tool_calls(reply)
            valid_calls = [c for c in calls if "_parse_error" not in c]
            parse_errors = [c for c in calls if "_parse_error" in c]

            # ---- no tool call ----
            if not valid_calls and not parse_errors:
                consecutive_no_call += 1
                step.tool_name  = "<no-call>"
                preview         = (reply or "").strip()
                step.error      = (
                    f"no tool call ({consecutive_no_call}/{NO_CALL_SOFT_LIMIT}): "
                    + (preview[:160].replace("\n", " ") if preview else "empty")
                )
                step.duration_ms = int((time.time() - t0) * 1000)
                steps.append(step)
                if on_step:
                    with contextlib.suppress(Exception):
                        on_step(step)

                if consecutive_no_call >= NO_CALL_HARD_LIMIT and investigation_done == 0:
                    return AgentResult(
                        success=False,
                        summary=f"model emitted {consecutive_no_call} empty responses with no tool calls",
                        steps=steps, iterations=i,
                        writes_audit=self.sandbox.writes_audit(),
                    )
                if consecutive_no_call >= NO_CALL_SOFT_LIMIT:
                    return AgentResult(
                        success=False,
                        summary=f"model stuck: {consecutive_no_call} consecutive responses without a tool call",
                        steps=steps, iterations=i,
                        writes_audit=self.sandbox.writes_audit(),
                    )

                reprompt = _build_no_call_reprompt(
                    attempt=consecutive_no_call,
                    last_tool=last_successful_tool,
                    had_prose=bool(preview),
                )
                messages.append({"role": "user", "content": reprompt})
                continue

            consecutive_no_call = 0

            # ---- parse error ----
            if parse_errors and not valid_calls:
                err = parse_errors[0]
                step.error      = f"parse error: {err['_parse_error']}"
                step.duration_ms = int((time.time() - t0) * 1000)
                steps.append(step)
                reprompt = (
                    f"JSON parse error: {err['_parse_error']}\n"
                    "Re-emit the tool block with valid JSON."
                )
                messages.append({"role": "user", "content": reprompt})
                continue

            call           = valid_calls[0]
            step.tool_name = call.get("name", "")
            step.tool_args = call.get("args", {}) or {}

            # ---- require at least one investigation tool before done ----
            if self.require_investigation and step.tool_name == "done" and investigation_done == 0:
                step.error      = "rejected: done called before any investigation"
                step.duration_ms = int((time.time() - t0) * 1000)
                steps.append(step)
                if on_step:
                    with contextlib.suppress(Exception):
                        on_step(step)
                reprompt = "REJECTED: call list_dir or another tool first before calling done."
                messages.append({"role": "user", "content": reprompt})
                continue

            # ---- dispatch ----
            try:
                result = dispatch_tool(self.sandbox, step.tool_name, step.tool_args)
            except SandboxError as e:
                step.error      = str(e)
                step.duration_ms = int((time.time() - t0) * 1000)
                steps.append(step)
                if on_step:
                    with contextlib.suppress(Exception):
                        on_step(step)
                err_payload = json.dumps({"error": str(e)}, ensure_ascii=False)
                reprompt    = f"```tool_result\n{err_payload}\n```\nTool returned an error. Try a different approach."
                messages.append({"role": "user", "content": reprompt})
                continue
            except Exception as e:
                step.error      = f"unexpected: {e}"
                step.duration_ms = int((time.time() - t0) * 1000)
                steps.append(step)
                if on_step:
                    with contextlib.suppress(Exception):
                        on_step(step)
                messages.append({"role": "user", "content": f"Tool crashed: {e}. Try a different approach."})
                continue

            step.tool_result = result
            step.duration_ms = int((time.time() - t0) * 1000)

            # Store lightweight summary separately — don't overwrite the real result
            if isinstance(result, dict):
                summary_keys = ("path", "count", "total_lines", "start_line",
                                "end_line", "truncated", "bytes", "created",
                                "deleted", "dry_run", "moved", "copied")
                step.tool_result_summary = {k: result[k] for k in summary_keys if k in result}
            else:
                step.tool_result_summary = {}

            steps.append(step)
            if on_step:
                with contextlib.suppress(Exception):
                    on_step(step)

            # ---- done signal ----
            if isinstance(result, dict) and result.get("_done"):
                return AgentResult(
                    success=True,
                    summary=str(result.get("summary", "")),
                    steps=steps, iterations=i,
                    writes_audit=self.sandbox.writes_audit(),
                )

            investigation_done     += 1
            last_successful_tool    = step.tool_name
            result_text             = self._format_result(result)
            step.tool_result_summary["_payload_chars"] = len(result_text)

            reprompt = (
                f"```tool_result\n{result_text}\n```\n"
                f"Task: {task}\n"
                "Next tool call, or `done` when fully complete."
            )
            messages.append({"role": "user", "content": reprompt})

        return AgentResult(
            success=False,
            summary=f"max iterations ({self.max_iterations}) reached without completion",
            steps=steps, iterations=self.max_iterations,
            writes_audit=self.sandbox.writes_audit(),
        )


# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

_USE_COLOR = True


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _green(t):  return _c("32", t)
def _red(t):    return _c("31", t)
def _yellow(t): return _c("33", t)
def _blue(t):   return _c("34", t)
def _cyan(t):   return _c("36", t)
def _grey(t):   return _c("90", t)
def _bold(t):   return _c("1",  t)


def _fmt_bytes(n: int) -> str:
    if n < 1024:        return f"{n}B"
    if n < 1048576:     return f"{n/1024:.1f}KB"
    return f"{n/1048576:.2f}MB"


# ---------------------------------------------------------------------------
# Live stats / step reporter
# ---------------------------------------------------------------------------

class _LiveStats:
    def __init__(self, max_iter: int, *, show_stats: bool = True,
                 show_thoughts: bool = False) -> None:
        self.max_iter      = max_iter
        self.show_stats    = show_stats
        self.show_thoughts = show_thoughts
        self.t_start       = time.time()
        self.tools_ok      = 0
        self.tools_err     = 0
        self.no_call_warns = 0
        self.bytes_in      = 0
        self.total_ms      = 0
        self.tool_counts: dict[str, int] = {}

    def update(self, step: AgentStep) -> None:
        self.total_ms += step.duration_ms or 0
        is_no_call     = bool(step.error and step.error.startswith("no tool call"))
        if is_no_call:
            self.no_call_warns += 1
            return
        if step.tool_name and step.tool_name != "<no-call>":
            self.tool_counts[step.tool_name] = self.tool_counts.get(step.tool_name, 0) + 1
        if step.error:
            self.tools_err += 1
        elif step.tool_name and step.tool_name != "<no-call>":
            self.tools_ok += 1
        if isinstance(step.tool_result_summary, dict):
            self.bytes_in += int(step.tool_result_summary.get("_payload_chars", 0) or 0)

    def elapsed(self) -> float:
        return time.time() - self.t_start

    def _bar(self, current: int, width: int = 18) -> str:
        frac   = max(0.0, min(1.0, current / max(1, self.max_iter)))
        filled = int(frac * width)
        return _grey("[") + _green("#" * filled) + _grey("-" * (width - filled)) + _grey("]")

    def render_line(self, current: int) -> str:
        bar = self._bar(current)
        nc  = f"  {_yellow('○')} {self.no_call_warns}" if self.no_call_warns else ""
        return (
            f"  {bar} {self.elapsed():.1f}s  "
            f"{_green('ok')} {self.tools_ok}  "
            f"{_red('err')} {self.tools_err}{nc}"
        )

    def render_footer(self, current: int) -> str:
        bar  = self._bar(current)
        parts = [
            f"  {bar} [{current}/{self.max_iter}]",
            f"{_green('ok')}{self.tools_ok}",
            f"{_red('err')}{self.tools_err}",
            f"{_yellow('○')}{self.no_call_warns}" if self.no_call_warns else "",
            f"{_bold('ms')} {self.total_ms}",
            f"{_grey(f'elapsed {self.elapsed():.1f}s')}",
        ]
        return "  ".join(p for p in parts if p)


_TOOL_STRIP_RE = re.compile(r"```(?:tool|tool_call|tool_use)\s*\n.*?\n```", re.DOTALL | re.IGNORECASE)


def _strip_tool_blocks(text: str) -> str:
    return _TOOL_STRIP_RE.sub("", text or "").strip()


def _make_stream_callback() -> Callable[[str], None]:
    sys.stderr.write(_grey("\n  "))
    sys.stderr.flush()
    def _stream(token: str) -> None:
        sys.stderr.write(token)
        sys.stderr.flush()
    return _stream


def _make_step_reporter(stats: _LiveStats, max_iter: int) -> Callable[[AgentStep], None]:
    def _emit(step: AgentStep) -> None:
        if stats.show_thoughts:
            thought = _strip_tool_blocks(step.model_text or "")
            for ln in thought.splitlines()[:6]:
                print(f"  {_grey('│')} {ln[:200]}", file=sys.stderr)

        stats.update(step)

        is_no_call = step.error and step.error.startswith("no tool call")
        if is_no_call:
            marker = _yellow("○")
            detail = f"  {_yellow('─')} {step.error}"
        elif step.error:
            marker = _red("✗")
            detail = f"  {_red('─')} {step.error}"
        else:
            marker = _green("✓")
            detail = ""

        name      = step.tool_name or "<no-call>"
        arg_keys  = list(step.tool_args.keys()) if step.tool_args else []
        hint      = ""
        if not step.error and isinstance(step.tool_result_summary, dict):
            r = step.tool_result_summary
            if "count" in r:         hint = f"  → {r['count']} entries"
            elif "total_lines" in r: hint = f"  → {r['total_lines']} lines"
            elif "bytes" in r and r.get("bytes") is not None:
                hint = f"  → {_fmt_bytes(int(r['bytes']))}"

        counter = _bold(f"[{step.iteration:>2}/{max_iter}]")
        print(
            f"  {counter} {marker} {name}({', '.join(arg_keys)}){hint}{detail}"
            f"  {_grey(f'({step.duration_ms}ms)')}",
            file=sys.stderr,
        )
        if stats.show_stats:
            print(stats.render_line(step.iteration), file=sys.stderr)

    return _emit


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def _load_history(limit: int = 100) -> list[str]:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    except OSError:
        return []
    seen: set = set()
    out: list[str] = []
    for ln in reversed(lines):
        task = ln.split("\t", 1)[-1] if "\t" in ln else ln
        if task in seen:
            continue
        seen.add(task)
        out.append(task)
        if len(out) >= limit:
            break
    return list(reversed(out))


def _append_history(task: str) -> None:
    one_line = task.replace("\n", " ").strip()
    if not one_line:
        return
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{one_line}\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Banner / summary printing
# ---------------------------------------------------------------------------

def _print_banner(root: Path, profile_name: str, model: str,
                  mode: str, max_iter: int, task: str) -> None:
    sep = _grey("─" * 60)
    print("", file=sys.stderr)
    print(sep, file=sys.stderr)
    print(f"  {_bold(_cyan(f'File Agent v{VERSION}'))}  {_grey(f'profile: {profile_name}')}", file=sys.stderr)
    print(sep, file=sys.stderr)
    print(f"  {_grey('Root')}    {_bold(str(root))}", file=sys.stderr)
    print(f"  {_grey('Model')}   {model}", file=sys.stderr)
    print(f"  {_grey('Mode')}    {_yellow(mode)}", file=sys.stderr)
    print(f"  {_grey('MaxIter')} {max_iter}", file=sys.stderr)
    preview = task if len(task) <= 200 else task[:200] + "…"
    print(f"  {_grey('Task')}    {preview}", file=sys.stderr)
    print(sep + "\n", file=sys.stderr)


def _print_summary(result: AgentResult, stats: _LiveStats) -> None:
    sep = _grey("─" * 60)
    print("\n" + sep, file=sys.stderr)
    if result.success:
        print(f"  {_green('SUCCESS')} after {result.iterations} iteration(s)", file=sys.stderr)
    else:
        print(f"  {_red('FAILED')} after {result.iterations} iteration(s)", file=sys.stderr)
    if result.summary:
        print(f"  {_grey('Summary')}  {result.summary}", file=sys.stderr)
    nc = f"  {_yellow('○')} {stats.no_call_warns}" if stats.no_call_warns else ""
    print(
        f"  {_grey('Stats')}    {_green('ok')} {stats.tools_ok}  "
        f"{_red('err')} {stats.tools_err}{nc}  "
        f"elapsed {stats.elapsed():.1f}s",
        file=sys.stderr,
    )
    print(f"\n{stats.render_footer(result.iterations)}\n", file=sys.stderr)
    print(sep, file=sys.stderr)


# ---------------------------------------------------------------------------
# Interactive config command handler
# ---------------------------------------------------------------------------

CONFIG_HELP = """\
Config commands:
  /config                   Show active profile
  /profiles                 List all profiles
  /profile use <name>       Switch to a profile
  /profile save <name>      Save current session settings as a new profile
  /profile delete <name>    Delete a profile
  /profile show <name>      Show a profile's details
  /set key <value>          Set API key for active profile
  /set url <value>          Set base URL for active profile
  /set model <name>         Set active model (adds to list if new)
  /set iter <n>             Set max iterations for active profile
  /model add <name>         Add a model to the active profile
  /model remove <name>      Remove a model from the active profile
  /models                   List models in active profile
  /root [path]              Show or change the sandbox root
  /history                  Show recent task history
  /help                     Show this help
  /quit                     Exit

CLI flags (for one-shot runs):
  --no-stream               Disable token streaming (default: on)
  --show-thoughts / -t      Show model reasoning in real-time
  --no-stats                Hide live stats bar
  --read-only               Disable all write operations
  --dry-run                 Simulate writes without executing them
  --max-iter / -n <n>       Override max iterations
  --quiet / -q              Minimal output
  --verbose / -v            Debug logging"""


def handle_config_command(cmd: str, cfg: ConfigManager,
                          sandbox_root: list[str]) -> Optional[str]:
    """
    Handle a / command. Returns a status string, or None if not a config command.
    sandbox_root is a one-element list so the caller can see root changes.
    """
    parts = cmd.strip().split(None, 2)
    if not parts or not parts[0].startswith("/"):
        return None

    verb = parts[0].lower()

    # /help
    if verb in ("/help", "/?"):
        print(CONFIG_HELP, file=sys.stderr)
        return "ok"

    # /config
    if verb == "/config":
        name = cfg.active_profile_name
        p    = cfg.active_profile()
        key_display = "****" if p["api_key"] else "(not set)"
        print(f"\n  Active profile : {_bold(name)}", file=sys.stderr)
        print(f"  API key        : {key_display}", file=sys.stderr)
        print(f"  Base URL       : {p['base_url']}", file=sys.stderr)
        print(f"  Active model   : {p['active_model'] or '(not set)'}", file=sys.stderr)
        print(f"  Models         : {', '.join(p['models']) or '(none)'}", file=sys.stderr)
        print(f"  Max iter       : {p['max_iter']}", file=sys.stderr)
        print(f"  Read-only      : {p['read_only']}", file=sys.stderr)
        print(f"  Dry-run        : {p['dry_run']}", file=sys.stderr)
        print(f"  Sandbox root   : {sandbox_root[0] or '(not set)'}\n", file=sys.stderr)
        return "ok"

    # /profiles
    if verb == "/profiles":
        names   = cfg.profile_names()
        active  = cfg.active_profile_name
        if not names:
            print("  (no profiles saved)", file=sys.stderr)
        else:
            for n in names:
                marker = _green("▸") if n == active else " "
                p      = cfg.get_profile(n)
                print(f"  {marker} {_bold(n)}  {_grey(p.get('base_url',''))}  {p.get('active_model','')}", file=sys.stderr)
        return "ok"

    # /profile <sub> [name]
    if verb == "/profile":
        sub  = parts[1].lower() if len(parts) > 1 else ""
        name = parts[2].strip() if len(parts) > 2 else ""

        if sub == "use":
            if not name:
                return _red("Usage: /profile use <name>")
            if cfg.use_profile(name):
                return _green(f"Switched to profile '{name}'")
            return _red(f"Profile '{name}' not found. Use /profiles to list.")

        if sub == "delete":
            if not name:
                return _red("Usage: /profile delete <name>")
            if cfg.delete_profile(name):
                return _green(f"Profile '{name}' deleted.")
            return _red(f"Profile '{name}' not found.")

        if sub == "save":
            if not name:
                return _red("Usage: /profile save <name>")
            cfg.save_profile(name, cfg.active_profile())
            cfg.active_profile_name = name
            cfg.save()
            return _green(f"Current settings saved as profile '{name}'.")

        if sub == "show":
            target = name or cfg.active_profile_name
            p      = cfg.get_profile(target)
            key_d  = ("*" * min(8, len(p["api_key"])) + p["api_key"][-4:]) if p["api_key"] else "(not set)"
            print(f"\n  Profile : {_bold(target)}", file=sys.stderr)
            print(f"  key     : {key_d}", file=sys.stderr)
            print(f"  url     : {p['base_url']}", file=sys.stderr)
            print(f"  model   : {p['active_model']}", file=sys.stderr)
            print(f"  models  : {', '.join(p['models']) or '(none)'}", file=sys.stderr)
            print(f"  iter    : {p['max_iter']}", file=sys.stderr)
            return "ok"

        return _red(f"Unknown sub-command '{sub}'. Try: use, save, delete, show")

    # /set <field> <value>
    if verb == "/set":
        field_ = parts[1].lower() if len(parts) > 1 else ""
        value  = parts[2].strip() if len(parts) > 2 else ""

        if field_ == "key":
            if not value:
                return _red("Usage: /set key <api_key>")
            cfg.update_active(api_key=value)
            return _green("API key updated.")

        if field_ == "url":
            if not value:
                return _red("Usage: /set url <base_url>")
            cfg.update_active(base_url=value)
            return _green(f"Base URL set to '{value}'.")

        if field_ == "model":
            if not value:
                return _red("Usage: /set model <model_id>")
            cfg.set_active_model(value)
            return _green(f"Active model set to '{value}'.")

        if field_ == "iter":
            try:
                n = int(value)
                assert n > 0
            except (ValueError, AssertionError):
                return _red("Usage: /set iter <positive_integer>")
            cfg.update_active(max_iter=n)
            return _green(f"Max iterations set to {n}.")

        return _red(f"Unknown field '{field_}'. Options: key, url, model, iter")

    # /model add|remove
    if verb == "/model":
        sub   = parts[1].lower() if len(parts) > 1 else ""
        model = parts[2].strip() if len(parts) > 2 else ""
        if sub == "add":
            if not model:
                return _red("Usage: /model add <model_id>")
            cfg.add_model(model)
            return _green(f"Model '{model}' added.")
        if sub == "remove":
            if not model:
                return _red("Usage: /model remove <model_id>")
            return _green(f"Model '{model}' removed.") if cfg.remove_model(model) \
                   else _red(f"Model '{model}' not in list.")
        return _red("Usage: /model add|remove <model_id>")

    # /models
    if verb == "/models":
        p      = cfg.active_profile()
        active = p.get("active_model", "")
        models = p.get("models", [])
        if not models:
            print("  (no models saved — use /model add <id>)", file=sys.stderr)
        else:
            for m in models:
                mark = _green("▸") if m == active else " "
                print(f"  {mark} {m}", file=sys.stderr)
        return "ok"

    # /root [path]
    if verb == "/root":
        new_root = parts[1].strip() if len(parts) > 1 else ""
        if new_root:
            resolved = str(Path(new_root).expanduser().resolve())
            sandbox_root[0] = resolved
            cfg.last_sandbox_root = resolved
            return _green(f"Sandbox root set to '{resolved}'.")
        return f"  Current sandbox root: {sandbox_root[0] or '(not set)'}"

    # /history
    if verb == "/history":
        hist = _load_history(30)
        if not hist:
            print("  (no history)", file=sys.stderr)
        else:
            for idx, h in enumerate(hist[-20:], 1):
                print(f"  {_grey(f'{idx:>3}.')} {h[:100]}", file=sys.stderr)
        return "ok"

    # /quit / /exit
    if verb in ("/quit", "/exit"):
        return "__quit__"

    return _red(f"Unknown command '{verb}'. Type /help for help.")


# ---------------------------------------------------------------------------
# Sandbox root resolution
# ---------------------------------------------------------------------------

def _resolve_root(cli_root: Optional[str], cfg: ConfigManager) -> Optional[str]:
    if cli_root:
        return cli_root
    env = os.environ.get("AGENT_ROOT", "").strip()
    if env:
        return env
    saved = cfg.last_sandbox_root
    if saved and Path(saved).is_dir():
        print(f"  {_grey('Using last sandbox root:')} {saved}", file=sys.stderr)
        return saved
    cwd = Path(os.getcwd()).resolve()
    if cwd not in Sandbox.DANGEROUS_ROOTS:
        return str(cwd)
    # Must prompt
    print(_yellow("  Current directory is unsafe as sandbox root."), file=sys.stderr)
    sys.stderr.write(_cyan("  Sandbox root: "))
    sys.stderr.flush()
    try:
        ans = input().strip()
    except EOFError:
        return None
    return ans or None


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _prompt_task() -> str:
    print(_bold(_cyan("\nTask")) + _grey("  (blank line or '.' to submit)"), file=sys.stderr)
    sys.stderr.write(_cyan("  > "))
    sys.stderr.flush()
    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip() == "." or (not line and lines):
                break
            if not line and not lines:
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-agent.py",
        description=f"File Agent v{VERSION} — autonomous file ops via LLM",
    )
    p.add_argument("task",          nargs="?",            help="Task description (optional)")
    p.add_argument("--root",   "-r", default=None,         help="Sandbox root folder")
    p.add_argument("--task-file", "-f", default=None,      help="Read task from file")
    p.add_argument("--profile", "-p", default=None,        help="Profile name to use")
    p.add_argument("--model",   "-m", default=None,        help="Override model ID")
    p.add_argument("--api-key",       default=None,        help="Override API key")
    p.add_argument("--api-url",       default=None,        help="Override API base URL")
    p.add_argument("--max-iter", "-n", type=int, default=None, help="Max iterations")
    p.add_argument("--read-only",     action="store_true", help="Disable all writes")
    p.add_argument("--dry-run",       action="store_true", help="Simulate writes only")
    p.add_argument("--allow-dangerous", action="store_true", help="Skip dangerous-root check")
    p.add_argument("--max-result-chars", type=int, default=50_000)
    p.add_argument("--output", "-o",  default=None,        help="Write result JSON here")
    p.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    p.add_argument("--no-color",     action="store_true")
    p.add_argument("--no-stats",     action="store_true")
    p.add_argument("--show-thoughts","-t", action="store_true", help="Stream model prose")
    p.add_argument("--no-stream",   action="store_true", help="Disable streaming")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--quiet",  "-q", action="store_true")
    p.add_argument("--version", "-V", action="version", version=f"File Agent v{VERSION}")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _main() -> int:
    global _USE_COLOR
    args   = _build_parser().parse_args()
    _configure_logging("DEBUG" if args.verbose else ("WARNING" if args.quiet else "INFO"))

    if args.no_color or not sys.stderr.isatty():
        _USE_COLOR = False

    # ---- load / bootstrap config ----
    cfg = ConfigManager()
    if not cfg.has_profiles():
        ok = run_wizard(cfg)
        if not ok:
            print(_red("Setup cancelled."), file=sys.stderr)
            return 2

    # ---- select profile ----
    if args.profile:
        if not cfg.use_profile(args.profile):
            print(_red(f"Profile '{args.profile}' not found. Available: {cfg.profile_names()}"), file=sys.stderr)
            return 1

    profile = cfg.active_profile()

    # ---- resolve credentials (CLI overrides profile) ----
    api_key  = args.api_key  or profile.get("api_key",      "")
    api_url  = args.api_url  or profile.get("base_url",     "https://api.openai.com/v1")
    model    = args.model    or profile.get("active_model", "")
    max_iter = args.max_iter or profile.get("max_iter",     100)
    read_only = args.read_only or profile.get("read_only",  False)
    dry_run   = args.dry_run   or profile.get("dry_run",    False)

    if not api_key:
        print(_yellow("  No API key set. Use /set key <value> or run with --api-key"), file=sys.stderr)

    if not model:
        print(_yellow("  No model set. Use /set model <id> or run with --model"), file=sys.stderr)

    # ---- sandbox root ----
    sandbox_root = [_resolve_root(args.root, cfg) or ""]

    # ---- build LLM client ----
    client = LLMClient(api_key=api_key, base_url=api_url, model=model)

    mode_str = ("READ_ONLY" if read_only else "READ_WRITE") + (" + DRY_RUN" if dry_run else "")

    def _build_agent(root: str) -> Optional[FileAgent]:
        try:
            return FileAgent(
                client=client, root=root,
                max_iterations=max_iter,
                read_only=read_only, dry_run=dry_run,
                allow_dangerous=args.allow_dangerous,
                max_result_chars=args.max_result_chars,
            )
        except SandboxError as e:
            print(_red(f"Sandbox error: {e}"), file=sys.stderr)
            return None

    def _run_one(task: str, root: str) -> int:
        agent = _build_agent(root)
        if agent is None:
            return 4
        if not args.quiet:
            _print_banner(agent.sandbox.root, cfg.active_profile_name,
                          client.model, mode_str, max_iter, task)
        _append_history(task)
        cfg.last_sandbox_root = root

        stats    = _LiveStats(max_iter, show_stats=not args.no_stats,
                              show_thoughts=args.show_thoughts)
        callback = None if args.quiet else _make_step_reporter(stats, max_iter)
        stream_cb = None
        if not args.no_stream and not args.quiet:
            stream_cb = _make_stream_callback()

        try:
            result = asyncio.run(agent.run(task, on_step=callback, stream_callback=stream_cb))
        except KeyboardInterrupt:
            print(_red("\n  Interrupted."), file=sys.stderr)
            return 130

        if not args.quiet:
            _print_summary(result, stats)

        if args.output:
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
                print(_grey(f"  Result written to {args.output}"), file=sys.stderr)
            except OSError as e:
                print(_red(f"Could not write output: {e}"), file=sys.stderr)

        return 0 if result.success else 1

    # ---- one-shot mode (task given on CLI) ----
    if args.task or args.task_file:
        if args.task_file:
            try:
                task = Path(args.task_file).read_text(encoding="utf-8").strip()
            except OSError as e:
                print(_red(f"Cannot read task file: {e}"), file=sys.stderr)
                return 2
        else:
            task = args.task.strip()
        if not task:
            print(_red("Task is empty."), file=sys.stderr)
            return 2
        if not sandbox_root[0]:
            print(_red("No sandbox root. Use --root or run interactively."), file=sys.stderr)
            return 2
        return _run_one(task, sandbox_root[0])

    # ---- interactive / zero-arg mode ----
    print(
        f"\n  {_bold(_cyan(f'File Agent v{VERSION}'))}  "
        f"profile: {_bold(cfg.active_profile_name)}  "
        f"model: {_bold(client.model or '(none)')}",
        file=sys.stderr,
    )
    if sandbox_root[0]:
        print(f"  root: {_bold(sandbox_root[0])}", file=sys.stderr)
    else:
        print(_yellow("  No sandbox root set — use /root <path> before running a task."), file=sys.stderr)
    print(_grey("  Type /help for commands, /quit to exit.\n"), file=sys.stderr)

    last_code = 0
    while True:
        try:
            task = _prompt_task()
        except KeyboardInterrupt:
            print(_grey("\n  (use /quit to exit)"), file=sys.stderr)
            continue

        if not task:
            continue

        # Handle /commands
        if task.startswith("/"):
            result = handle_config_command(task, cfg, sandbox_root)
            if result == "__quit__":
                break
            if result and result != "ok":
                print(f"  {result}", file=sys.stderr)
            # Rebuild client if credentials changed
            p = cfg.active_profile()
            client.api_key  = args.api_key  or p.get("api_key",      client.api_key)
            client.base_url = args.api_url  or p.get("base_url",     client.base_url)
            client.model    = args.model    or p.get("active_model", client.model)
            continue

        # !! = repeat last task
        if task in ("!!", "/last"):
            hist = _load_history(1)
            if not hist:
                print(_yellow("  No history yet."), file=sys.stderr)
                continue
            task = hist[-1]
            print(_grey(f"  Repeating: {task[:80]}"), file=sys.stderr)

        if not sandbox_root[0]:
            print(_yellow("  Set a sandbox root first: /root <path>"), file=sys.stderr)
            continue

        if not client.api_key:
            print(_yellow("  Set API key first: /set key <your_key>"), file=sys.stderr)
            continue

        if not client.model:
            print(_yellow("  Set a model first: /set model <model_id>"), file=sys.stderr)
            continue

        last_code = _run_one(task, sandbox_root[0])

    return last_code


if __name__ == "__main__":
    sys.exit(_main())
