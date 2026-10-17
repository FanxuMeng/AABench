#!/usr/bin/env python3
"""Download the complete public AA-LCR repository. Python 3.9+, no dependencies.

Repeated runs resume .part files via HTTP Range and verify upstream hashes.
Usage: python3 download_aa_lcr.py --output ./AA-LCR
       python3 download_aa_lcr.py --output ./AA-LCR --verify-only
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = "ArtificialAnalysis/AA-LCR"
ENDPOINT = "https://huggingface.co"
DEFAULT_REVISION = "9a77ef56b717057ade24ceab4d273712a0b4f19e"
META = ".aa_lcr_download"
REQUIRED = {"AA-LCR_Dataset.csv", "extracted_text/AA-LCR_extracted-text.zip", "README.md"}
CHUNK = 64 * 1024


class DownloadError(Exception):
    pass


class RetryableError(DownloadError):
    pass


def safe_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise DownloadError("文件清单中存在非法路径。")
    rel = PurePosixPath(name)
    if rel.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise DownloadError("文件清单中存在越界路径。")
    target = root.joinpath(*rel.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise DownloadError("文件路径通过符号链接指向下载目录外。")
    if any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
        # The output root is resolved before entering the downloader.
        raise DownloadError("下载目录内不应包含符号链接。")
    return target


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("not an object")
        return data
    except (ValueError, OSError) as e:
        raise DownloadError(f"无法读取状态/校验文件 {path.name}；请保留原目录排查。") from e


def validate_state(state: dict, root: Path) -> None:
    if state.get("schema") != 1 or state.get("repo_id") != REPO:
        raise DownloadError("状态文件版本或仓库不匹配，请使用新的下载目录。")
    commit = state.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise DownloadError("状态文件中的 commit 无效。")
    files = state.get("files")
    if not isinstance(files, list) or not files:
        raise DownloadError("文件清单为空或格式错误。")
    names = set()
    for item in files:
        if not isinstance(item, dict):
            raise DownloadError("文件记录格式错误。")
        name = item.get("path")
        safe_path(root, name)
        if name in names or name.split("/")[0] == META:
            raise DownloadError("文件清单包含重复路径或保留路径。")
        names.add(name)
        if type(item.get("size")) is not int or item["size"] < 0:
            raise DownloadError("缺少有效的远端文件大小。")
        kind = item.get("hash_kind")
        width = 64 if kind == "sha256" else 40 if kind == "git-sha1" else 0
        digest = item.get("hash")
        if not width or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{%d}" % width, digest):
            raise DownloadError("缺少有效的远端文件哈希。")
    if not REQUIRED.issubset(names):
        raise DownloadError("该 revision 缺少题目 CSV、上下文 ZIP 或 README，拒绝声明完整下载。")


def checksums(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1()
    git_sha1.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            sha256.update(chunk)
            git_sha1.update(chunk)
    return sha256.hexdigest(), git_sha1.hexdigest()


def verified_hash(path: Path, item: dict) -> str | None:
    if not path.is_file() or path.stat().st_size != item["size"]:
        return None
    sha256, git_sha1 = checksums(path)
    actual = sha256 if item["hash_kind"] == "sha256" else git_sha1
    return sha256 if actual == item["hash"] else None


def retry(operation, retries: int, delay: float = 1.0):
    for attempt in range(retries + 1):
        try:
            return operation()
        except HTTPError as e:
            if e.code not in (408, 429, 500, 502, 503, 504):
                raise DownloadError(f"HTTP {e.code}；检查网络、仓库权限和 revision。") from e
            hint = e.headers.get("Retry-After", "") if e.headers else ""
            wait_hint = min(float(hint), 60) if hint.isdigit() else 0
            label = f"HTTP {e.code}"
        except URLError as e:
            if isinstance(e.reason, ssl.SSLCertVerificationError):
                raise DownloadError("TLS 证书校验失败，请修复 Python 的证书环境；不要关闭证书校验。") from e
            wait_hint, label = 0, f"网络错误（{type(e.reason).__name__}）"
        except (TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException, RetryableError) as e:
            wait_hint, label = 0, type(e).__name__
        if attempt == retries:
            raise DownloadError(f"{label}，已用完 {retries} 次重试；残片已保留，稍后运行同一命令。")
        seconds = max(min(delay * 2 ** min(attempt, 6), 30), wait_hint)
        print(f"  {label}，{seconds:g} 秒后重试 ({attempt + 1}/{retries})。", flush=True)
        time.sleep(seconds)


def request(url: str, timeout: float, offset: int = 0):
    headers = {"User-Agent": "aa-lcr-downloader/1.0", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urlopen(Request(url, headers=headers), timeout=timeout)


def fetch_state(revision: str, timeout: float) -> dict:
    url = f"{ENDPOINT}/api/datasets/{REPO}/revision/{quote(revision, safe='')}?blobs=true"
    with request(url, timeout) as response:
        info = json.load(response)
    if not isinstance(info, dict):
        raise DownloadError("官方 API 未返回有效的仓库清单。")
    if re.fullmatch(r"[0-9a-f]{40}", revision) and info.get("sha") != revision:
        raise DownloadError("API 返回的 commit 与指定版本不一致，拒绝下载。")
    files = []
    for item in info.get("siblings", []):
        lfs = item.get("lfs")
        files.append({
            "path": item["rfilename"],
            "size": item.get("size", lfs.get("size") if lfs else None),
            "hash_kind": "sha256" if lfs else "git-sha1",
            "hash": lfs["sha256"] if lfs else item.get("blobId"),
        })
    return {"schema": 1, "repo_id": REPO, "requested_revision": revision,
            "commit": info.get("sha"), "files": sorted(files, key=lambda f: f["path"])}


def transfer(url: str, partial: Path, item: dict, timeout: float) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > item["size"]:
        partial.unlink()
        offset = 0
    try:
        response = request(url, timeout, offset)
    except HTTPError as e:
        if e.code == 416 and offset:
            partial.unlink()
            raise RetryableError("Range rejected; restart this partial") from e
        raise
    with response:
        status = response.status
        if status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or tuple(map(int, match.groups())) != (offset, item["size"] - 1, item["size"]):
                raise DownloadError("服务返回了不一致的 Content-Range；保留残片并停止，避免拼接错误数据。")
            mode = "ab"
        elif status == 200:
            if offset:
                print("  服务器未接受 Range，本文件从头下载，已完成的其他文件不受影响。", flush=True)
            offset, mode = 0, "wb"
        else:
            raise DownloadError(f"意外的 HTTP 状态 {status}。")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise DownloadError("服务器返回压缩传输，不能可靠进行字节续传。")
        count, last = offset, time.monotonic()
        print(f"  {'续传' if offset else '下载'} {item['path']}：{offset:,}/{item['size']:,} bytes", flush=True)
        with partial.open(mode) as f:
            try:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    if count + len(block) > item["size"]:
                        raise DownloadError("接收到的数据超过远端声明大小。")
                    f.write(block)
                    count += len(block)
                    if time.monotonic() - last >= 1:
                        print(f"    {count:,}/{item['size']:,} bytes", flush=True)
                        last = time.monotonic()
            finally:
                f.flush()
                os.fsync(f.fileno())
    if count != item["size"]:
        raise RetryableError("响应提前结束")


def download_file(root: Path, commit: str, item: dict, timeout: float, retries: int) -> str:
    target = safe_path(root, item["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = verified_hash(target, item)
    if digest:
        print(f"已校验，跳过：{item['path']}", flush=True)
        return digest
    if target.exists():
        print(f"检测到不完整/损坏文件，重新下载：{item['path']}", flush=True)
    partial = safe_path(root, f"{META}/partials/{item['path']}.part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ENDPOINT}/datasets/{REPO}/resolve/{commit}/{quote(item['path'], safe='/')}?download=true"

    def attempt():
        if partial.exists() and partial.stat().st_size == item["size"]:
            value = verified_hash(partial, item)
            if value:
                os.replace(partial, target)
                return value
            partial.unlink()
        transfer(url, partial, item, timeout)
        value = verified_hash(partial, item)
        if not value:
            partial.unlink()
            raise RetryableError("远端哈希校验失败，重新下载本文件")
        os.replace(partial, target)
        return value

    return retry(attempt, retries)


def verify(root: Path) -> None:
    state = read_json(root / META / "state.json")
    validate_state(state, root)
    manifest = read_json(root / META / "manifest.json")
    if (manifest.get("schema") != 1 or manifest.get("commit") != state["commit"]
            or manifest.get("repo_id") != REPO or manifest.get("complete") is not True):
        raise DownloadError("缺少匹配的完整下载清单。")
    records = manifest.get("sha256")
    if not isinstance(records, dict) or set(records) != {x["path"] for x in state["files"]}:
        raise DownloadError("校验清单文件集合与远端快照不一致。")
    for item in state["files"]:
        actual = verified_hash(safe_path(root, item["path"]), item)
        if not actual or actual != records[item["path"]]:
            raise DownloadError(f"校验失败：{item['path']}。重新运行下载命令可修复本地数据。")
    print(f"离线校验通过：{len(records)} 个文件；commit={state['commit']}")


@contextmanager
def directory_lock(root: Path):
    # macOS/Linux advisory lock, automatically released after crashes or Ctrl-C.
    try:
        import fcntl
    except ImportError as e:
        raise DownloadError("下载模式适用于 macOS/Linux；Windows 可使用 WSL。离线校验不受此限制。") from e
    folder = root / META
    if folder.is_symlink():
        raise DownloadError("下载状态目录不应是符号链接。")
    folder.mkdir(parents=True, exist_ok=True)
    with safe_path(root, f"{META}/download.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise DownloadError("另一个进程正在使用此目录，请等待其结束。") from e
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run(args) -> None:
    root = args.output.expanduser().resolve()
    if args.verify_only:
        verify(root)
        return
    root.mkdir(parents=True, exist_ok=True)
    with directory_lock(root):
        state_path = safe_path(root, f"{META}/state.json")
        if state_path.exists():
            state = read_json(state_path)
            validate_state(state, root)
            if args.revision and args.revision not in (state["commit"], state.get("requested_revision")):
                raise DownloadError("现有目录已锁定其他版本，请换一个 --output 目录。")
        else:
            revision = args.revision or DEFAULT_REVISION
            print(f"获取官方仓库清单：{REPO} @ {revision}", flush=True)
            state = retry(lambda: fetch_state(revision, args.timeout), args.retries)
            validate_state(state, root)
            atomic_json(state_path, state)
        print(f"固定版本：{state['commit']}；共 {len(state['files'])} 个文件，"
              f"{sum(x['size'] for x in state['files']):,} bytes。", flush=True)
        manifest_path = safe_path(root, f"{META}/manifest.json")
        # Remove the completion marker before repairs; original expected hashes stay in state.
        manifest_path.unlink(missing_ok=True)
        hashes = {}
        for item in state["files"]:
            hashes[item["path"]] = download_file(root, state["commit"], item, args.timeout, args.retries)
        atomic_json(manifest_path, {"schema": 1, "repo_id": REPO, "commit": state["commit"],
                                   "complete": True, "sha256": hashes})
        verify(root)
        print(f"下载完成：{root}\n上传时请保留 {META}/state.json 和 manifest.json。")


def main() -> int:
    parser = argparse.ArgumentParser(description="完整下载 AA-LCR，自动续传并校验；无需安装第三方库。")
    parser.add_argument("--output", type=Path, default=Path("AA-LCR"), help="输出目录，默认 ./AA-LCR")
    parser.add_argument("--revision", help=f"首次下载的版本；默认固定为 {DEFAULT_REVISION}。恢复沿用本地版本。")
    parser.add_argument("--retries", type=int, default=5, help="每项请求失败后最多重试次数，默认 5")
    parser.add_argument("--timeout", type=float, default=60, help="连接/读取超时秒数，默认 60")
    parser.add_argument("--verify-only", action="store_true", help="只做离线完整性校验，不联网")
    args = parser.parse_args()
    if args.retries < 0 or args.timeout <= 0:
        parser.error("retries 必须 >= 0，timeout 必须 > 0")
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        print("\n已中断，已有状态和残片会保留。运行同一命令继续下载。", file=sys.stderr)
        return 130
    except (DownloadError, OSError, ValueError, KeyError) as e:
        print(f"失败：{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
