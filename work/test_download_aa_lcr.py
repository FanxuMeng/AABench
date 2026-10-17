"""Network-free regression tests for the delivered AA-LCR downloader."""
import argparse
from contextlib import redirect_stdout
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "outputs/aa_lcr_downloader/download_aa_lcr.py"
SPEC = importlib.util.spec_from_file_location("aa_lcr_downloader_under_test", SCRIPT)
dl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dl)
COMMIT = "a" * 40


class Response:
    def __init__(self, body=b"", status=200, headers=None, events=None):
        self.status = status
        self.headers = headers or {}
        self.stream = io.BytesIO(body)
        self.events = None if events is None else iter(events)

    def read(self, size=-1):
        if self.events is None:
            return self.stream.read(size)
        event = next(self.events, b"")
        if isinstance(event, BaseException):
            raise event
        return event

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def file_item(name, payload, kind="sha256"):
    digest = (hashlib.sha256(payload).hexdigest() if kind == "sha256" else
              hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest())
    return {"path": name, "size": len(payload), "hash_kind": kind, "hash": digest}


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.quiet = redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)
        self.sleep = patch.object(dl.time, "sleep", return_value=None)
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def args(self, revision=None, verify_only=False):
        return argparse.Namespace(output=self.root, revision=revision, verify_only=verify_only,
                                  timeout=1, retries=1)

    def fixture(self):
        payloads = {
            "README.md": b"readme\n",
            "AA-LCR_Dataset.csv": b"id,question\n1,example\n",
            "extracted_text/AA-LCR_extracted-text.zip": b"synthetic fixture, not a real ZIP",
        }
        state = {"schema": 1, "repo_id": dl.REPO, "requested_revision": "main",
                 "commit": COMMIT,
                 "files": [file_item(name, payload, "git-sha1" if name == "README.md" else "sha256")
                           for name, payload in sorted(payloads.items())]}
        return state, payloads

    def write_complete_fixture(self):
        state, payloads = self.fixture()
        (self.root / dl.META).mkdir(exist_ok=True)
        for name, payload in payloads.items():
            target = self.root / name
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_bytes(payload)
        dl.atomic_json(self.root / dl.META / "state.json", state)
        dl.atomic_json(self.root / dl.META / "manifest.json", {
            "schema": 1, "repo_id": dl.REPO, "commit": COMMIT, "complete": True,
            "sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
        })
        return state, payloads

    def test_range_resumes_at_last_written_offset_after_connection_drop(self):
        payload = b"0123456789"
        item = file_item("sample.bin", payload)
        offsets = []

        def request(url, timeout, offset=0):
            offsets.append(offset)
            if len(offsets) == 1:
                return Response(events=[payload[:4], ConnectionResetError("disconnected")])
            self.assertEqual(offset, 4)
            return Response(payload[4:], status=206, headers={"Content-Range": "bytes 4-9/10"})

        with patch.object(dl, "request", side_effect=request):
            digest = dl.download_file(self.root, COMMIT, item, 1, 1)
        self.assertEqual(offsets, [0, 4])
        self.assertEqual((self.root / "sample.bin").read_bytes(), payload)
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())

    def test_server_ignoring_range_restarts_instead_of_appending(self):
        payload = b"0123456789"
        partial = self.root / "sample.part"
        partial.write_bytes(payload[:4])
        with patch.object(dl, "request", return_value=Response(payload)) as request:
            dl.transfer("https://example.invalid/file", partial, file_item("sample.bin", payload), 1)
        request.assert_called_once_with("https://example.invalid/file", 1, 4)
        self.assertEqual(partial.read_bytes(), payload)

    def test_inconsistent_content_range_stops_without_altering_partial(self):
        partial = self.root / "sample.part"
        partial.write_bytes(b"0123")
        with patch.object(dl, "request", return_value=Response(
                b"56789", status=206, headers={"Content-Range": "bytes 5-9/10"})):
            with self.assertRaises(dl.DownloadError):
                dl.transfer("https://example.invalid/file", partial, file_item("sample", b"0123456789"), 1)
        self.assertEqual(partial.read_bytes(), b"0123")

    def test_same_size_corrupt_finished_file_is_repaired(self):
        payload = b"correct data"
        target = self.root / "sample.bin"
        target.write_bytes(b"x" * len(payload))
        with patch.object(dl, "request", return_value=Response(payload)) as request:
            dl.download_file(self.root, COMMIT, file_item("sample.bin", payload), 1, 0)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(target.read_bytes(), payload)

    def test_same_size_corrupt_partial_is_discarded_and_repaired(self):
        payload = b"correct data"
        partial = self.root / dl.META / "partials/sample.bin.part"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"x" * len(payload))
        with patch.object(dl, "request", return_value=Response(payload)) as request:
            dl.download_file(self.root, COMMIT, file_item("sample.bin", payload), 1, 0)
        self.assertEqual(request.call_args.args[2], 0)
        self.assertEqual((self.root / "sample.bin").read_bytes(), payload)
        self.assertFalse(partial.exists())

    def test_bad_remote_hash_retries_from_zero(self):
        payload = b"correct data"
        with patch.object(dl, "request", side_effect=[Response(b"x" * len(payload)), Response(payload)]) as request:
            dl.download_file(self.root, COMMIT, file_item("sample.bin", payload), 1, 1)
        self.assertEqual([c.args[2] for c in request.call_args_list], [0, 0])
        self.assertEqual((self.root / "sample.bin").read_bytes(), payload)

    def test_valid_finished_file_does_not_use_network(self):
        payload = b"valid"
        (self.root / "sample.bin").write_bytes(payload)
        with patch.object(dl, "request", side_effect=AssertionError("network must not be called")):
            result = dl.download_file(self.root, COMMIT, file_item("sample.bin", payload), 1, 0)
        self.assertEqual(result, hashlib.sha256(payload).hexdigest())

    def test_metadata_resolves_once_and_every_download_uses_fixed_commit(self):
        state, payloads = self.fixture()
        siblings = []
        for item in state["files"]:
            entry = {"rfilename": item["path"], "size": item["size"]}
            if item["hash_kind"] == "sha256":
                entry["lfs"] = {"sha256": item["hash"], "size": item["size"]}
            else:
                entry["blobId"] = item["hash"]
            siblings.append(entry)
        calls = []

        def request(url, timeout, offset=0):
            calls.append(url)
            if "/api/datasets/" in url:
                self.assertIn("/revision/main?blobs=true", url)
                return Response(json.dumps({"sha": COMMIT, "siblings": siblings}).encode())
            self.assertIn("/resolve/" + COMMIT + "/", url)
            name = url.split("/resolve/" + COMMIT + "/", 1)[1].split("?", 1)[0]
            return Response(payloads[name])

        with patch.object(dl, "request", side_effect=request):
            dl.run(self.args(revision="main"))
        self.assertEqual(sum("/api/datasets/" in url for url in calls), 1)
        self.assertEqual(len(calls), 4)
        with patch.object(dl, "request", side_effect=AssertionError("must not re-resolve main")):
            dl.run(self.args(revision="main"))
        self.assertEqual(dl.read_json(self.root / dl.META / "state.json")["commit"], COMMIT)

    def test_explicit_revision_conflict_is_rejected_before_network(self):
        self.write_complete_fixture()
        with patch.object(dl, "request", side_effect=AssertionError("no network")):
            with self.assertRaises(dl.DownloadError):
                dl.run(self.args(revision="b" * 40))

    def test_verify_is_offline(self):
        self.write_complete_fixture()
        with patch.object(dl, "request", side_effect=AssertionError("no network")):
            dl.run(self.args(verify_only=True))

    def test_verify_rejects_same_size_corruption_without_rewriting_manifest(self):
        self.write_complete_fixture()
        manifest = self.root / dl.META / "manifest.json"
        original_manifest = manifest.read_bytes()
        (self.root / "README.md").write_bytes(b"Xeadme\n")
        with patch.object(dl, "request", side_effect=AssertionError("no network")):
            with self.assertRaises(dl.DownloadError):
                dl.verify(self.root)
        self.assertEqual(manifest.read_bytes(), original_manifest)

    def test_verify_rejects_missing_file(self):
        self.write_complete_fixture()
        (self.root / "AA-LCR_Dataset.csv").unlink()
        with patch.object(dl, "request", side_effect=AssertionError("no network")):
            with self.assertRaises(dl.DownloadError):
                dl.verify(self.root)

    def test_directory_lock_rejects_competitor_and_releases_after_exception(self):
        with dl.directory_lock(self.root):
            with self.assertRaises(dl.DownloadError):
                with dl.directory_lock(self.root):
                    self.fail("second lock unexpectedly acquired")
        with self.assertRaises(RuntimeError):
            with dl.directory_lock(self.root):
                raise RuntimeError("simulated failure")
        with dl.directory_lock(self.root):
            pass

    def test_request_emits_range_and_identity_encoding(self):
        with patch.object(dl, "urlopen", return_value=Response(b"")) as urlopen:
            dl.request("https://example.invalid/file", 12, 7)
        req = urlopen.call_args.args[0]
        self.assertEqual(req.get_header("Range"), "bytes=7-")
        self.assertEqual(req.get_header("Accept-encoding"), "identity")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 12)

    def test_explicit_commit_must_match_api_response(self):
        with patch.object(dl, "request", return_value=Response(json.dumps({
                "sha": "b" * 40, "siblings": []}).encode())):
            with self.assertRaises(dl.DownloadError):
                dl.fetch_state(COMMIT, 1)

    def test_malformed_commit_or_hash_raise_controlled_errors(self):
        for field, value in [("commit", None), ("commit", 123), ("hash", None), ("hash", ["bad"])]:
            with self.subTest(field=field, value=value):
                state, _ = self.fixture()
                if field == "commit":
                    state[field] = value
                else:
                    state["files"][0][field] = value
                with self.assertRaises(dl.DownloadError):
                    dl.validate_state(state, self.root)

    def test_path_empty_components_and_traversal_are_rejected(self):
        for name in ("a//b", "a/", "/a", "a/../b", "./a", "../a"):
            with self.subTest(name=name):
                with self.assertRaises(dl.DownloadError):
                    dl.safe_path(self.root, name)

    def test_real_localhost_http_disconnect_and_range_resume(self):
        payload = bytes(range(256)) * 600
        cut = dl.CHUNK + 123
        ranges = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                supplied_range = self.headers.get("Range")
                ranges.append(supplied_range)
                if supplied_range is None:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload[:cut])
                    self.wfile.flush()
                    self.close_connection = True
                else:
                    start = int(supplied_range.removeprefix("bytes=").removesuffix("-"))
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
                    self.send_header("Content-Length", str(len(payload) - start))
                    self.end_headers()
                    self.wfile.write(payload[start:])

        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except OSError as error:
            self.skipTest(f"sandbox cannot bind localhost: {error}")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(dl, "ENDPOINT", f"http://127.0.0.1:{server.server_port}"):
                result = dl.download_file(self.root, COMMIT, file_item("sample.bin", payload), 2, 1)
            self.assertEqual(ranges, [None, f"bytes={cut}-"])
            self.assertEqual((self.root / "sample.bin").read_bytes(), payload)
            self.assertEqual(result, hashlib.sha256(payload).hexdigest())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
