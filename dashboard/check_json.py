"""Verify the local dashboard serves strict JSON (no bare NaN/Infinity).

JSON.parse in the browser rejects the NaN/Infinity literals that Python's json
module emits by default, so a single diverged run can break the whole page.
Run this on each login node after restarting a server.
"""
import json
import socket
import sys
import urllib.request


def strict(c):
    raise ValueError(f"bare {c!r} literal")


def main() -> int:
    host = socket.gethostname().split(".")[0]
    url = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else 8080}/api/runs"
    try:
        raw = urllib.request.urlopen(url, timeout=30).read().decode()
    except Exception as exc:  # noqa: BLE001 - report, don't traceback
        print(f"{host}: UNREACHABLE {url} -> {exc}")
        return 1
    try:
        data = json.loads(raw, parse_constant=strict)
    except ValueError as exc:
        print(f"{host}: BROKEN -> {exc}  ({len(raw)} bytes)")
        return 1
    # Nulls are mostly legitimate gaps (eval runs every N epochs), not just
    # sanitized NaNs -- the two are indistinguishable once serialized, so this
    # is only a rough "does the payload look sane" number.
    nulls = sum(v is None for r in data["runs"]
                for s in r.get("series", {}).values() if isinstance(s, list)
                for v in s)
    print(f"{host}: OK  {len(data['runs'])} runs, {len(raw)} bytes, "
          f"{nulls} null series entries (gaps + sanitized non-finite)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
