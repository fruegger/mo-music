#!/usr/bin/env python3
"""Stream a big HTTP file to stdout, in order, fetching its chunks over several connections at once.

For servers that throttle each connection (Zenodo gives ~2-5 MB/s per connection) but accept Range requests.
Memory use is about workers * chunk size; nothing is written to disk, so it can feed `tar -xz` directly:

    python parallel_stream.py URL --workers 12 | tar -xz ...

Progress goes to stderr. A failed chunk is retried with a back-off; the stream never skips or reorders data."""

import argparse
import sys
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def total_size(url: str) -> int:
    """File size from a one-byte range request (Content-Range: bytes 0-0/<size>)."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status != 206:
            raise SystemExit(f"{url}: no Range support (status {r.status})")
        return int(r.headers["Content-Range"].rsplit("/", 1)[1])


def fetch(url: str, start: int, end: int, retries: int = 8) -> bytes:
    """Bytes `start..end` inclusive."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise OSError(f"status {r.status}")
                data = r.read()
            if len(data) != end - start + 1:
                raise OSError(f"short read: {len(data)} of {end - start + 1}")
            return data
        except Exception as e:  # noqa: BLE001 - any failure of one chunk is retried
            print(f"chunk at {start}: {e!r}; retry {attempt + 1}/{retries}", file=sys.stderr, flush=True)
            time.sleep(min(30, 2**attempt))
    raise SystemExit(f"chunk at {start} failed {retries} times")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--workers", type=int, default=12, help="parallel connections")
    p.add_argument("--chunk-mb", type=int, default=64)
    p.add_argument("--start", type=int, default=0, help="first byte (testing)")
    p.add_argument("--end", type=int, help="last byte, inclusive (testing); default: the end of the file")
    args = p.parse_args()

    size = total_size(args.url)
    first, last = args.start, min(args.end if args.end is not None else size - 1, size - 1)
    chunk = args.chunk_mb * 1024 * 1024
    ranges = [(s, min(s + chunk - 1, last)) for s in range(first, last + 1, chunk)]
    total = last - first + 1
    out = sys.stdout.buffer
    done, t0, t_report = 0, time.time(), 0.0

    with ThreadPoolExecutor(args.workers) as pool:
        pending: deque = deque()
        todo = iter(ranges)
        for _ in range(args.workers):  # keep `workers` chunks in flight
            r = next(todo, None)
            if r:
                pending.append(pool.submit(fetch, args.url, *r))
        try:
            while pending:
                data = pending.popleft().result()
                r = next(todo, None)
                if r:
                    pending.append(pool.submit(fetch, args.url, *r))
                out.write(data)
                done += len(data)
                if time.time() - t_report > 20:
                    t_report = time.time()
                    rate = done / (t_report - t0)
                    eta = (total - done) / rate if rate else 0
                    print(f"{done / 1e9:.1f}/{total / 1e9:.1f} GB ({100 * done / total:.0f}%), "
                          f"{rate / 1e6:.1f} MB/s, ETA {eta / 60:.0f} min", file=sys.stderr, flush=True)
            out.flush()
        except BrokenPipeError:  # the consumer (tar) went away; its own error says why
            sys.exit(1)
    print(f"done: {done / 1e9:.1f} GB in {(time.time() - t0) / 60:.1f} min", file=sys.stderr)


if __name__ == "__main__":
    main()
