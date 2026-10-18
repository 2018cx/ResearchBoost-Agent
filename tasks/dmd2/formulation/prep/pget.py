#!/usr/bin/env python3
"""Parallel ranged HTTP download (stdlib only): pget.py URL DEST [--parts N] [--threads T] [--chunk MB].

The file is split into N parts (DEST.parts/NNN). Whatever is missing of every part is fetched as chunks of at most
--chunk MB by a pool of --threads connections (work stealing: slow connections only hold up one small chunk), then
appended to its part. Interrupted downloads resume from the part files."""
import argparse, os, queue, threading, time, urllib.request

UA = {'User-Agent': 'pget/2.0'}


def resolve(url):
    req = urllib.request.Request(url, method='HEAD', headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.geturl(), int(r.headers['Content-Length'])


def fetch_range(url, start, end, path, tries=40):
    for attempt in range(tries):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if start + have > end:
            return True
        req = urllib.request.Request(url, headers=dict(UA, Range=f'bytes={start + have}-{end}'))
        try:
            with urllib.request.urlopen(req, timeout=30) as r, open(path, 'ab') as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception:
            time.sleep(min(20, 1 + attempt))
    return os.path.getsize(path) == end - start + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('url'); p.add_argument('dest')
    p.add_argument('--parts', type=int, default=16)
    p.add_argument('--threads', type=int, default=32)
    p.add_argument('--chunk', type=int, default=32, help='MB')
    a = p.parse_args()
    if os.path.exists(a.dest):
        print('exists', a.dest); return
    url, size = resolve(a.url)
    d = a.dest + '.parts'
    os.makedirs(d, exist_ok=True)
    existing = sorted(f for f in os.listdir(d) if len(f) == 3 and f.isdigit())
    n = len(existing) if existing else (a.parts if size > (32 << 20) else 1)
    step = (size + n - 1) // n
    t0 = time.time()
    jobs, chunk = [], a.chunk << 20
    for i in range(n):
        s, e = i * step, min(size, (i + 1) * step) - 1
        part = f'{d}/{i:03d}'
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if not os.path.exists(part):
            open(part, 'wb').close()
        k = 0
        for cs in range(s + have, e + 1, chunk):
            jobs.append((i, k, cs, min(e, cs + chunk - 1), f'{part}.c{k:05d}'))
            k += 1
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    failed = []

    def worker():
        while True:
            try:
                i, k, cs, ce, path = q.get_nowait()
            except queue.Empty:
                return
            if not fetch_range(url, cs, ce, path):
                failed.append(path)
    th = [threading.Thread(target=worker) for _ in range(min(a.threads, max(1, len(jobs))))]
    for t in th: t.start()
    for t in th: t.join()
    if failed:
        raise SystemExit(f'{len(failed)} chunks failed, rerun to resume')
    for i in range(n):
        part = f'{d}/{i:03d}'
        subs = sorted(f for f in os.listdir(d) if f.startswith(f'{i:03d}.c'))
        with open(part, 'ab') as out:
            for sub in subs:
                with open(f'{d}/{sub}', 'rb') as f:
                    while True:
                        b = f.read(1 << 24)
                        if not b: break
                        out.write(b)
                os.remove(f'{d}/{sub}')
    with open(a.dest + '.tmp', 'wb') as out:
        for i in range(n):
            with open(f'{d}/{i:03d}', 'rb') as f:
                while True:
                    b = f.read(1 << 24)
                    if not b: break
                    out.write(b)
    if os.path.getsize(a.dest + '.tmp') != size:
        raise SystemExit(f'size mismatch {os.path.getsize(a.dest + ".tmp")} != {size}')
    os.rename(a.dest + '.tmp', a.dest)
    for i in range(n): os.remove(f'{d}/{i:03d}')
    os.rmdir(d)
    print(f'{a.dest}: {size/2**20:.0f} MiB in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
