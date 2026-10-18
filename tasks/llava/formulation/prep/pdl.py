"""Parallel HTTP range downloader: pdl.py URL OUT [--parts N] [--sha256 HEX]. Resumable per part."""
import argparse, hashlib, os, sys, threading, time, urllib.request

def head(url):
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length']), r.geturl()

def fetch(url, out, a, b, done_flag):
    if os.path.exists(done_flag):
        return
    for t in range(30):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b - 1}'})
            with urllib.request.urlopen(req, timeout=60) as r, open(out, 'r+b') as f:
                f.seek(a)
                pos = a
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    pos += len(chunk)
            if pos == b:
                open(done_flag, 'w').close()
                return
            a = pos                       # continue where the connection stopped
        except Exception as e:
            print(f'part {a}-{b} retry {t}: {e}', flush=True)
            time.sleep(min(30, 2 * (t + 1)))
    raise SystemExit(f'part {a}-{b} failed')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('url'); p.add_argument('out')
    p.add_argument('--parts', type=int, default=16)
    p.add_argument('--sha256')
    a = p.parse_args()
    if a.sha256 and os.path.exists(a.out + '.ok'):
        print('already ok', a.out); return
    size, final = head(a.url)
    part_dir = a.out + '.parts'
    os.makedirs(part_dir, exist_ok=True)
    if not os.path.exists(a.out) or os.path.getsize(a.out) != size:
        with open(a.out, 'wb') as f:
            f.truncate(size)
    n = a.parts
    step = (size + n - 1) // n
    t0 = time.time()
    ths = [threading.Thread(target=fetch, args=(final, a.out, i * step, min(size, (i + 1) * step), f'{part_dir}/{i}.done'))
           for i in range(n) if i * step < size]
    for t in ths: t.start()
    for t in ths: t.join()
    if len(os.listdir(part_dir)) != len(ths):
        raise SystemExit('incomplete')
    dt = time.time() - t0
    print(f'{a.out}: {size / 1e6:.0f} MB in {dt:.0f}s ({size / 1e6 / max(dt, 1):.1f} MB/s)', flush=True)
    if a.sha256:
        h = hashlib.sha256()
        with open(a.out, 'rb') as f:
            for c in iter(lambda: f.read(1 << 24), b''):
                h.update(c)
        if h.hexdigest() != a.sha256:
            raise SystemExit(f'sha256 mismatch {a.out}: {h.hexdigest()}')
        open(a.out + '.ok', 'w').close()
    for f in os.listdir(part_dir):
        os.remove(os.path.join(part_dir, f))
    os.rmdir(part_dir)

if __name__ == '__main__':
    main()
