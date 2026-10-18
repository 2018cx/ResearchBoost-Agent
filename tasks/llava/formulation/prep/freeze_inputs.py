"""Write the frozen-input digests into ws/task/formulation.json (same digest as bench.content_digest).
freeze_inputs.py name=path [name=path ...]   (paths absolute or relative to ws/); other keys of the file are kept."""
import os
import hashlib, json, sys
from pathlib import Path

WS = Path(os.environ['WORK']) / 'llava' / 'ws'


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def content_digest(p):
    p = Path(p)
    if p.is_file():
        return {'sha256': sha256_file(p)}
    h, n = hashlib.sha256(), 0
    for f in sorted(p.rglob('*')):
        if f.is_file():
            h.update(str(f.relative_to(p)).encode() + b'\0' + sha256_file(f).encode() + b'\n')
            n += 1
    return {'sha256': h.hexdigest(), 'files': n}


def main():
    f = WS / 'task' / 'formulation.json'
    data = json.loads(f.read_text()) if f.exists() else {}
    fi = data.setdefault('frozen_inputs', {})
    for arg in sys.argv[1:]:
        name, _, rel = arg.partition('=')
        target = Path(rel) if Path(rel).is_absolute() else WS / rel
        d = content_digest(target)
        fi[name] = {'path': rel, **d}
        print(name, d)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data, indent=1) + '\n')


if __name__ == '__main__':
    main()
