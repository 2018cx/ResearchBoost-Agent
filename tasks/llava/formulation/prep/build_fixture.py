"""Fixture for the LLaVA-1.5 finetune bench: a seeded random subset of llava_v1_5_mix665k.json and only the images it
needs, pulled out of the public image zips on the Hugging Face hub by HTTP range requests (central directory + wanted members).

Sampling (seed 20261010): image-text samples uniformly at random from all 624,610 image samples (so the source mix
coco/gqa/ocr_vqa/textvqa/vg follows the 665k mix), text-only samples uniformly from the 40,688 ShareGPT samples.
  timing.json   18 x 128 image samples + 2 x 128 text samples (the sampler keeps one megabatch of each modality for
                the mixed tail, so the body is 17 image steps + 1 text step + the 2 mixed tail steps)
  case_mm.json  128 image samples (first 128 image samples of timing.json)  -> one optimizer step, image-text
  case_text.json 128 text samples (first 128 text samples of timing.json)   -> one optimizer step, text only
  probe.json    24 image + 8 text samples not in timing.json (held-out loss probe for trajectories)
"""
import concurrent.futures as cf, hashlib, json, os, random, struct, sys, threading, time, urllib.request, zlib, collections

A = os.environ['ASSETS']
SRC = f'{A}/data/llava_v1_5_mix665k.json'
OUT = f'{A}/data/fixture'
IMG = f'{A}/data/images'
CACHE = f'{A}/data/zipcache'
H = os.environ.get('HF_ENDPOINT', 'https://huggingface.co') + '/datasets'
ZIPS = {   # llava path prefix -> (zip url, member prefix inside the zip)
    'coco/train2017/': (f'{H}/pcuenq/coco-2017-mirror/resolve/main/train2017.zip', 'train2017/'),
    'gqa/images/': (f'{H}/Feeky929/GQA-images/resolve/main/images.zip', 'images/'),
    'textvqa/train_images/': (f'{H}/BoyangZ/text_vqa_train_val_images/resolve/main/train_val_images.zip', 'train_images/'),
    'vg/VG_100K/': (f'{H}/BoyangZ/VisualGenome_VG_100K_1_and_2/resolve/main/images.zip', 'VG_100K/'),
    'vg/VG_100K_2/': (f'{H}/BoyangZ/VisualGenome_VG_100K_1_and_2/resolve/main/images2.zip', 'VG_100K_2/'),
    'ocr_vqa/images/': (f'{H}/weizhiwang/llava_v15_instruction_images/resolve/main/ocr_vqa_images_llava_v15.zip', None),
}
SEED, MM_N, TXT_N, PROBE_MM, PROBE_TXT = 20261010, 18 * 128, 2 * 128, 24, 8


def get(url, a, b, tries=12):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b - 1}'})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            if len(data) == b - a:
                return data
        except Exception as e:
            err = e
        time.sleep(min(20, 2 * (t + 1)))
    raise IOError(f'{url} [{a},{b}) failed: {err}')


def head(url):
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
        return int(r.headers['Content-Length']), r.geturl()


def pget(url, a, b, part=2 << 20, workers=12):
    spans = [(x, min(b, x + part)) for x in range(a, b, part)]
    with cf.ThreadPoolExecutor(workers) as ex:
        return b''.join(ex.map(lambda s: get(url, *s), spans))


def central_directory(url, tag):
    os.makedirs(CACHE, exist_ok=True)
    path = f'{CACHE}/{tag}.cd'
    size, final = head(url)
    if not os.path.exists(path):
        tail = get(final, max(0, size - 131072), size)
        i = tail.rfind(b'PK\x05\x06')
        _, _, _, _, n, cd_size, cd_off, _ = struct.unpack('<IHHHHIIH', tail[i:i + 22])
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n == 0xFFFF:
            j = tail.rfind(b'PK\x06\x07')
            z64_off = struct.unpack('<Q', tail[j + 8:j + 16])[0]
            rec = get(final, z64_off, z64_off + 56)
            n, cd_size, cd_off = struct.unpack('<QQQ', rec[32:56])
        data = pget(final, cd_off, cd_off + cd_size)
        with open(path + '.tmp', 'wb') as f:
            f.write(data)
        os.replace(path + '.tmp', path)
    data = open(path, 'rb').read()
    entries, p = {}, 0
    while p + 46 <= len(data) and data[p:p + 4] == b'PK\x01\x02':
        method, = struct.unpack('<H', data[p + 10:p + 12])
        csize, usize, nlen, elen, clen = struct.unpack('<IIHHH', data[p + 20:p + 34])
        off, = struct.unpack('<I', data[p + 42:p + 46])
        name = data[p + 46:p + 46 + nlen].decode('utf-8', 'replace')
        extra = data[p + 46 + nlen:p + 46 + nlen + elen]
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack('<HH', extra[q:q + 4])
            if hid == 1:
                vals, r = extra[q + 4:q + 4 + hlen], 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack('<Q', vals[r:r + 8])[0]; r += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack('<Q', vals[r:r + 8])[0]; r += 8
                if off == 0xFFFFFFFF:
                    off = struct.unpack('<Q', vals[r:r + 8])[0]; r += 8
            q += 4 + hlen
        entries[name] = (method, csize, usize, off)
        p += 46 + nlen + elen + clen
    return final, entries


def member(url, ent):
    method, csize, usize, off = ent
    blob = get(url, off, off + 30 + 1024 + csize)
    assert blob[:4] == b'PK\x03\x04', 'bad local header'
    nlen, elen = struct.unpack('<HH', blob[26:30])
    raw = blob[30 + nlen + elen:30 + nlen + elen + csize]
    if method == 0:
        return raw
    if method == 8:
        return zlib.decompressobj(-15).decompress(raw)
    raise ValueError(f'compression method {method}')


def main():
    d = json.load(open(SRC))
    mm = [i for i, s in enumerate(d) if 'image' in s]
    tx = [i for i, s in enumerate(d) if 'image' not in s]
    rng = random.Random(SEED)
    mm_pick = rng.sample(mm, MM_N + PROBE_MM + 400)       # spares replace samples whose image is missing
    tx_pick = rng.sample(tx, TXT_N + PROBE_TXT)
    need = sorted({d[i]['image'] for i in mm_pick})
    by_prefix = collections.defaultdict(list)
    for p in need:
        pre = next(k for k in ZIPS if p.startswith(k))
        by_prefix[pre].append(p)
    have = set()
    os.makedirs(IMG, exist_ok=True)
    for pre, paths in sorted(by_prefix.items()):
        url, inner = ZIPS[pre]
        t0 = time.time()
        final, entries = central_directory(url, pre.strip('/').replace('/', '_'))
        index = {}
        for name, ent in entries.items():
            if name.endswith('/'):
                continue
            if inner is None or name.startswith(inner) or ('/' + inner) in name:
                index.setdefault(os.path.basename(name), ent)
        todo = [p for p in paths if not os.path.exists(f'{IMG}/{p}')]
        missing = [p for p in paths if os.path.basename(p) not in index]
        print(f'{pre}: {len(entries)} zip entries, need {len(paths)}, missing in zip {len(missing)}, '
              f'cd {time.time() - t0:.0f}s', flush=True)

        def fetch(p):
            if os.path.basename(p) not in index:
                return None
            data = member(final, index[os.path.basename(p)])
            os.makedirs(os.path.dirname(f'{IMG}/{p}'), exist_ok=True)
            with open(f'{IMG}/{p}.tmp', 'wb') as f:
                f.write(data)
            os.replace(f'{IMG}/{p}.tmp', f'{IMG}/{p}')
            return p
        t0 = time.time()
        with cf.ThreadPoolExecutor(16) as ex:
            for r in ex.map(fetch, todo):
                pass
        have |= {p for p in paths if os.path.exists(f'{IMG}/{p}')}
        print(f'  fetched {len(todo)} in {time.time() - t0:.0f}s', flush=True)
    ok = [i for i in mm_pick if d[i]['image'] in have]
    if len(ok) < MM_N + PROBE_MM:
        raise SystemExit(f'only {len(ok)} image samples have images')
    timing_mm, probe_mm = sorted(ok[:MM_N]), ok[MM_N:MM_N + PROBE_MM]
    timing_tx, probe_tx = sorted(tx_pick[:TXT_N]), tx_pick[TXT_N:]
    os.makedirs(OUT, exist_ok=True)
    sets = {
        'timing': sorted(timing_mm + timing_tx),
        'case_mm': timing_mm[:128],
        'case_text': timing_tx[:128],
        'probe': sorted(probe_mm + probe_tx),
    }
    manifest = {'source': SRC, 'source_sha256': 'ce959ce6e23073ee1cd1a8a2ef1c633768c10d4174327b8b2dc7113b91af6cf8',
                'seed': SEED, 'method': __doc__, 'zips': {k: v[0] for k, v in ZIPS.items()}, 'sets': {}}
    for name, idx in sets.items():
        rows = [d[i] for i in idx]
        text = json.dumps(rows)
        with open(f'{OUT}/{name}.json', 'w') as f:
            f.write(text)
        manifest['sets'][name] = {'n': len(rows), 'source_indices': idx,
                                  'sha256': hashlib.sha256(text.encode()).hexdigest(),
                                  'text_only': sum('image' not in r for r in rows)}
    used = sorted({d[i]['image'] for idx in sets.values() for i in idx if 'image' in d[i]})
    unused = sorted(have - set(used))
    for p in unused:
        os.remove(f'{IMG}/{p}')
    manifest['images'] = len(used)
    with open(f'{OUT}/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: (v['n'], v['text_only']) for k, v in manifest['sets'].items()}), 'images', len(used))


if __name__ == '__main__':
    main()
