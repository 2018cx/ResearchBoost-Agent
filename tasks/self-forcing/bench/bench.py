"""Self-Forcing DMD training bench: one command per question.

Run from the workspace root with the self_forcing Python. GPU phases take /tmp/sf_gpu.lock themselves, one at a
time; do NOT wrap these commands in flock (that would deadlock).

  python bench/bench.py check   --tree <dir> --out <dir> [--config <file>] [--quick] [--skip-timing]
        full verdict (gates 0-3 below) -> <out>/check.json; --quick: gate 0 only; --skip-timing: no gate 1
  python bench/bench.py time    --tree <dir> --out <dir> [--config <file>]
        iteration timing of one tree, no comparison -> <out>/time.json
  python bench/bench.py profile --tree <dir> --out <dir> [--config <file>]
        Nsight Systems trace of one timing window (2 update cycles) + kernel/NVTX summaries -> <out>/profile.json
  python bench/bench.py baseline --out <dir>          (formulator) reference against itself: noise floor

Every GPU run holds the GPU lock, waits until the visible GPUs are empty and idle (INVALID after 10 minutes),
runs in its own process group with a timeout (bench.json timeouts_s; TIMEOUT kills the group) and a fresh short
TMPDIR, and records the GPU state after. Every timing run gets a fresh compile cache (cold compile, nothing shared
between runs), is scanned for the invalid-run signatures in bench.json, fails if torch.compile compiled anything
inside a timing window, and carries a fingerprint. Outputs carry cost.gpu_hours. Every command checks the frozen
inputs declared in task/formulation.json at start (mismatch: nothing is measured) and at end (mismatch: ERROR),
and the source digest of the tree at start and end (the tree changed during the run: ERROR).

Candidate switches: a JSON file {"environment": {"NAME": "value"}} given with --config; the bench sets those
variables for the candidate's processes only. The caller's environment is not consulted.

Verdict (check); each gate stops the evaluation when it fails:
  0. structure  train.py must equal the reference AST and configs/ must be byte-identical (the bench builds the
                trainer from the frozen configs exactly as train.py does). A 4-iteration case (generator iteration,
                critic-only iteration, EMA creation, EMA update) is recorded in deterministic mode with the
                candidate's own trainer; exact checks (iteration kind, step counter, prompts, RNG fingerprints,
                optimizer step counts, learning rates, top-level forward counts of generator / fake_score /
                real_score / text_encoder, cross-rank pattern, tensor names and shapes) must hold, else INCORRECT.
  1. benefit    reference, candidate, reference: fresh torchruns of the real entrypoint train.py on the timing
                config; fingerprints must agree (else ERROR). speedup = mean reference ms/iteration over candidate
                ms/iteration; it must exceed max(min_speedup, 1 + 2 x reference spread), else UNAVAILABLE
                (no_benefit) without a precision evaluation.
  2. bitwise    every tensor of the gate-0 record bitwise identical to the reference -> PASS (N0).
  3. statistical  paired-seed training trajectories (losses only, upstream config from ode_init, deterministic
                kernels) on the frozen seeds, 3 then 6. delta = 2 x reference seed sigma per endpoint. With 3 seeds:
                an endpoint beyond delta and significant -> INCORRECT, all within delta -> PASS, otherwise 6 seeds;
                with 6 seeds any endpoint beyond delta -> INCORRECT, else PASS.
  Per-tensor tolerance gates are reported as diagnostics only.
"""
import argparse
import ast
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent                       # workspace
CFG = json.loads((BENCH / 'bench.json').read_text())
LOCK = Path(CFG.get('gpu_lock', '/tmp/sf_gpu.lock'))
PY = Path(sys.executable)
TORCHRUN = PY.parent / 'torchrun'
TIERS = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5']
SCRATCH = '/tmp'
TIMEOUT = {'timing': 2400, 'numerics': 3600, 'trajectory': 10800, 'profile': 2400, **CFG.get('timeouts_s', {})}
RESERVED = {'TMPDIR', 'TMP', 'TEMP', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH',
            'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG',
            'MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK'}
SIGNATURES = [(re.compile(x['regex']), x['reason']) for x in CFG.get('invalid_run_signatures', [])]
COST = {'gpu_seconds': 0.0, 'gpu_runs': 0}
TREE_SKIP = {'.git', '__pycache__', 'wan_models', 'checkpoints', 'videos', 'images', 'logs', 'outputs', 'wandb'}


def path(key):
    p = Path(key)
    return p if p.is_absolute() else (ROOT / p).resolve()


def log(msg):
    print(f'[bench {datetime.now():%H:%M:%S}] {msg}', flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def evaluator_sha256():
    h = hashlib.sha256()
    for d in ('bench', 'numerics'):
        for f in sorted((ROOT / d).rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py', '.json', '.yaml'):
                if d == 'bench' and f.parts[-2] in ('records', 'ref'):
                    continue
                h.update(str(f.relative_to(ROOT)).encode())
                h.update(f.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- GPU runs

class GpuLock:
    def __init__(self, what, timeout=6 * 3600):
        self.what, self.timeout = what, timeout

    def __enter__(self):
        self.f = open(LOCK, 'a+')
        t0 = time.time()
        while True:
            try:
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t0 > self.timeout:
                    raise SystemExit(f'could not get {LOCK} within {self.timeout}s for {self.what}')
                time.sleep(10)
        self.t0 = time.time()
        log(f'GPU lock held: {self.what}')
        return self

    def __exit__(self, *_):
        COST['gpu_seconds'] += (time.time() - self.t0) * CFG['world_size']
        COST['gpu_runs'] += 1
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def cost():
    return {'gpu_hours': round(COST['gpu_seconds'] / 3600, 3), 'gpu_runs': COST['gpu_runs']}


def scratch_dir(prefix):
    free = shutil.disk_usage(SCRATCH).free / 2**30
    if free < CFG.get('min_scratch_free_gib', 3):
        raise SystemExit(f'{SCRATCH} has only {free:.1f} GiB free (< {CFG.get("min_scratch_free_gib", 3)}); '
                         f'clean up before measuring')
    return tempfile.TemporaryDirectory(prefix=prefix, dir=SCRATCH)


def gpu_state():
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True)
    out = []
    for line in q.stdout.strip().splitlines():
        idx, uuid, mem, util = [x.strip() for x in line.split(',')]
        out.append({'index': int(idx), 'uuid': uuid, 'memory_mib': int(mem), 'util': int(util)})
    return out


def wait_idle(what, patience=600):
    t0 = time.time()
    while True:
        busy = [g for g in gpu_state() if g['memory_mib'] > 1024 or g['util'] > 5]
        if not busy:
            return None
        if time.time() - t0 > patience:
            return f'GPUs busy for {patience}s before {what}: {busy}'
        time.sleep(15)


def run_gpu(cmd, cwd, env, log_file, what, timeout):
    with GpuLock(what), scratch_dir('sf-') as tmp:
        problem = wait_idle(what)
        if problem:
            return {'status': 'INVALID', 'reason': problem}
        before = gpu_state()
        env = dict(env, TMPDIR=tmp)
        t0 = time.time()
        with open(log_file, 'w') as lf:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                rc, status = proc.wait(timeout=timeout), None
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                rc, status = None, 'TIMEOUT'
        # whatever the run left behind in its process group goes too
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        wall = time.time() - t0
        time.sleep(3)
        after = gpu_state()
        res = {'status': status or 'ok', 'returncode': rc, 'wall_s': round(wall, 1), 'gpus_before': before}
        if status == 'TIMEOUT':
            res['reason'] = f'{what} exceeded {timeout}s'
        leaked = [g for g in after if g['memory_mib'] > 1024]
        if leaked:
            res['warning'] = f'GPU memory still in use after {what}: {leaked}'
        return res


def signatures(log_file):
    text = Path(log_file).read_text(errors='replace') if Path(log_file).exists() else ''
    return [{'regex': rx.pattern, 'reason': reason} for rx, reason in SIGNATURES if rx.search(text)]


# ---------------------------------------------------------------- inputs and sources

def tree_digest(tree):
    h = hashlib.sha256()
    tree = Path(tree)
    for f in sorted(tree.rglob('*')):
        rel = f.relative_to(tree)
        if any(part in TREE_SKIP for part in rel.parts) or f.is_symlink() or not f.is_file():
            continue
        h.update(str(rel).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def input_fingerprint(p, kind):
    p = path(p)
    if not p.exists():
        return None
    if kind == 'sha256':
        return sha256_file(p)
    st = p.stat()
    return f'{st.st_size}:{st.st_mtime_ns}'


def verify_inputs(when):
    spec = json.loads((ROOT / 'task' / 'formulation.json').read_text())['inputs']
    bad = {}
    for rel, want in spec.items():
        got = input_fingerprint(rel, want['kind'])
        if got != want['value']:
            bad[rel] = {'want': want['value'], 'got': got}
    return {'when': when, 'ok': not bad, 'mismatches': bad, 'checked': len(spec)}


def check_reference():
    ref = path(CFG['reference'])
    dirty = subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(ref), 'status', '--porcelain',
                            '--untracked-files=no'], capture_output=True, text=True).stdout.strip()
    head = subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(ref), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True).stdout.strip()
    if dirty or head != CFG['reference_commit']:
        raise SystemExit(f'reference tree is not the frozen upstream commit (head {head}, dirty: {bool(dirty)})')
    return ref


def link_assets(tree):
    """The training code reads wan_models/, checkpoints/ and prompts/ relative to its working directory."""
    tree = Path(tree)
    for rel, target in CFG['asset_links'].items():
        link = tree / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            if link.is_symlink() and os.readlink(link) == str(path(target)):
                continue
            raise SystemExit(f'{link} exists and is not the asset link to {path(target)}')
        link.symlink_to(path(target))


def materialize(src, dest):
    def ignore(directory, names):
        return [n for n in names if n in TREE_SKIP or n.endswith('.pyc')]
    shutil.copytree(src, dest, ignore=ignore, symlinks=False)
    link_assets(dest)
    return dest


def candidate_environment(config_file):
    if not config_file:
        return {}
    data = json.loads(Path(config_file).read_text())
    env = {str(k): str(v) for k, v in (data.get('environment') or {}).items()}
    bad = sorted(set(env) & RESERVED)
    if bad:
        raise SystemExit(f'candidate config sets variables the bench manages itself: {bad}')
    return env


def versions():
    out = {'python': sys.version.split()[0]}
    for pkg in ('torch', 'flash_attn', 'triton', 'diffusers', 'transformers'):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def fingerprint_diff(a, b):
    if a is None or b is None:
        return ['missing fingerprint']
    keys = ('math_mode', 'gpus', 'compile_cache', 'instrumentation', 'versions', 'timing_plan')
    return [k for k in keys if a.get(k) != b.get(k)]


# ---------------------------------------------------------------- gate 0: structure guard

def structure_guard(ref, tree):
    problems = []
    try:
        want = ast.dump(ast.parse((Path(ref) / 'train.py').read_text()), include_attributes=False)
        got = ast.dump(ast.parse((Path(tree) / 'train.py').read_text()), include_attributes=False)
        if want != got:
            problems.append('train.py differs from the reference (the bench builds the trainer exactly as its main() '
                            'does; put setup changes into the modules the trainer calls)')
    except (OSError, SyntaxError) as exc:
        problems.append(f'train.py unreadable: {exc}')
    for f in sorted((Path(ref) / 'configs').rglob('*')):
        if f.is_file():
            g = Path(tree) / f.relative_to(ref)
            if not g.exists() or g.read_bytes() != f.read_bytes():
                problems.append(f'{f.relative_to(ref)} differs from the reference (configs are fixed)')
    extra = [str(f.relative_to(tree)) for f in (Path(tree) / 'configs').rglob('*')
             if f.is_file() and not (Path(ref) / f.relative_to(tree)).exists()]
    if extra:
        problems.append(f'files added under configs/: {extra}')
    return problems


# ---------------------------------------------------------------- gate 0/2: short record

def numerics(tree, out, cand_env):
    n = CFG['numerics']
    kwargs = {'repo': str(tree), 'config': str(path(n['config']))}
    rec = out / 'records'
    shutil.rmtree(rec, ignore_errors=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), **cand_env)
    cmd = [str(TORCHRUN), '--standalone', '--nnodes=1', f'--nproc-per-node={CFG["world_size"]}',
           '-m', 'numerics', 'record', '--adapter', n['adapter'], '--adapter-kwargs', json.dumps(kwargs),
           '--cases', str(path(n['cases'])), '--sketch-dim', str(n['sketch_dim']),
           '--math-mode', CFG['math_mode'], '--deterministic', '--out', str(rec)]
    r = run_gpu(cmd, ROOT, env, out / 'record.log', 'numerics record', TIMEOUT['numerics'])
    if r['status'] != 'ok' or r['returncode']:
        return {'status': 'ERROR', 'reason': r.get('reason') or f'record exited {r["returncode"]}; see record.log'}
    report = out / 'compare.json'
    r = subprocess.run([str(PY), '-m', 'numerics', 'compare', '--ref', str(path(n['ref'])), '--cand', str(rec),
                        '--thresholds', str(path(n['thresholds'])),
                        *[a for x in n.get('exclude', []) for a in ('--exclude', f'{x["regex"]}={x["reason"]}')],
                        '--out', str(report)], cwd=ROOT, env=env, capture_output=True, text=True)
    (out / 'compare.log').write_text(r.stdout + r.stderr)
    if not report.exists():
        return {'status': 'ERROR', 'reason': f'compare exited {r.returncode}; see compare.log'}
    data = json.loads(report.read_text())
    if not n.get('keep_records'):
        shutil.rmtree(rec, ignore_errors=True)
    summary = {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in data.items()
               if k not in ('status', 'measured_tier')}
    bitwise = (data.get('checked') == data.get('bitwise_identical') and not data.get('dtype_changes')
               and not data.get('precision_mode_changes') and data.get('bitwise_trusted', False))
    return {'status': data.get('status'), 'measured_tier': data.get('measured_tier'), 'report': str(report),
            'errors': len(data.get('errors') or []), 'exact_failures': data.get('exact_failures') or [],
            'bitwise': bitwise, 'summary': summary}


# ---------------------------------------------------------------- gate 3: trajectories

def trajectories(tree, out, cand_env):
    tj = CFG['trajectory']
    spec = json.loads(path(tj['spec']).read_text())
    frozen_path = path(tj['frozen'])
    seeds = json.loads(frozen_path.read_text())['reference_seeds']
    kwargs = {'repo': str(tree), 'config': str(path(CFG['numerics']['config']))}
    rec, done, report = out / 'trajectory', [], None
    env = dict(os.environ, PYTHONPATH=str(ROOT), **cand_env)
    schedule = tj['seed_schedule']
    for idx, n in enumerate(schedule):
        final = idx == len(schedule) - 1
        new = [x for x in seeds[:n] if x not in done]
        for seed in new:
            # one process per seed: a second trainer built in the same process does not get all GPU memory back
            cases = out / f'traj_cases_s{seed}.json'
            cases.write_text(json.dumps([dict(spec['case'], name=f'traj_s{seed}', seed=seed)]))
            cmd = [str(TORCHRUN), '--standalone', '--nnodes=1', f'--nproc-per-node={CFG["world_size"]}',
                   '-m', 'numerics', 'record', '--adapter', CFG['numerics']['adapter'], '--adapter-kwargs',
                   json.dumps(kwargs), '--cases', str(cases), '--losses-only', '--math-mode', CFG['math_mode'],
                   '--deterministic', '--out', str(rec)]
            r = run_gpu(cmd, ROOT, env, out / f'traj_s{seed}.log', f'trajectory: seed {seed}', TIMEOUT['trajectory'])
            if r['status'] != 'ok' or r['returncode']:
                return {'verdict': 'error',
                        'reason': r.get('reason') or f'trajectory record exited {r["returncode"]}; see traj_s{seed}.log'}
            done.append(seed)
        report_path = out / f'traj_report_{n}.json'
        subprocess.run([str(PY), '-m', 'numerics', 'traj-compare', '--frozen', str(frozen_path), '--cand', str(rec),
                        *(['--final'] if final else []), '--out', str(report_path)],
                       cwd=ROOT, env=env, capture_output=True, text=True)
        if not report_path.exists():
            return {'verdict': 'error', 'reason': 'traj-compare produced no report'}
        report = json.loads(report_path.read_text())
        report['report'] = str(report_path)
        if report['verdict'] != 'inconclusive':
            break
    return report


# ---------------------------------------------------------------- gate 1: timing

def time_tree(tree, out, label, cand_env=None, profile=False):
    t = CFG['timing']
    out.mkdir(parents=True, exist_ok=True)
    scratch = scratch_dir('sf-t-')
    results = Path(scratch.name)
    prefix = out / f'timing_{label}'
    argv = [str(TORCHRUN), '--standalone', '--nnodes=1', f'--nproc-per-node={CFG["world_size"]}',
            str(BENCH / 'timing.py'), '--out', str(prefix), '--warmup', str(t['warmup']), '--window', str(t['window']),
            '--repeats', str(t['repeats']), *(['--profile'] if profile else []),
            '--', '--config_path', str(path(t['config'])), '--logdir', str(results / 'logdir'),
            '--no_save', '--no_visualize', '--disable-wandb']
    caches = {k: str(results / v) for k, v in (('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton'),
                                                ('CUDA_CACHE_PATH', 'cuda'))}
    for v in caches.values():
        Path(v).mkdir()
    env = dict(os.environ, **caches, **(cand_env or {}))
    env.pop('PYTHONPATH', None)
    if CFG['math_mode'] == 'default':
        env.pop('NVIDIA_TF32_OVERRIDE', None)
    if profile:
        nsys = shutil.which('nsys') or CFG.get('nsys')
        if not nsys or not Path(nsys).exists():
            raise SystemExit('nsys not found; source $WORK/opt/nsight/env.sh first')
        argv = [nsys, 'profile', '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                '--trace=cuda,nvtx,osrt,cublas,cudnn', '--sample=none', '--cuda-memory-usage=false',
                '--force-overwrite=true', '-o', str(out / f'nsys_{label}'), *argv]
    for f in Path(out).glob(f'timing_{label}.r*.json'):
        f.unlink()
    log_file = out / f'timing_{label}.log'
    what = f'timing {label}{" (nsys)" if profile else ""}'
    try:
        r = run_gpu(argv, tree, env, log_file, what, TIMEOUT['profile' if profile else 'timing'])
    finally:
        scratch.cleanup()
    fingerprint = {'argv': argv[argv.index(str(TORCHRUN)):], 'candidate_environment': cand_env or {},
                   'math_mode': CFG['math_mode'], 'gpus': [(g['index'], g['uuid']) for g in r.get('gpus_before') or []],
                   'compile_cache': 'cold (fresh per run)', 'instrumentation': 'nsys' if profile else 'none',
                   'timing_plan': [t['warmup'], t['window'], t['repeats']], 'versions': versions()}
    run = {'run_status': r['status'], 'wall_s': r.get('wall_s'), 'fingerprint': fingerprint,
           **({'warning': r['warning']} if r.get('warning') else {})}
    if r['status'] != 'ok':
        return {'status': r['status'], 'reason': r['reason'], **run}
    ranks = []
    for rank in range(CFG['world_size']):
        f = Path(f'{prefix}.r{rank}.json')
        ranks.append(json.loads(f.read_text()) if f.exists() else None)
    if r['returncode'] or any(x is None or x['status'] != 'ok' for x in ranks):
        errors = [x['error'] for x in ranks if x and x.get('error')]
        return {'status': 'ERROR', 'reason': f'timing exited {r["returncode"]}; {errors[:1]}; see {log_file.name}',
                **run}
    hits = signatures(log_file)
    if hits:
        return {'status': 'INVALID', 'reason': 'invalid-run signature in the log: ' + hits[0]['reason'],
                'signatures': hits, **run}
    return {**summarize_timing(ranks, r['wall_s'], 1 if profile else t['repeats']), **run}


def summarize_timing(ranks, wall, repeats):
    t = CFG['timing']
    ratio = CFG['update_ratio']
    per_window = {}
    for x in ranks:
        for w in x['windows']:
            prev = per_window.get(w['k'])
            if prev is None or w['seconds'] > prev['seconds']:
                per_window[w['k']] = w
    problems = []
    ws = [per_window[k] for k in range(repeats) if k in per_window]
    if len(ws) != repeats:
        problems.append(f'{len(ws)} of {repeats} windows')
    for w in ws:
        if w.get('compiles_in_window'):
            problems.append(f'window {w["k"]}: torch.compile compiled {w["compiles_in_window"]} graphs inside the '
                            f'timing window (still compiling in steady state)')
        expect = t['window'] // ratio
        if w['generator_steps'] != expect or w['kinds'].count('gen') != expect:
            problems.append(f'window {w["k"]}: {w["generator_steps"]} generator steps, expected {expect} '
                            f'(update schedule or step structure changed)')
    ms = [1000 * w['seconds'] / w['iterations'] for w in ws]
    med = statistics.median(ms) if ms else None
    return {
        'status': 'ok' if not problems and ms else 'ERROR', 'problems': problems,
        'ms_per_iteration_windows': [round(v, 1) for v in ms], 'ms_per_iteration': round(med, 1) if med else None,
        'window_spread': round((max(ms) - min(ms)) / med, 4) if ms else None,
        'ms_per_cycle': round(med * ratio, 1) if med else None,
        'peak_memory_allocated_gib': round(max(x['peak_memory_allocated_bytes'] for x in ranks) / 2**30, 2),
        'peak_memory_reserved_gib': round(max(x['peak_memory_reserved_bytes'] for x in ranks) / 2**30, 2),
        'startup_to_first_iteration_s': round(max(x['startup_to_first_iteration_s'] or 0 for x in ranks), 1),
        'process_wall_s': round(wall, 1),
    }


def compare_timing(ref_a, cand, ref_b):
    ra, rb, c = ref_a['ms_per_iteration'], ref_b['ms_per_iteration'], cand['ms_per_iteration']
    ref = (ra + rb) / 2
    return {'reference_ms_per_iteration': round(ref, 1), 'candidate_ms_per_iteration': c,
            'speedup': round(ref / c, 4), 'reference_spread': round(abs(ra - rb) / ref, 4)}


def benefit(tree, out, cand_env):
    ref_src = check_reference()
    with scratch_dir('sf-ref-') as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / 'tree')
        ref_a = time_tree(ref_tree, out, 'ref_a')
        cand = time_tree(tree, out, 'cand', cand_env=cand_env) if ref_a['status'] == 'ok' else None
        ref_b = time_tree(ref_tree, out, 'ref_b') if cand and cand['status'] == 'ok' else None
    runs = {'ref_a': ref_a, 'cand': cand, 'ref_b': ref_b}
    if any(v is None or v['status'] != 'ok' for v in runs.values()):
        return None, {'speedup': None, 'runs': runs}
    diffs = {k: fingerprint_diff(ref_a.get('fingerprint'), v.get('fingerprint')) for k, v in runs.items() if k != 'ref_a'}
    bad = {k: d for k, d in diffs.items() if d}
    if bad:
        return None, {'speedup': None, 'runs': runs, 'fingerprint_diffs': diffs,
                      'reason': f'timing runs not comparable: fingerprint differs from ref_a in {bad}'}
    perf = {**compare_timing(ref_a, cand, ref_b), 'runs': runs}
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * perf['reference_spread'])
    perf['benefit'] = {'speedup': perf['speedup'], 'speedup_bar': round(bar, 4),
                       'peak_memory_allocated_gib': cand['peak_memory_allocated_gib'],
                       'reference_peak_memory_allocated_gib': round((ref_a['peak_memory_allocated_gib'] +
                                                                     ref_b['peak_memory_allocated_gib']) / 2, 2),
                       'passed': perf['speedup'] > bar}
    return perf['benefit']['passed'], perf


# ---------------------------------------------------------------- verdict

def evaluate(tree, out, skip_timing=False, cand_env=None, quick=False):
    cand_env = cand_env or {}
    problems = structure_guard(check_reference(), tree)
    if problems:
        return 'ERROR', {'passed': False, 'stage': 'structure_guard', 'problems': problems}, None
    if quick:
        num = numerics(tree, out, cand_env)
        if num.get('errors') or num['status'] not in ('PASS', 'INCORRECT'):
            return 'ERROR', {'passed': False, 'stage': 'structure', 'numerics': num}, None
        if num['exact_failures']:
            return 'INCORRECT', {'passed': False, 'stage': 'structure', 'verdict': 'structure', 'numerics': num}, None
        return 'PASS', {'passed': True, 'stage': 'structure (quick self-test: gates 1-3 not run)',
                        'bitwise': num['bitwise'], 'measured_tier': num.get('measured_tier')}, None
    performance = None
    if not skip_timing:
        ok, performance = benefit(tree, out, cand_env)
        if ok is None:
            return 'ERROR', {'passed': False, 'stage': 'benefit',
                             'note': (performance or {}).get('reason') or 'timing failed'}, performance
        if not ok:
            performance['measured_speedup'] = performance.pop('speedup')
            return 'UNAVAILABLE', {'passed': None, 'stage': 'benefit', 'verdict': 'no_benefit',
                                   'note': 'not faster than the reference: rejected without a precision evaluation'}, \
                performance
    num = numerics(tree, out, cand_env)
    tier = num.get('measured_tier')
    correctness = {'stage': 'bitwise', 'measured_tier': tier, 'numerics': num}
    if num.get('errors') or num['status'] not in ('PASS', 'INCORRECT'):
        return 'ERROR', dict(correctness, passed=False, note='numerics could not compare the records'), performance
    if num['exact_failures']:
        return 'INCORRECT', dict(correctness, passed=False, verdict='structure',
                                 note='exact checks failed: the candidate does different work'), performance
    if tier not in TIERS or TIERS.index(tier) > TIERS.index(CFG['agent_max_tier']):
        return 'INCORRECT', dict(correctness, passed=False, verdict='needs_signoff',
                                 note=f'measured tier {tier} exceeds {CFG["agent_max_tier"]}'), performance
    if num['bitwise']:
        return 'PASS', dict(correctness, passed=True, verdict='bitwise'), performance
    traj = trajectories(tree, out, cand_env)
    correctness.update(stage='statistical', trajectory=traj)
    if traj['verdict'] == 'equivalent':
        return 'PASS', dict(correctness, passed=True, verdict='equivalent'), performance
    if traj['verdict'] == 'error':
        return 'ERROR', dict(correctness, passed=False, verdict='error'), performance
    note = ('an endpoint deviates from the reference beyond 2 x its seed spread' if traj['verdict'] == 'different'
            else 'no verdict after the full seed schedule')
    return 'INCORRECT', dict(correctness, passed=False, verdict=traj['verdict'], note=note), performance


def guarded(out, fn, tree=None):
    """Input check before (nothing measured on a mismatch) and after, source digest before and after."""
    out.mkdir(parents=True, exist_ok=True)
    start = verify_inputs('start')
    if not start['ok']:
        res = {'status': 'ERROR', 'stage': 'inputs', 'inputs': [start],
               'note': 'a frozen input differs from task/formulation.json; nothing was measured'}
        return res
    digest = tree_digest(tree) if tree else None
    res = fn()
    end = verify_inputs('end')
    res['inputs'] = [start, end]
    if not end['ok']:
        res['status'] = 'ERROR'
        res['note'] = 'a frozen input changed during the run'
    if tree:
        res['source_digest'] = digest
        if tree_digest(tree) != digest:
            res['status'] = 'ERROR'
            res['note'] = 'the tree changed while it was being measured'
    return res


def finish(out, name, res):
    res['cost'] = cost()
    res['evaluator_sha256'] = evaluator_sha256()
    res['finished'] = datetime.now().isoformat(timespec='seconds')
    (out / name).write_text(json.dumps(res, indent=1) + '\n')
    log(f'wrote {out / name}: status {res.get("status")}')
    print(json.dumps({k: res.get(k) for k in ('status', 'speedup', 'verdict') if k in res}
                     | ({'speedup': (res.get('performance') or {}).get('speedup')} if 'performance' in res else {})))
    return 0 if res.get('status') in ('PASS', 'ok') else 1


def cmd_check(a):
    tree, out = Path(a.tree).resolve(), Path(a.out).resolve()
    cand_env = candidate_environment(a.config)
    link_assets(tree)

    def run():
        status, correctness, performance = evaluate(tree, out, skip_timing=a.skip_timing, cand_env=cand_env,
                                                    quick=a.quick)
        return {'status': status, 'tree': str(tree), 'config': a.config, 'environment': cand_env,
                'correctness': correctness, 'performance': performance}
    return finish(out, 'check.json', guarded(out, run, tree))


def cmd_time(a):
    tree, out = Path(a.tree).resolve(), Path(a.out).resolve()
    cand_env = candidate_environment(a.config)
    link_assets(tree)

    def run():
        res = time_tree(tree, out, 'tree', cand_env=cand_env)
        return dict(res, tree=str(tree), config=a.config, environment=cand_env)
    return finish(out, 'time.json', guarded(out, run, tree))


def nsys_summaries(rep, out):
    nsys = shutil.which('nsys') or CFG.get('nsys')
    stats = {}
    for report in ('cuda_gpu_kern_sum', 'nvtx_sum', 'cuda_gpu_mem_time_sum', 'cuda_api_sum'):
        prefix = out / f'nsys_{report}'
        r = subprocess.run([nsys, 'stats', '--force-export=true', '--report', report, '--format', 'csv',
                            '--output', str(prefix), str(rep)], capture_output=True, text=True)
        files = sorted(out.glob(f'nsys_{report}*.csv'))
        stats[report] = [str(f.name) for f in files] if files else f'failed: {r.stderr[-400:]}'
    top = []
    kern = sorted(out.glob('nsys_cuda_gpu_kern_sum*.csv'))
    if kern:
        import csv
        with open(kern[0]) as f:
            rows = list(csv.DictReader(f))
        for row in rows[:40]:
            top.append({'time_pct': row.get('Time (%)'), 'total_ms': round(float(row.get('Total Time (ns)', 0)) / 1e6, 1),
                        'instances': row.get('Instances'), 'name': (row.get('Name') or '')[:160]})
    return stats, top


def cmd_profile(a):
    tree, out = Path(a.tree).resolve(), Path(a.out).resolve()
    cand_env = candidate_environment(a.config)
    link_assets(tree)

    def run():
        res = time_tree(tree, out, 'profile', cand_env=cand_env, profile=True)
        rep = out / 'nsys_profile.nsys-rep'
        if res['status'] == 'ok' and rep.exists():
            res['trace'] = str(rep)
            res['nsys_stats'], res['top_kernels'] = nsys_summaries(rep, out)
            lines = [f'| {k["time_pct"]} | {k["total_ms"]} | {k["instances"]} | `{k["name"]}` |' for k in res['top_kernels']]
            (out / 'top_kernels.md').write_text('| time % | total ms | calls | kernel |\n|---|---|---|---|\n'
                                                + '\n'.join(lines) + '\n')
        elif res['status'] == 'ok':
            res['status'] = 'ERROR'
            res['reason'] = 'nsys wrote no report'
        res['window'] = (f'{CFG["timing"]["window"]} iterations after a warmup of {CFG["timing"]["warmup"]}: '
                         f'{CFG["timing"]["window"] // CFG["update_ratio"]} generator iterations, the rest '
                         f'critic-only; one NVTX range per iteration')
        return dict(res, tree=str(tree), config=a.config, environment=cand_env)
    return finish(out, 'profile.json', guarded(out, run, tree))


def cmd_baseline(a):
    out = Path(a.out).resolve()

    def run():
        ref_src = check_reference()
        with scratch_dir('sf-ref-') as tmp:
            tree = materialize(ref_src, Path(tmp) / 'tree')
            runs = {k: time_tree(tree, out, k) for k in ('ref_a', 'ref_b', 'ref_c')}
        ok = [v for v in runs.values() if v['status'] == 'ok']
        ms = [v['ms_per_iteration'] for v in ok]
        return {'status': 'ok' if len(ok) == len(runs) else 'ERROR', 'runs': runs, 'ms_per_iteration': ms,
                'spread': round((max(ms) - min(ms)) / statistics.mean(ms), 4) if ms else None}
    return finish(out, 'baseline.json', guarded(out, run))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('check')
    k.add_argument('--tree', required=True)
    k.add_argument('--out', required=True)
    k.add_argument('--config')
    k.add_argument('--quick', action='store_true')
    k.add_argument('--skip-timing', action='store_true')
    t = sub.add_parser('time')
    t.add_argument('--tree', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--config')
    f = sub.add_parser('profile')
    f.add_argument('--tree', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--config')
    b = sub.add_parser('baseline')
    b.add_argument('--out', required=True)
    a = p.parse_args()
    return {'check': cmd_check, 'time': cmd_time, 'profile': cmd_profile, 'baseline': cmd_baseline}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
