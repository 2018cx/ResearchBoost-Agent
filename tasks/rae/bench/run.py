"""RAE Stage-1 bench: one command per question.

Run from the task workspace root with the rae Python. GPU phases take /tmp/rae_gpu01.lock
themselves, one phase at a time; do NOT wrap these commands in flock (that would deadlock).

  python bench/run.py check --tree <dir> --out <dir> [--config <file>]
                                                   the verdict: gates 0-3 below on a copy of the tree
                                                   -> <dir>/check.json
  python bench/run.py check --tree <dir> --out <dir> [--config <file>] --quick
                                                   self-test of a live tree (gate 0 only); never a score
  python bench/run.py sweep --tree <dir> --config a.json --config b.json ... --out <dir>
                                                   screen: reference, one timing per switch config, reference again;
                                                   a failing config is recorded and the sweep goes on; sweep.json
                                                   is rewritten after every run; no correctness, never a score
  python bench/run.py time --tree <dir> --out <dir>
                                                   phase-weighted timing of one tree, no comparison
  python bench/run.py profile --tree <dir> [--config <file>] [--phase all] --out <dir>
                                                   Nsight Systems trace of one timing window per training phase;
                                                   per phase: its own log and records, traced-window checks
                                                   (no compile inside, expected D steps), inputs verified
  python bench/run.py baseline --out <dir>         reference against itself (noise floor, baseline card)

Every GPU run (timing, numerics record, trajectories) goes through run_gpu: it holds the GPU lock, waits until the
visible GPUs are empty and idle (INVALID after 10 minutes otherwise), runs in its own process group with a timeout
(bench.json timeouts_s; TIMEOUT kills the group) with a fresh short TMPDIR, and records the GPU state after. Every timing run also gets a fresh
compile cache under /tmp (cold compile, nothing shared between runs or configurations), is scanned for
the invalid-run signatures in bench.json (e.g. compile falling back to eager at the recompile limit -> INVALID),
fails if torch.compile compiled anything inside a timing window, and carries a fingerprint (argv, switches, math
mode, GPU UUIDs, cache, library versions). Outputs carry cost.gpu_hours (lock hold time x cards). Every command
(check, sweep, time, profile, baseline) hashes every frozen input declared in task/formulation.json
(weights, the HF and torch hub caches the training code loads, fixtures, resume states) before (mismatch: nothing
is measured) and after (mismatch: ERROR). Callers read these fields; they do not need a wrapper of their own.

Candidate switches: a candidate's opt-in switches live in the config file passed with --config (JSON or YAML with
an `environment` mapping). The bench reads that file and sets those variables for the candidate's processes only;
the caller's environment is not consulted.

Math mode: reference and candidate both start in the frozen math mode from bench.json (math_mode), set by the
bench before any training code is imported. A candidate that changes math flags is measured N2.

Verdict flow (the one GOAL.md describes); each gate stops the evaluation when it fails:
  0. coverage   main() outside the per-step loop body and parse_args() must equal the reference AST (the
                correctness adapter re-creates that setup), and prepare_dataloader must yield bitwise the
                same first batches as the reference.
  1. benefit    reference, candidate, reference: fresh torchruns of the real entrypoint on the timing config; their
                fingerprints must agree with the first reference run (else ERROR, not comparable). No speedup above
                max(min_speedup, 1 + 2 x reference spread) over the reference -> rejected as no benefit (status
                UNAVAILABLE, verdict no_benefit). Peak memory is recorded.
  2. bitwise    2 short cases (1 reconstruction step, 1 adversarial step) recorded in deterministic mode. Exact
                checks (phase, call counts, RNG, invariants, cross-rank sync) must hold, else INCORRECT. Every
                tensor bitwise identical -> PASS (N0), adopt.
  3. statistical  paired-seed training trajectories (losses only, compressed schedule, deterministic kernels so that
                the same code and seed reproduce the reference trajectory exactly) on the frozen seeds, in the
                seed_schedule of bench.json. delta = 2 x reference seed sigma per endpoint. Before the last step of
                the schedule: an endpoint beyond delta and significant -> INCORRECT, all within delta -> PASS,
                otherwise more seeds; at the last step any endpoint beyond delta -> INCORRECT, else PASS.
  Per-tensor tolerance gates and fp64 arbitration are reported as diagnostics only.
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
ROOT = BENCH.parent                       # task workspace
CFG = json.loads((BENCH / 'bench.json').read_text())
LOCK = Path('/tmp/rae_gpu01.lock')
PY = Path(sys.executable)
TORCHRUN = PY.parent / 'torchrun'
TIERS = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5']
SCRATCH = '/tmp'                          # short paths: unix sockets of compile workers live under TMPDIR
TIMEOUT = {'timing': 1800, 'numerics': 1800, 'trajectory': 3600, **CFG.get('timeouts_s', {})}
# Variables the bench sets itself; a candidate config may not override them.
RESERVED = {'TMPDIR', 'TMP', 'TEMP', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH',
            'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'EXPERIMENT_NAME', 'NVIDIA_TF32_OVERRIDE'}
# Log lines that mean a run did not execute the configuration it was asked to measure (train-opt-measurement skill).
SIGNATURES = [(re.compile(x['regex']), x['reason']) for x in CFG.get('invalid_run_signatures', [])]
COST = {'gpu_seconds': 0.0, 'gpu_runs': 0}     # GPU lock hold time x cards, over the whole command


def path(key):
    value = Path(key)
    return value if value.is_absolute() else (ROOT / value).resolve()


def log(msg):
    print(f'[bench {datetime.now():%H:%M:%S}] {msg}', flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def evaluator_sha256():
    """Joint hash of the frozen evaluator: numerics/, bench/ code + config + thresholds + records manifests."""
    files = sorted(p for p in (ROOT / 'numerics').rglob('*.py'))
    files += sorted(p for p in BENCH.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                    and 'records' not in p.parts and 'states' not in p.parts)
    files += sorted(p for p in (BENCH / 'records').rglob('*.json'))
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode() + b'\0' + sha256_file(p).encode() + b'\n')
    return h.hexdigest()


class GpuLock:
    def __init__(self, what, timeout=3600):
        self.what, self.timeout = what, timeout

    def __enter__(self):
        self.fd = os.open(LOCK, os.O_RDONLY)          # same lock file as flock(1); read access is enough
        start, said = time.time(), False
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.acquired = time.time()
                return self
            except BlockingIOError:
                if not said:
                    log(f'waiting for {LOCK} ({self.what}); if this command is wrapped in flock, unwrap it')
                    said = True
                if time.time() - start > self.timeout:
                    raise SystemExit(f'GPU lock not acquired within {self.timeout}s')
                time.sleep(5)

    def __exit__(self, *_):
        COST['gpu_seconds'] += (time.time() - self.acquired) * CFG['world_size']
        COST['gpu_runs'] += 1
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def cost():
    return {'gpu_hours': round(COST['gpu_seconds'] / 3600, 4), 'gpu_runs': COST['gpu_runs'],
            'method': f'GPU lock hold time x {CFG["world_size"]} cards: includes startup, compile, warmup and failed runs'}


def scratch_dir(prefix):
    """A temporary directory on the short-path scratch disk, after checking it has room."""
    free = shutil.disk_usage(SCRATCH).free / 2**30
    need = CFG.get('min_scratch_free_gib', 3)
    if free < need:
        raise SystemExit(f'{SCRATCH} has {free:.1f} GiB free, the bench needs {need}: free space first '
                         f'(the bench writes source copies, compile caches and training results there)')
    return tempfile.TemporaryDirectory(prefix=prefix, dir=SCRATCH)


def gpu_state():
    """Visible GPUs: index, UUID, memory in use, utilization. None when nvidia-smi is unavailable."""
    smi = shutil.which('nvidia-smi')
    if not smi:
        return None
    r = subprocess.run([smi, '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, timeout=60)
    rows = []
    for line in r.stdout.splitlines():
        idx, uuid, mem, util = [x.strip() for x in line.split(',')]
        rows.append({'index': int(idx), 'uuid': uuid, 'memory_used_mib': float(mem), 'utilization': float(util)})
    return rows


def wait_idle(what, patience=600):
    """Every visible GPU nearly empty and idle (no other process), waiting up to `patience` seconds.
    Returns (state, problem); problem is None when idle."""
    start = time.time()
    while True:
        state = gpu_state()
        busy = [g for g in state or [] if g['memory_used_mib'] > 1024 or g['utilization'] > 5]
        if not busy:
            return state, None
        if time.time() - start > patience:
            return state, f'GPU {[g["index"] for g in busy]} not idle before {what} for {patience}s (another process?)'
        time.sleep(15)


def run_gpu(cmd, cwd, env, log_file, what, timeout):
    """One GPU job under the lock: idle check, own process group, timeout, a fresh short TMPDIR (the data loader's
    AF_UNIX sockets live there; a deep TMPDIR breaks them with 'AF_UNIX path too long'). Returns a dict with
    returncode, status (ok / TIMEOUT / INVALID), wall seconds and the GPU state before and after."""
    with GpuLock(what), scratch_dir('rae-g-') as tmpdir:
        env = dict(env if env is not None else os.environ, TMPDIR=tmpdir, TMP=tmpdir, TEMP=tmpdir)
        before, problem = wait_idle(what)
        if problem:
            return {'status': 'INVALID', 'reason': problem, 'returncode': None, 'gpus_before': before}
        log(what)
        start = time.time()
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc, status = proc.wait(timeout=timeout), None
            except subprocess.TimeoutExpired:
                for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
                    try:
                        os.killpg(proc.pid, sig)
                        proc.wait(timeout=grace)
                        break
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        continue
                rc, status = None, 'TIMEOUT'
        wall = time.time() - start
        after, _ = wait_idle(f'{what} (after)', patience=60)
    out = {'status': status or 'ok', 'returncode': rc, 'wall_s': round(wall, 1), 'gpus_before': before,
           'gpus_after': after}
    if status:
        out['reason'] = f'{what} exceeded {timeout}s; process group killed; see {Path(log_file).name}'
    elif any(g['memory_used_mib'] > 1024 for g in after or []):
        out['warning'] = 'GPU memory still in use 60s after the run ended (leaked process?)'
    return out


def signatures(log_file):
    """Invalid-run signatures found in a log: [{reason, line}]."""
    hits, text = [], Path(log_file).read_text(errors='replace') if Path(log_file).exists() else ''
    for pattern, reason in SIGNATURES:
        m = pattern.search(text)
        if m:
            line = text[text.rfind('\n', 0, m.start()) + 1:text.find('\n', m.end())]
            hits.append({'reason': reason, 'line': line.strip()[:300]})
    return hits


def tree_digest(tree):
    """sha256 over the relative paths and contents of a source tree (symlinks to assets excluded)."""
    h = hashlib.sha256()
    for p in sorted(x for x in Path(tree).rglob('*') if x.is_file() and not x.is_symlink()
                    and '__pycache__' not in x.parts and '.caches' not in x.parts):
        h.update(str(p.relative_to(tree)).encode() + b'\0' + sha256_file(p).encode() + b'\n')
    return h.hexdigest()


def content_digest(p):
    """sha256 of a file; for a directory, sha256 over (relative path, file sha256) of every file in sorted order,
    the same digest task/formulation.json declares."""
    p = Path(p)
    if p.is_file():
        return {'sha256': sha256_file(p)}
    h, n = hashlib.sha256(), 0
    for f in sorted(p.rglob('*')):
        if f.is_file():
            h.update(str(f.relative_to(p)).encode() + b'\0' + sha256_file(f).encode() + b'\n')
            n += 1
    return {'sha256': h.hexdigest(), 'files': n}


def frozen_inputs():
    """(name, path, declared sha256) of every input task/formulation.json declares: weights, caches the training
    code loads (HF, torch hub), fixtures, resume states, thresholds, frozen trajectories."""
    assets = json.loads((ROOT / 'task' / 'formulation.json').read_text())['assets']
    out = []
    for name, a in assets.items():
        if isinstance(a, dict) and 'path' in a and 'sha256' in a:
            out.append((name, a['path'], a['sha256']))
        elif isinstance(a, dict):
            out += [(f'{name}:{k}', k, v) for k, v in a.items() if isinstance(v, str) and len(v) == 64]
    return out


def verify_inputs(when):
    """Hash every frozen input and compare with its declared value. Returns {ok, checked, mismatches, seconds}."""
    start, bad = time.time(), []
    items = frozen_inputs()
    for name, rel, want in items:
        target = path(rel)
        got = content_digest(target)['sha256'] if target.exists() else None
        if got != want:
            bad.append({'input': name, 'path': rel, 'declared': want, 'found': got})
    out = {'when': when, 'ok': not bad, 'checked': len(items), 'mismatches': bad, 'seconds': round(time.time() - start, 1)}
    log(f'inputs {when}: {len(items) - len(bad)}/{len(items)} match the formulation')
    return out


def guarded_inputs():
    """Before a command that runs training code: the frozen inputs must match, or nothing is measured."""
    before = verify_inputs('before')
    if not before['ok']:
        raise SystemExit(f'frozen inputs differ from task/formulation.json, nothing measured: {before["mismatches"]}')
    return before


# Fingerprint fields two timing runs must share to be compared (the switches and argv are what a sweep varies).
FINGERPRINT_KEYS = ('math_mode', 'gpus', 'compile_cache', 'instrumentation', 'versions')


def fingerprint_diff(a, b):
    """Fingerprint fields that differ between two timing runs ([] when comparable)."""
    a, b = a or {}, b or {}
    return [k for k in FINGERPRINT_KEYS if a.get(k) != b.get(k)]


def source_end_check(result, tree, key='source_digest'):
    """Digest the materialized source again before it is removed; a change means the code wrote into its own tree
    while it ran, so what was measured is not what was copied."""
    result[f'{key}_end'] = tree_digest(tree)
    if result[f'{key}_end'] != result[key]:
        result['status'] = 'ERROR'
        result['reason'] = 'the source tree changed while it ran (the code wrote into its own tree)'
    return result


def versions():
    out = {}
    for name in ('torch', 'triton', 'numpy'):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


# ---------------------------------------------------------------- source trees

def materialize(src, dest):
    """Copy a source tree (no .git) and attach the read-only weights the upstream code expects."""
    src, dest = Path(src).resolve(), Path(dest)
    def ignore(directory, names):
        rel = Path(directory).resolve().relative_to(src)
        drop = {'.git', '__pycache__'} & set(names)
        if rel == Path('.'):
            drop |= {'models'} & set(names)           # weights live outside the tree
        if rel == Path('src/disc'):
            drop |= {'.caches'} & set(names)
        return drop
    shutil.copytree(src, dest, symlinks=False, ignore=ignore)
    for d in [dest, *(p for p in dest.rglob('*') if p.is_dir())]:
        d.chmod(d.stat().st_mode | 0o700)        # read-only sources (e.g. the reference checkout); the copy must not be
    (dest / 'models').symlink_to(path(CFG['assets']['models']))
    caches = dest / 'src' / 'disc' / '.caches'
    caches.mkdir(parents=True, exist_ok=True)
    (caches / 'vgg.pth').symlink_to(path(CFG['assets']['lpips_vgg']))
    return dest


def candidate_environment(config_file):
    """`environment` mapping from a candidate's switch config (JSON or YAML); {} without a config."""
    if config_file is None:
        return {}
    text = Path(config_file).read_text()
    if str(config_file).endswith('.json'):
        data = json.loads(text)
    else:
        import yaml
        data = yaml.safe_load(text)
    env = (data or {}).get('environment') or {}
    if not isinstance(env, dict) or not all(isinstance(k, str) for k in env):
        raise SystemExit(f'{config_file}: environment must be a mapping of variable names to values')
    reserved = sorted(RESERVED & set(env))
    if reserved:
        raise SystemExit(f'{config_file}: {reserved} are set by the bench (scratch, compile caches, devices, math '
                         f'mode) and cannot be candidate switches')
    return {k: str(v) for k, v in env.items()}


def check_reference():
    ref = path(CFG['reference'])
    head = subprocess.run(['git', '-C', str(ref), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(ref), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True).stdout.strip()
    if head != CFG['reference_commit'] or dirty:
        raise SystemExit(f'reference checkout changed (HEAD {head}, dirty={bool(dirty)})')
    return ref


# ---------------------------------------------------------------- 1. setup guard

def _is_step_loop(node):
    t = node.target
    return (isinstance(node, ast.For) and isinstance(t, ast.Tuple) and len(t.elts) == 2
            and isinstance(t.elts[0], ast.Name) and t.elts[0].id == 'step'
            and isinstance(t.elts[1], ast.Tuple) and len(t.elts[1].elts) == 2
            and isinstance(t.elts[1].elts[0], ast.Name) and t.elts[1].elts[0].id == 'images')


def setup_fingerprint(tree):
    module = ast.parse((Path(tree) / 'src' / 'train_stage1.py').read_text())
    parts, loops = {}, 0
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('main', 'parse_args'):
            if node.name == 'main':
                for inner in ast.walk(node):
                    if isinstance(inner, ast.For) and _is_step_loop(inner):
                        inner.body = [ast.Pass()]
                        loops += 1
            parts[node.name] = ast.dump(node, include_attributes=False)
    return parts, loops


def setup_guard(ref, tree):
    want, _ = setup_fingerprint(ref)
    got, loops = setup_fingerprint(tree)
    problems = []
    if loops != 1:
        problems.append(f'expected exactly one `for step, (images, _) in ...` loop in main(), found {loops}')
    for name in ('parse_args', 'main'):
        if got.get(name) != want.get(name):
            problems.append(f'{name}() differs from the reference outside the per-step loop body')
    return problems


# ---------------------------------------------------------------- 2. loader check

LOADER_PROBE = r'''
import hashlib, json, sys, torch
from torchvision import transforms
sys.path.insert(0, "src")
from utils.train_utils import prepare_dataloader
fixture, world, batch, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
t = transforms.Compose([transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.RandomCrop(256), transforms.ToTensor()])
out = []
for rank in range(world):
    torch.manual_seed(1000 + rank)
    loader, sampler = prepare_dataloader(fixture, batch, workers, rank, world, transform=t)
    sampler.set_epoch(0)
    it = iter(loader)
    for _ in range(2):
        images, labels = next(it)
        out.append([hashlib.sha256(images.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                    hashlib.sha256(labels.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                    list(images.shape), str(images.dtype)])
print(json.dumps({"steps_per_epoch": len(loader), "batches": out}))
'''


def loader_digest(tree):
    t = CFG['timing']
    r = subprocess.run([str(PY), '-c', LOADER_PROBE, str(path(t['fixture'])), str(CFG['world_size']),
                        str(CFG['per_rank_batch']), str(t['num_workers'])],
                       cwd=tree, capture_output=True, text=True, timeout=900)
    if r.returncode:
        return None, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1]), None


# ---------------------------------------------------------------- 3. numerics

def numerics(tree, out, cand_env):
    n = CFG['numerics']
    kwargs = {'repo': str(tree), 'config': str(path(n['config'])), 'fixture': str(path(n['fixture'])),
              'batch': CFG['per_rank_batch'], 'steps_per_epoch': n['steps_per_epoch']}
    rec = out / 'records'
    env = dict(os.environ, PYTHONPATH=str(ROOT), **cand_env)
    cmd = [str(TORCHRUN), '--standalone', '--nnodes=1', f'--nproc-per-node={CFG["world_size"]}',
           '-m', 'numerics', 'record', '--adapter', n['adapter'], '--adapter-kwargs', json.dumps(kwargs),
           '--cases', str(path(n['cases'])), '--sketch-dim', str(n['sketch_dim']),
           # no --dtype-trace: under its dispatch mode compiled regions fall back to eager, so a traced 1-step case
           # would record uncompiled code (a compiled candidate then looks bitwise identical to the reference)
           '--math-mode', CFG['math_mode'], *(['--deterministic'] if n.get('deterministic') else []),
           '--out', str(rec)]
    r = run_gpu(cmd, ROOT, env, out / 'record.log', 'numerics record', TIMEOUT['numerics'])
    if r['status'] != 'ok' or r['returncode']:
        return {'status': 'ERROR', 'reason': r.get('reason') or f'record exited {r["returncode"]}; see record.log'}
    report = out / 'compare.json'
    r = subprocess.run([str(PY), '-m', 'numerics', 'compare', '--ref', str(path(n['ref'])), '--cand', str(rec),
                        '--golden', str(path(n['golden'])), '--thresholds', str(path(n['thresholds'])),
                        *[a for x in n.get('exclude', []) for a in ('--exclude', f'{x["regex"]}={x["reason"]}')],
                        '--out', str(report)], cwd=ROOT, env=env, capture_output=True, text=True)
    (out / 'compare.log').write_text(r.stdout + r.stderr)
    if not report.exists():
        return {'status': 'ERROR', 'reason': f'compare exited {r.returncode}; see compare.log'}
    data = json.loads(report.read_text())
    if not CFG['numerics'].get('keep_records'):
        shutil.rmtree(rec, ignore_errors=True)
    summary = {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in data.items()
               if k not in ('status', 'measured_tier')}
    bitwise = (data.get('checked') == data.get('bitwise_identical') and not data.get('dtype_changes')
               and not data.get('precision_mode_changes') and data.get('bitwise_trusted', False))
    return {'status': data.get('status'), 'measured_tier': data.get('measured_tier'), 'report': str(report),
            'errors': len(data.get('errors') or []), 'exact_failures': data.get('exact_failures') or [],
            'bitwise': bitwise, 'summary': summary}


def trajectories(tree, out, cand_env):
    """Gate 3: candidate trajectories on the frozen seeds, 3 -> 6 -> 10, until a verdict that is not inconclusive."""
    tj = CFG['trajectory']
    spec = json.loads(path(tj['spec']).read_text())
    frozen_path = path(tj['frozen'])
    seeds = json.loads(frozen_path.read_text())['reference_seeds']
    kwargs = {'repo': str(tree), 'config': str(path(CFG['numerics']['config'])), 'fixture': str(path(tj['fixture'])),
              'batch': CFG['per_rank_batch'], 'steps_per_epoch': tj['steps_per_epoch']}
    rec, done, report = out / 'trajectory', [], None
    env = dict(os.environ, PYTHONPATH=str(ROOT), **cand_env)
    schedule = tj['seed_schedule']
    for idx, n in enumerate(schedule):
        final = idx == len(schedule) - 1
        new = [x for x in seeds[:n] if x not in done]
        cases = out / f'traj_cases_{n}.json'
        cases.write_text(json.dumps([dict(spec['case'], name=f'traj_s{x}', seed=x) for x in new]))
        cmd = [str(TORCHRUN), '--standalone', '--nnodes=1', f'--nproc-per-node={CFG["world_size"]}',
               '-m', 'numerics', 'record', '--adapter', CFG['numerics']['adapter'], '--adapter-kwargs',
               json.dumps(kwargs), '--cases', str(cases), '--losses-only', '--math-mode', CFG['math_mode'],
               '--deterministic', '--out', str(rec)]   # reproducible: a seed pairs exactly with the reference run
        r = run_gpu(cmd, ROOT, env, out / f'traj_{n}.log', f'trajectories: seeds {new}', TIMEOUT['trajectory'])
        if r['status'] != 'ok' or r['returncode']:
            return {'verdict': 'error',
                    'reason': r.get('reason') or f'trajectory record exited {r["returncode"]}; see traj_{n}.log'}
        done += new
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


# ---------------------------------------------------------------- 4. timing

def _power_sampler(out_file):
    smi = shutil.which('nvidia-smi')
    if not smi:
        return None
    return subprocess.Popen([smi, '--query-gpu=timestamp,index,power.draw,clocks.sm,utilization.gpu',
                             '--format=csv,noheader,nounits', '-lms', '200'],
                            stdout=open(out_file, 'w'), stderr=subprocess.DEVNULL)


def _read_power(f):
    rows = []
    for line in Path(f).read_text().splitlines():
        try:
            ts, idx, pw, clk, util = [x.strip() for x in line.split(',')]
            t = datetime.strptime(ts, '%Y/%m/%d %H:%M:%S.%f').timestamp()
            rows.append((t, int(idx), float(pw), float(clk), float(util)))
        except ValueError:
            continue
    return rows


def time_tree(tree, out, label, profile_phase=None, cand_env=None):
    t = CFG['timing']
    out.mkdir(parents=True, exist_ok=True)
    scratch = scratch_dir('rae-t-')
    results = Path(scratch.name)
    tag = f'{label}_{profile_phase}' if profile_phase else label     # one log and record per profiled phase
    prefix = out / f'timing_{tag}'
    argv = [str(TORCHRUN), '--standalone', '--nnodes=1', f'--nproc-per-node={CFG["world_size"]}',
            str(BENCH / 'timing.py'), '--out', str(prefix), '--steps-per-epoch', str(t['steps_per_epoch']),
            '--warmup', *map(str, t['warmup']), '--window', *map(str, t['window']), '--repeats', str(t['repeats']),
            '--math-mode', CFG['math_mode']]
    if profile_phase:
        argv += ['--profile-phase', profile_phase]
    argv += ['--', '--config', str(path(t['config'])), '--data-path', str(path(t['fixture'])),
             '--results-dir', str(results), '--precision', CFG['precision'], '--image-size', '256']
    # A fresh compile cache per run: every run compiles cold, no run inherits another configuration's artifacts.
    caches = {k: str(results / v) for k, v in (('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton'),
                                                ('CUDA_CACHE_PATH', 'cuda'))}
    for v in caches.values():
        Path(v).mkdir()
    env = dict(os.environ, EXPERIMENT_NAME=f'timing-{label}', PYTHONPATH=str(Path(tree) / 'src'), **caches,
               **(cand_env or {}))
    if profile_phase:
        nsys = shutil.which('nsys') or CFG.get('nsys')
        if not nsys:
            raise SystemExit('nsys not found; set bench.json nsys or source the Nsight env first')
        argv = [nsys, 'profile', '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                '--trace=cuda,nvtx,osrt,cublas,cudnn', '--sample=none', '--force-overwrite=true',
                '-o', str(out / f'nsys_{label}_{profile_phase}'), *argv]
    for f in Path(out).glob(f'timing_{tag}.r*.json'):
        f.unlink()                                 # a stale record must not stand in for this run
    log_file = out / f'timing_{tag}.log'
    what = f'timing {label}{" (nsys " + profile_phase + ")" if profile_phase else ""}'
    sampler = None if profile_phase else _power_sampler(out / f'power_{label}.csv')
    try:
        r = run_gpu(argv, tree, env, log_file, what, TIMEOUT['timing'])
    finally:
        if sampler:
            sampler.terminate()
            sampler.wait()
        scratch.cleanup()
    fingerprint = {'argv': argv, 'candidate_environment': cand_env or {}, 'math_mode': CFG['math_mode'],
                   'gpus': [(g['index'], g['uuid']) for g in r.get('gpus_before') or []],
                   'compile_cache': 'cold (fresh per run)', 'instrumentation': 'nsys' if profile_phase else 'none',
                   'signatures_checked': [reason for _, reason in SIGNATURES], 'versions': versions()}
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
    if profile_phase:
        window, problems = profile_window(ranks, profile_phase)
        return {'status': 'INVALID' if problems else 'ok', 'problems': problems, 'window': window,
                'trace': str(out / f'nsys_{label}_{profile_phase}.nsys-rep'), 'log': log_file.name, **run}
    return {**summarize_timing(ranks, out / f'power_{label}.csv', r['wall_s']), **run}


def profile_window(ranks, phase):
    """The traced window of a profiled phase (the first window of that phase): present on every rank, nothing
    compiled inside it, the expected number of discriminator steps. Returns (window summary, problems)."""
    t = CFG['timing']
    idx = ('recon', 'disc', 'gan').index(phase)
    expect_other = 0 if phase == 'recon' else t['window'][idx] * t['disc_updates']
    ws = [w for x in ranks for w in x['windows'] if w['phase'] == phase and w['k'] == 0]
    problems = []
    if len(ws) != len(ranks):
        problems.append(f'{phase}: traced window missing on {len(ranks) - len(ws)} rank(s)')
    if any(w.get('compiles_in_window') for w in ws):
        problems.append(f'{phase}: torch.compile compiled graphs inside the traced window (not steady state)')
    if any(w['other_optimizer_steps'] != expect_other for w in ws):
        problems.append(f'{phase}: discriminator steps in the traced window differ from {expect_other} '
                        f'(schedule or step structure changed)')
    window = {'steps': t['window'][idx], 'seconds_max_rank': round(max((w['seconds'] for w in ws), default=0), 4),
              'compiles_in_window': [w.get('compiles_in_window') for w in ws],
              'other_optimizer_steps': [w['other_optimizer_steps'] for w in ws]}
    return window, problems


def summarize_timing(ranks, power_csv, wall):
    t = CFG['timing']
    per_window = {}
    for x in ranks:
        for w in x['windows']:
            key = (w['phase'], w['k'])
            prev = per_window.get(key)
            if prev is None or w['seconds'] > prev['seconds']:
                per_window[key] = w
    power = _read_power(power_csv) if Path(power_csv).exists() else []
    phases, problems = {}, []
    for phase_index, phase in enumerate(('recon', 'disc', 'gan')):
        ws = [per_window[(phase, k)] for k in range(t['repeats']) if (phase, k) in per_window]
        if len(ws) != t['repeats']:
            problems.append(f'{phase}: {len(ws)} of {t["repeats"]} windows')
            continue
        expect_other = 0 if phase == 'recon' else t['window'][phase_index] * t['disc_updates']
        compiled = [w for w in ws if w.get('compiles_in_window')]
        if compiled:
            problems.append(f'{phase}: torch.compile compiled {sum(w["compiles_in_window"] for w in compiled)} '
                            f'graphs inside timing windows (still compiling in steady state)')
        for w in ws:
            if w['other_optimizer_steps'] != expect_other:
                problems.append(f'{phase} window {w["k"]}: {w["other_optimizer_steps"]} discriminator '
                                f'steps, expected {expect_other} (schedule or step structure changed)')
        ms = [1000 * w['seconds'] / w['steps'] for w in ws]
        inside = [p for p in power if any(w['wall_start'] <= p[0] <= w['wall_end'] for w in ws)]
        watts = {}
        for p in inside:
            watts.setdefault(p[1], []).append(p[2])
        total_w = sum(statistics.mean(v) for v in watts.values()) if watts else None
        clocks = [p[3] for p in inside]
        median_ms = statistics.median(ms)
        phases[phase] = {
            'ms_per_step_windows': [round(v, 2) for v in ms], 'ms_per_step': round(median_ms, 2),
            'window_spread': round((max(ms) - min(ms)) / median_ms, 4),
            'avg_power_w_all_gpus': round(total_w, 1) if total_w else None,
            'joules_per_sample': round(total_w * median_ms / 1000 / (CFG['world_size'] * CFG['per_rank_batch']), 3)
            if total_w else None,
            'sm_clock_median_mhz': statistics.median(clocks) if clocks else None,
        }
    weights = t['phase_weights']
    weighted = (sum(weights[p] * phases[p]['ms_per_step'] for p in phases) / sum(weights[p] for p in phases)
                if len(phases) == 3 else None)
    return {
        'status': 'ok' if not problems else 'ERROR', 'problems': problems, 'phases': phases,
        'weighted_ms_per_step': round(weighted, 2) if weighted else None,
        'images_per_s_local': round(CFG['world_size'] * CFG['per_rank_batch'] / (weighted / 1000), 1) if weighted else None,
        'peak_memory_allocated_gib': round(max(x['peak_memory_allocated_bytes'] for x in ranks) / 2**30, 2),
        'startup_to_first_step_s': round(max(x['startup_to_first_step_s'] or 0 for x in ranks), 1),
        'process_wall_s': round(wall, 1),
    }


def compare_timing(ref_a, cand, ref_b):
    def ratio(a, b):
        return round(a / b, 4) if a and b else None
    out = {'phases': {}}
    for phase in ('recon', 'disc', 'gan'):
        ra, rb, c = ref_a['phases'][phase]['ms_per_step'], ref_b['phases'][phase]['ms_per_step'], cand['phases'][phase]['ms_per_step']
        ref = (ra + rb) / 2
        out['phases'][phase] = {'reference_ms': round(ref, 2), 'candidate_ms': c, 'speedup': ratio(ref, c),
                                'reference_spread': round(abs(ra - rb) / ref, 4)}
    ra, rb = ref_a['weighted_ms_per_step'], ref_b['weighted_ms_per_step']
    ref = (ra + rb) / 2
    out.update(reference_weighted_ms=round(ref, 2), candidate_weighted_ms=cand['weighted_ms_per_step'],
               speedup=ratio(ref, cand['weighted_ms_per_step']),
               reference_spread=round(abs(ra - rb) / ref, 4))
    return out


# ---------------------------------------------------------------- pipelines

def coverage(tree):
    """Gate 0a (CPU): setup guard and data-loader digest. Returns (status, correctness) or None when it holds."""
    ref_src = check_reference()
    problems = setup_guard(ref_src, tree)
    if problems:
        return 'ERROR', {'passed': False, 'stage': 'setup_guard', 'problems': problems,
                         'note': 'the correctness adapter re-creates main() setup from the reference; move the '
                                 'change into a module the training code calls, or into the per-step loop'}
    want, err = loader_digest(ref_src)
    got, err2 = loader_digest(tree)
    if want is None:
        return 'ERROR', {'passed': False, 'stage': 'loader_check', 'reference_error': err}
    if got != want:
        return 'INCORRECT', {'passed': False, 'stage': 'loader_check', 'error': err2,
                             'note': 'prepare_dataloader yields different first batches than the reference'}
    return None


def benefit(tree, out, cand_env):
    """Gate 1: reference, candidate, reference timing and peak memory."""
    ref_src = check_reference()
    with scratch_dir('rae-ref-') as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / 'tree')
        ref_a = time_tree(ref_tree, out, 'ref_a')
        cand = time_tree(tree, out, 'cand', cand_env=cand_env) if ref_a['status'] == 'ok' else None
        ref_b = time_tree(ref_tree, out, 'ref_b') if cand and cand['status'] == 'ok' else None
    runs = {'ref_a': ref_a, 'cand': cand, 'ref_b': ref_b}
    if any(v is None or v['status'] != 'ok' for v in runs.values()):
        return None, {'speedup': None, 'runs': runs}
    # Only runs that agree on everything but the switches can be compared (train-opt-measurement skill).
    diffs = {k: fingerprint_diff(ref_a.get('fingerprint'), v.get('fingerprint')) for k, v in runs.items() if k != 'ref_a'}
    bad = {k: d for k, d in diffs.items() if d}
    if bad:
        return None, {'speedup': None, 'runs': runs, 'fingerprint_diffs': diffs,
                      'reason': f'timing runs not comparable: fingerprint differs from ref_a in {bad}'}
    perf = {**compare_timing(ref_a, cand, ref_b), 'runs': runs, 'fingerprint_diffs': diffs}
    rules = CFG['benefit']
    speed_bar = max(rules['min_speedup'], 1 + 2 * perf['reference_spread'])
    ref_mem = (ref_a['peak_memory_allocated_gib'] + ref_b['peak_memory_allocated_gib']) / 2
    b = {'speedup': perf['speedup'], 'speedup_bar': round(speed_bar, 4),
         'peak_memory_gib': cand['peak_memory_allocated_gib'], 'reference_peak_memory_gib': round(ref_mem, 2)}
    b['passed'] = perf['speedup'] > speed_bar
    perf['benefit'] = b
    return b['passed'], perf


def evaluate(tree, out, skip_timing=False, cand_env=None, quick=False):
    """Gates 0 -> 1 -> 2 -> 3. Returns (status, correctness, performance). quick: gate 0 only."""
    cand_env = cand_env or {}
    failed = coverage(tree)
    if failed:
        return failed[0], failed[1], None
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
            measured = performance.pop('speedup')
            performance['measured_speedup'] = measured
            return 'UNAVAILABLE', {'passed': None, 'stage': 'benefit', 'verdict': 'no_benefit',
                                   'note': 'not faster than the reference: rejected without a precision '
                                           'evaluation (gate 1)'}, performance
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
                                 note=f'measured tier {tier} exceeds agent_max {CFG["agent_max_tier"]}'), performance
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


def cmd_check(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = [guarded_inputs()]
    with scratch_dir('rae-check-') as tmp:
        tree = materialize(args.tree, Path(tmp) / 'tree')
        digest = tree_digest(tree)
        cand_env = candidate_environment(args.config)
        status, correctness, performance = evaluate(tree, out, skip_timing=args.skip_timing, cand_env=cand_env,
                                                    quick=args.quick)
        result = {'self_test': args.quick, 'note': 'verdict on a copy of the live tree taken at the start' +
                  (' (quick: gate 0 only, not a score)' if args.quick else ''),
                  'source_digest': digest, 'candidate_environment': cand_env, 'status': status,
                  'correctness': correctness, 'performance': performance}
        source_end_check(result, tree)
    result.update(cost=cost(), inputs=inputs + [verify_inputs('after')])
    if not result['inputs'][-1]['ok']:
        result['status'] = 'ERROR'
    (out / 'check.json').write_text(json.dumps(result, indent=1) + '\n')
    log(f'check {status}; wrote {out / "check.json"}')


def cmd_sweep(args):
    """Quick screen of switch configs on one live tree (a copy taken at the start): reference, each config once,
    reference again. Every config is its own fresh process with a cold compile cache and a timeout; a config that
    fails is recorded and the sweep goes on. sweep.json is rewritten after every run."""
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    labels = [Path(c).stem for c in args.config]
    if len(set(labels)) != len(labels):
        raise SystemExit(f'config file names must be distinct (they label the runs): {labels}')
    result = {'note': 'screen, one timing run per config between two reference runs: not a score',
              'complete': False, 'started': datetime.now().isoformat(timespec='seconds'),
              'evaluator_sha256': evaluator_sha256(), 'reference': {}, 'configs': {}}

    def save():
        result['cost'] = cost()
        (out / 'sweep.json').write_text(json.dumps(result, indent=1) + '\n')

    def brief(res):
        keep = ('status', 'reason', 'weighted_ms_per_step', 'peak_memory_allocated_gib', 'process_wall_s',
                'problems', 'signatures', 'warning')
        return {**{k: res[k] for k in keep if res.get(k) is not None},
                'phases_ms': {k: v['ms_per_step'] for k, v in (res.get('phases') or {}).items()},
                'fingerprint': res.get('fingerprint')}

    def comparable(row):
        """A row whose fingerprint differs from reference a (other than the switches) cannot be compared."""
        diff = fingerprint_diff(result['reference']['a'].get('fingerprint'), row.get('fingerprint'))
        if diff and row['status'] == 'ok':
            row.update(status='INVALID', reason=f'fingerprint differs from reference a in {diff}')
        row['fingerprint_diff_vs_reference_a'] = diff
        return row

    result['inputs'] = [guarded_inputs()]
    with scratch_dir('rae-sweep-') as tmp:
        ref_tree = materialize(check_reference(), Path(tmp) / 'ref')
        tree = materialize(args.tree, Path(tmp) / 'tree')
        result['source_digest'] = tree_digest(tree)
        result['reference']['a'] = brief(time_tree(ref_tree, out, 'sweep_ref_a'))
        save()
        if result['reference']['a']['status'] != 'ok':
            result.update(complete=True, status='ERROR', reason='reference timing failed: nothing to compare with')
            save()
            raise SystemExit(f'reference timing failed: {result["reference"]["a"]}')
        for cfg, label in zip(args.config, labels):
            try:
                env = candidate_environment(cfg)
                res = time_tree(tree, out, f'sweep_{label}', cand_env=env)
            except (Exception, SystemExit) as error:
                env, res = None, {'status': 'ERROR', 'reason': f'{type(error).__name__}: {error}'}
            result['configs'][label] = comparable({'config': str(Path(cfg).resolve()), 'config_sha256': sha256_file(cfg)
                                                   if Path(cfg).exists() else None, 'environment': env, **brief(res)})
            log(f'{label}: {res["status"]} {res.get("weighted_ms_per_step")} ms')
            save()
        try:
            result['reference']['b'] = comparable(brief(time_tree(ref_tree, out, 'sweep_ref_b')))
        except (Exception, SystemExit) as error:
            result['reference']['b'] = {'status': 'ERROR', 'reason': f'{type(error).__name__}: {error}'}
        source_end_check(result, tree)
    refs = [r['weighted_ms_per_step'] for r in result['reference'].values() if r['status'] == 'ok']
    ref = statistics.mean(refs)
    result['reference'].update(weighted_ms=round(ref, 2),
                               spread=round((max(refs) - min(refs)) / ref, 4) if len(refs) == 2 else None)
    for row in result['configs'].values():
        row['speedup_vs_reference'] = round(ref / row['weighted_ms_per_step'], 4) if row['status'] == 'ok' else None
    result['inputs'].append(verify_inputs('after'))
    if not result['inputs'][-1]['ok']:
        result['status'] = 'ERROR'
        result['reason'] = 'frozen inputs changed during the sweep: rows not usable'
    result.update(complete=True, finished=datetime.now().isoformat(timespec='seconds'))
    save()
    log(f'wrote {out / "sweep.json"}')


def cmd_time(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = [guarded_inputs()]
    with scratch_dir('rae-time-') as tmp:
        tree = materialize(args.tree, Path(tmp) / 'tree')
        digest = tree_digest(tree)
        res = time_tree(tree, out, args.label, cand_env=candidate_environment(args.config))
        res['source_digest'] = digest
        source_end_check(res, tree)
    inputs.append(verify_inputs('after'))
    if not inputs[-1]['ok']:
        res['status'] = 'ERROR'
    (out / f'time_{args.label}.json').write_text(
        json.dumps({**res, 'inputs': inputs, 'cost': cost()}, indent=1) + '\n')
    log(json.dumps({k: res.get(k) for k in ('status', 'weighted_ms_per_step', 'images_per_s_local')}))


def cmd_profile(args):
    """One Nsight Systems trace per training phase (the first timing window of each); --phase all = every phase."""
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    phases = ('recon', 'disc', 'gan') if args.phase == 'all' else (args.phase,)
    src, config = args.tree, args.config
    results = {'source': str(src), 'config': str(config) if config else None,
               'evaluator_sha256': evaluator_sha256(), 'inputs': [guarded_inputs()]}
    env = candidate_environment(config)
    with scratch_dir('rae-prof-') as tmp:
        tree = materialize(src, Path(tmp) / 'tree')
        results['source_digest'] = tree_digest(tree)
        for phase in phases:
            results[phase] = time_tree(tree, out, args.label, profile_phase=phase, cand_env=env)
        source_end_check(results, tree)
    results['inputs'].append(verify_inputs('after'))
    bad = [ph for ph in phases if results[ph]['status'] != 'ok']
    changed = results.get('source_digest_end') != results['source_digest']
    results['status'] = 'ERROR' if changed or not results['inputs'][-1]['ok'] else ('INVALID' if bad else 'ok')
    results['phase_weights'] = CFG['timing']['phase_weights']
    results['cost'] = cost()
    (out / f'profile_{args.label}.json').write_text(json.dumps(results, indent=1) + '\n')
    log(f'profile {results["status"]}; ' + ', '.join(f'{ph} {results[ph]["status"]}' for ph in phases))


def cmd_baseline(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref_src = check_reference()
    inputs = [guarded_inputs()]
    with scratch_dir('rae-base-') as tmp:
        tree = materialize(ref_src, Path(tmp) / 'tree')
        # Reference against itself: the timing noise floor and the bitwise reproducibility of the reference.
        _, performance = benefit(tree, out, {})
        if args.timing_only:
            status, correctness = 'PASS', {'passed': True, 'stage': 'skipped (timing only)'}
        else:
            status, correctness, _ = evaluate(tree, out, skip_timing=True)
    result = {'baseline': True, 'note': 'reference evaluated as its own candidate',
              'evaluator_sha256': evaluator_sha256(), 'status': status,
              'correctness': correctness, 'performance': performance, 'cost': cost(),
              'inputs': inputs + [verify_inputs('after')]}
    (out / 'baseline.json').write_text(json.dumps(result, indent=1) + '\n')
    log(f'baseline {status}; wrote {out / "baseline.json"}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('check')
    k.add_argument('--tree', required=True)
    k.add_argument('--out', required=True)
    k.add_argument('--skip-timing', action='store_true')
    k.add_argument('--config', help='switch config (environment mapping) for the candidate')
    k.add_argument('--quick', action='store_true', help='gate 0 only: coverage, data loader, structure record')
    w = sub.add_parser('sweep')
    w.add_argument('--tree', required=True)
    w.add_argument('--config', action='append', required=True, help='switch config (environment mapping); repeat')
    w.add_argument('--out', required=True)
    t = sub.add_parser('time')
    t.add_argument('--tree', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--label', default='tree')
    t.add_argument('--config', help='candidate config with an environment mapping')
    f = sub.add_parser('profile')
    f.add_argument('--tree', required=True, help='a live source tree (with --config for its switches)')
    f.add_argument('--out', required=True)
    f.add_argument('--phase', choices=('all', 'recon', 'disc', 'gan'), default='all',
                   help='training phase(s) to trace; profile every phase before choosing hotspots')
    f.add_argument('--label', default='tree')
    f.add_argument('--config', help='candidate config with an environment mapping')
    b = sub.add_parser('baseline')
    b.add_argument('--out', required=True)
    b.add_argument('--timing-only', action='store_true')
    args = p.parse_args()
    {'check': cmd_check, 'sweep': cmd_sweep, 'time': cmd_time, 'profile': cmd_profile,
     'baseline': cmd_baseline}[args.cmd](args)


if __name__ == '__main__':
    main()
