"""DMD2 SDXL (4-step) distillation training bench on 4 GPUs: one command per question.

Run from the workspace root with the environment's Python. GPU runs take /tmp/dmd_gpu.lock themselves, one at a
time; do NOT wrap these commands in flock (that would deadlock).

  python bench/bench.py check   --tree <dir> --out <dir> [--config <file>] [--quick] [--skip-timing]
        the verdict (gates 0-3 below) -> <out>/check.json; --quick: gates 0 and 2 (exact checks) only, no timing,
        no trajectories; --skip-timing: no gate 1
  python bench/bench.py time    --tree <dir> --out <dir> [--config <file>]
        one timing run of the tree (2 windows of 20 iterations after 10 warmup iterations), with its speed relative
        to the frozen reference timing -> <out>/time.json
  python bench/bench.py profile --tree <dir> --out <dir> [--config <file>] [--nsys] [--iters N]
        torch.profiler over N iterations (default 5 = one update cycle) after the warmup: GPU busy fraction, top
        kernels, kernel categories, GPU time per model call -> <out>/profile.json, <out>/profile.md, a Chrome trace;
        --nsys: an Nsight Systems trace of the same span instead
  python bench/bench.py baseline --out <dir>          (formulator) the reference timed three times: noise floor

Every GPU run holds the GPU lock, waits until the visible GPUs are empty and idle (INVALID after 10 minutes), runs
the code from a fresh copy of the tree in its own process group with a timeout and a fresh TMPDIR and fresh compile
caches (cold compile), and kills what it started afterwards. Timing runs carry a fingerprint (argv, switches, math
mode, GPU UUIDs, cache policy, instrumentation, library versions), are scanned for invalid-run signatures, fail if
torch.compile compiled anything inside a timing window, and fail if the per-iteration random-number stream differs
from the frozen reference (that stream decides the backward-simulation depth, i.e. the work, of every iteration).
Every command checks the frozen inputs declared in task/formulation.json before (mismatch: nothing is measured) and
after (mismatch: ERROR), and the source digest of the tree copy before and after.

Candidate switches: a JSON file {"environment": {"NAME": "value"}} given with --config; the bench sets those
variables for the candidate's processes only. Variables the bench manages (bench.json reserved_env) are refused.

Verdict (check); each gate stops the evaluation when it fails:
  0. coverage   parse_args(), the `if __name__ == "__main__":` block and Trainer.train() outside the per-step loop
                body must equal the reference AST (the numerics adapter re-creates that setup and runs the loop body
                verbatim), and the tree's dataset classes (SDTextDataset, SDImageDatasetLMDB) must yield bitwise the
                same items and first batches on the frozen fixture -> else ERROR / INCORRECT.
  1. benefit    reference, candidate, reference: fresh `torchrun` runs of the real entrypoint main/train_sd.py
                with the frozen recipe; fingerprints must agree with the first reference run (else ERROR).
                speedup = mean reference time over the two windows / candidate time. INVALID (measure again
                later) when a reference is more than timing.frozen_band off the frozen reference, when the two
                references differ by more than timing.pair_band in a window, or when a run's window ratio is more
                than timing.window_band off the frozen reference's ratio. speedup <= max(1.02, 1 + 2 x reference
                spread) -> UNAVAILABLE (no_benefit), rejected without a precision evaluation.
  2. bitwise    one 1-step case per training phase ('gen': generator + guidance update; 'guidance': guidance-only
                update), recorded in deterministic mode, one process per case. Exact checks (phase, RNG fingerprints
                before/after, forward call counts, invariants incl. learning rates and data digests, tensor names and
                shapes, finiteness, cross-rank pattern) must hold, else INCORRECT. Every recorded tensor (loss/,
                out/, grad/, delta/, optim/, buffer/) bitwise identical -> PASS (N0). Tensors larger than
                numerics.sketch_dim are stored as linear CountSketches, which estimate distances without keeping
                3.4 GB of weights per step.
  3. statistical  only when gate 2 is not bitwise: paired-seed trajectories (losses only, deterministic kernels,
                compressed warmup, natural gen/guidance schedule), one process per seed, on the frozen seeds 3 then
                6; delta = 2 x the reference's seed sigma per endpoint (frozen from 6 reference seeds). With 3 seeds:
                an endpoint beyond delta and significant (paired t, p < 0.05) -> INCORRECT, all within -> PASS,
                otherwise 6 seeds; with 6 seeds any endpoint beyond delta -> INCORRECT, else PASS.
  Tiers: N0 everything bitwise with the same dtypes and math modes; N2 a dtype or math mode (TF32, reduced-precision
  reductions, SDPA backends) changed; otherwise N1. Above bench.json agent_max_tier -> INCORRECT (needs sign-off).
  Per-tensor tolerance gates (bench/thresholds.json) are reported as diagnostics only.
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
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent                       # workspace


def _expand(value):
    """bench.json and task/formulation.json name machine paths through ${ASSETS} (the task's read-only assets)."""
    if isinstance(value, str) and '$' in value:
        out = os.path.expandvars(value)
        if '$' in out:
            raise SystemExit(f'{value}: set ASSETS (and any other variable it names) before running the bench')
        return out
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


CFG = _expand(json.loads((BENCH / 'bench.json').read_text()))
sys.path.insert(0, str(BENCH))
import recipe  # noqa: E402

LOCK = Path(CFG['gpu_lock'])
PY = Path(CFG['python'])
TORCHRUN = PY.parent / 'torchrun'
TIERS = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5']
SCRATCH = Path(CFG.get('scratch', '/tmp'))
TIMEOUT = CFG['timeouts_s']
RESERVED = set(CFG['reserved_env'])
SIGNATURES = [(re.compile(x['regex']), x['reason']) for x in CFG.get('invalid_run_signatures', [])]
COST = {'gpu_seconds': 0.0, 'gpu_runs': 0}
TREE_SKIP = {'.git', '__pycache__', 'wandb', 'outputs', 'figures'}


def path(key):
    p = Path(_expand(key))
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
    """Joint hash of the frozen evaluator: numerics/, bench/ code and config, frozen records' manifests, task/."""
    files = sorted(p for p in (ROOT / 'numerics').rglob('*.py'))
    files += sorted(p for p in BENCH.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                    and p.suffix in ('.py', '.json', '.yaml') and 'ref' not in p.parts)
    files += sorted(p for p in (BENCH / 'records').rglob('*.json'))
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode() + b'\0' + sha256_file(p).encode() + b'\n')
    return h.hexdigest()


# ---------------------------------------------------------------- GPU runs

class GpuLock:
    def __init__(self, what, timeout=8 * 3600):
        self.what, self.timeout = what, timeout

    def __enter__(self):
        self.f = open(LOCK, 'a+')
        t0, said = time.time(), False
        while True:
            try:
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not said:
                    log(f'waiting for {LOCK} ({self.what}); if this command is wrapped in flock, unwrap it')
                    said = True
                if time.time() - t0 > self.timeout:
                    raise SystemExit(f'could not get {LOCK} within {self.timeout}s for {self.what}')
                time.sleep(5)
        self.t0 = time.time()
        log(f'GPU lock held: {self.what}')
        return self

    def __exit__(self, *_):
        COST['gpu_seconds'] += (time.time() - self.t0) * CFG['world_size']
        COST['gpu_runs'] += 1
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def cost():
    return {'gpu_hours': round(COST['gpu_seconds'] / 3600, 3), 'gpu_runs': COST['gpu_runs'],
            'method': f'GPU lock hold time x {CFG["world_size"]} cards (startup, warmup and failed runs included)'}


def scratch_dir(prefix):
    free = shutil.disk_usage(SCRATCH).free / 2**30
    if free < CFG.get('min_scratch_free_gib', 200):
        raise SystemExit(f'{SCRATCH} has only {free:.1f} GiB free (< {CFG.get("min_scratch_free_gib", 200)}); '
                         f'clean up before measuring (every run writes a ~26 GB initial checkpoint there)')
    return tempfile.TemporaryDirectory(prefix=prefix, dir=SCRATCH)


def gpu_state():
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=60)
    out = []
    for line in q.stdout.strip().splitlines():
        idx, uid, mem, util = [x.strip() for x in line.split(',')]
        out.append({'index': int(idx), 'uuid': uid, 'memory_mib': int(mem), 'util': int(util)})
    return out


def wait_idle(what, patience=600):
    t0 = time.time()
    while True:
        state = gpu_state()
        busy = [g for g in state if g['memory_mib'] > 1024 or g['util'] > 5]
        if not busy:
            return state, None
        if time.time() - t0 > patience:
            return state, f'GPUs busy for {patience}s before {what}: {busy}'
        time.sleep(15)


def _tagged(tag):
    me, pids = os.getuid(), []
    mark = f'DMD_BENCH_RUN={tag}'.encode()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            if d.stat().st_uid == me and mark in (d / 'environ').read_bytes().split(b'\0'):
                pids.append(int(d.name))
        except OSError:
            continue
    return pids


def _kill_tagged(tag):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = _tagged(tag)
        if not pids:
            return 0
        for p in pids:
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        time.sleep(5)
    return len(_tagged(tag))


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def base_env(tmp, cand_env=None, pythonpath=None):
    """Environment of every bench-launched process: the caller's environment minus reserved variables, then the
    candidate's switches, then what the bench manages (scratch, caches, offline hub, devices)."""
    env = {k: v for k, v in os.environ.items() if k not in RESERVED and not k.startswith('DMD_BENCH')}
    env.update(cand_env or {})
    caches = Path(tmp) / 'caches'
    env.update({
        'TMPDIR': str(tmp), 'TMP': str(tmp), 'TEMP': str(tmp),
        'TORCHINDUCTOR_CACHE_DIR': str(caches / 'inductor'), 'TRITON_CACHE_DIR': str(caches / 'triton'),
        'CUDA_CACHE_PATH': str(caches / 'cuda'), 'XDG_CACHE_HOME': str(caches / 'xdg'),
        'MPLCONFIGDIR': str(caches / 'mpl'),
        'HF_HOME': CFG['hf_home'], 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
        'WANDB_MODE': 'disabled', 'PYTHONNOUSERSITE': '1',
        'PATH': f'{PY.parent}:{env.get("PATH", "/usr/local/bin:/usr/bin:/bin")}',
    })
    env.pop('PYTHONPATH', None)
    if pythonpath:
        env['PYTHONPATH'] = ':'.join(str(p) for p in pythonpath)
    for v in ('inductor', 'triton', 'cuda', 'xdg', 'mpl'):
        (caches / v).mkdir(parents=True, exist_ok=True)
    return env


def run_gpu(cmd, cwd, env, log_file, what, timeout):
    """One GPU job under the lock: idle check, own process group, timeout, leftovers killed. Returns a dict with
    status (ok / TIMEOUT / INVALID), returncode, wall seconds and the GPU state before and after."""
    tag = uuid.uuid4().hex[:12]
    with GpuLock(what):
        before, problem = wait_idle(what)
        if problem:
            return {'status': 'INVALID', 'reason': problem, 'returncode': None, 'gpus_before': before}
        env = dict(env, DMD_BENCH_RUN=tag)
        if CFG.get('cpu_affinity'):        # the CPUs of the GPUs' NUMA node, for every run alike
            cmd = ['taskset', '-c', CFG['cpu_affinity'], *cmd]
        t0 = time.time()
        status = None
        with open(log_file, 'w') as lf:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                rc, status = None, 'TIMEOUT'
        wall = time.time() - t0
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        left = _kill_tagged(tag)
        after, _ = wait_idle(f'{what} (after)', patience=60)
    res = {'status': status or 'ok', 'returncode': rc, 'wall_s': round(wall, 1), 'gpus_before': before,
           'gpus_after': after}
    if status == 'TIMEOUT':
        res['reason'] = f'{what} exceeded {timeout}s; process group killed; see {Path(log_file).name}'
    if left:
        res['warning'] = f'{left} processes of the run survived SIGKILL'
    if any(g['memory_mib'] > 1024 for g in after or []):
        res['warning'] = (res.get('warning', '') + ' GPU memory still in use 60s after the run').strip()
    return res


def signatures(log_file):
    hits, text = [], Path(log_file).read_text(errors='replace') if Path(log_file).exists() else ''
    for pattern, reason in SIGNATURES:
        m = pattern.search(text)
        if m:
            line = text[text.rfind('\n', 0, m.start()) + 1:text.find('\n', m.end())]
            hits.append({'reason': reason, 'line': line.strip()[:300]})
    return hits


def launch(argv, port=None):
    """`torchrun` as upstream's SDv1.5 script does, on one node with CFG['world_size'] processes and a fresh port
    (the sandbox has its own network namespace: 127.0.0.1 only, and the hostname does not resolve)."""
    return [str(TORCHRUN), f'--nnodes=1', f'--nproc-per-node={CFG["world_size"]}', '--master-addr', '127.0.0.1',
            '--master-port', str(port or free_port()), *argv]


# ---------------------------------------------------------------- inputs and sources

def tree_digest(tree):
    h = hashlib.sha256()
    tree = Path(tree)
    for f in sorted(tree.rglob('*')):
        rel = f.relative_to(tree)
        if any(part in TREE_SKIP for part in rel.parts) or f.is_symlink() or not f.is_file():
            continue
        h.update(str(rel).encode() + b'\0' + sha256_file(f).encode() + b'\n')
    return h.hexdigest()


def file_fingerprint(p, kind):
    if kind == 'sha256':
        return sha256_file(p)
    st = Path(p).stat()
    return f'{st.st_size}:{st.st_mtime_ns}'


def verify_inputs(when):
    """Every frozen input file of task/formulation.json: sha256 for files below 400 MB, size:mtime for the large
    weight files (the assets are mounted read-only for the optimizer anyway)."""
    spec = json.loads((ROOT / 'task' / 'formulation.json').read_text())['inputs']
    t0, bad = time.time(), {}
    for rel, want in spec.items():
        p = path(rel)
        got = file_fingerprint(p, want['kind']) if p.exists() else None
        if got != want['value']:
            bad[rel] = {'want': want['value'], 'got': got}
    log(f'inputs {when}: {len(spec) - len(bad)}/{len(spec)} match the formulation ({time.time() - t0:.0f}s)')
    return {'when': when, 'ok': not bad, 'mismatches': bad, 'checked': len(spec), 'seconds': round(time.time() - t0, 1)}


def freeze_inputs(paths, big=400 << 20):
    """Formulator: fingerprints of every file under the given paths (symlinks resolved)."""
    out = {}
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(f for f in root.rglob('*') if f.is_file() and
                                                      not f.name.endswith('.lock') and f.name != 'lock.mdb')
        for f in files:
            kind = 'stat' if f.stat().st_size > big else 'sha256'
            out[str(f)] = {'kind': kind, 'value': file_fingerprint(f, kind)}
    return out


def check_reference():
    ref = path(CFG['reference'])
    git = ['git', '-c', 'safe.directory=*', '-C', str(ref)]
    dirty = subprocess.run([*git, 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                           text=True).stdout.strip()
    head = subprocess.run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if dirty or head != CFG['reference_commit']:
        raise SystemExit(f'reference tree is not the frozen upstream commit (head {head}, dirty: {bool(dirty)})')
    return ref


def materialize(src, dest):
    def ignore(directory, names):
        return [n for n in names if n in TREE_SKIP or n.endswith('.pyc')]
    shutil.copytree(src, dest, ignore=ignore, symlinks=False)
    for d in [Path(dest), *(p for p in Path(dest).rglob('*') if p.is_dir())]:
        d.chmod(d.stat().st_mode | 0o700)
    return Path(dest)


def candidate_environment(config_file):
    if not config_file:
        return {}
    data = json.loads(Path(config_file).read_text())
    extra = sorted(set(data) - {'environment'})
    if extra:
        raise SystemExit(f'{config_file}: only an "environment" mapping is accepted, found {extra}')
    env = {str(k): str(v) for k, v in (data.get('environment') or {}).items()}
    bad = sorted(k for k in env if k in RESERVED or k.startswith('DMD_BENCH'))
    if bad:
        raise SystemExit(f'{config_file}: {bad} are managed by the bench and cannot be candidate switches')
    return env


def versions():
    out = {'python': sys.version.split()[0]}
    for pkg in ('torch', 'triton', 'diffusers', 'transformers', 'accelerate', 'xformers', 'flash_attn'):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


FINGERPRINT_KEYS = ('argv', 'math_mode', 'gpus', 'compile_cache', 'instrumentation', 'versions', 'timing_plan')


def fingerprint_diff(a, b):
    if a is None or b is None:
        return ['missing fingerprint']
    return [k for k in FINGERPRINT_KEYS if a.get(k) != b.get(k)]


# ---------------------------------------------------------------- gate 0: coverage

def setup_fingerprint(tree):
    """AST of parse_args(), the __main__ block and Trainer.train() with the per-step loop body blanked."""
    module = ast.parse((Path(tree) / recipe.ENTRY).read_text())
    parts, loops = {}, 0
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'parse_args':
            parts['parse_args'] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and \
                isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__':
            parts['__main__'] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.ClassDef) and node.name == 'Trainer':
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == 'train':
                    for inner in ast.walk(fn):
                        if isinstance(inner, ast.For) and isinstance(inner.target, ast.Name) \
                                and inner.target.id == 'index':
                            inner.body = [ast.Pass()]
                            loops += 1
                    parts['Trainer.train'] = ast.dump(fn, include_attributes=False)
    return parts, loops


def setup_guard(ref, tree):
    want, _ = setup_fingerprint(ref)
    try:
        got, loops = setup_fingerprint(tree)
    except (OSError, SyntaxError) as exc:
        return [f'{recipe.ENTRY} unreadable: {exc}']
    problems = []
    if loops != 1:
        problems.append(f'expected exactly one `for index in ...` loop in Trainer.train(), found {loops}')
    for name in ('parse_args', '__main__', 'Trainer.train'):
        if got.get(name) != want.get(name):
            problems.append(f'{name} differs from the reference' +
                            (' outside the per-step loop body' if name == 'Trainer.train' else ''))
    return problems


LOADER_PROBE = r'''
import hashlib, json, sys, torch
sys.path.insert(0, ".")
from transformers import CLIPTokenizer
from main.utils import SDTextDataset
from main.sd_image_dataset import SDImageDatasetLMDB
model_id, prompts, latents = sys.argv[1], sys.argv[2], sys.argv[3]
t1 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text = SDTextDataset(prompts, t1, is_sdxl=False)
real = SDImageDatasetLMDB(latents, is_sdxl=False, tokenizer_one=t1)
def dig(d):
    h = hashlib.sha256()
    for k in sorted(d):
        v = d[k]
        h.update(k.encode())
        if isinstance(v, torch.Tensor):
            h.update(str(v.dtype).encode() + str(tuple(v.shape)).encode())
            h.update(v.contiguous().view(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()
out = {"lengths": [len(text), len(real)], "items": {}, "batches": {}}
for name, ds in (("text", text), ("real", real)):
    n = len(ds)
    out["items"][name] = [dig(ds[i]) for i in sorted({0, 1, 2, 3, 1000 % n, n // 2, n - 1})]
    g = torch.Generator().manual_seed(0)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, drop_last=True, generator=g, num_workers=0)
    it = iter(dl)
    out["batches"][name] = [dig(next(it)) for _ in range(3)]
print(json.dumps(out))
'''


def loader_digest(tree):
    with scratch_dir('dmd-ld-') as tmp:
        env = base_env(tmp)
        r = subprocess.run([str(PY), '-c', LOADER_PROBE, CFG['model_id'], str(path(CFG['fixture']['prompts'])),
                            str(path(CFG['fixture']['latents']))], cwd=tree, env=env, capture_output=True, text=True,
                           timeout=1800)
    if r.returncode:
        return None, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1]), None


def coverage(tree):
    """Gate 0 (CPU): setup guard and dataset digests. Returns (status, correctness) or None when it holds."""
    ref_src = check_reference()
    problems = setup_guard(ref_src, tree)
    if problems:
        return 'ERROR', {'passed': False, 'stage': 'setup_guard', 'problems': problems,
                         'note': 'the numerics adapter re-creates this setup from the reference and runs only the '
                                 'per-step loop body of the tree; move the change into code the trainer calls '
                                 '(Trainer.__init__, train_one_step, the model modules) or into the loop body'}
    frozen = json.loads((ROOT / 'task' / 'loader_digest.json').read_text())
    got, err = loader_digest(tree)
    if got != frozen:
        return 'INCORRECT', {'passed': False, 'stage': 'loader_check', 'error': err,
                             'note': 'the dataset classes yield different items or first batches than the reference '
                                     'on the frozen fixture'}
    return None


# ---------------------------------------------------------------- gate 2: bitwise records

def adapter_kwargs(tree, scratch, variant=None):
    kw = {'repo': str(tree), 'prompts': str(path(CFG['fixture']['prompts'])),
          'latents': str(path(CFG['fixture']['latents'])), 'ckpt': str(path(CFG['fixture']['ckpt'])),
          'scratch': str(scratch)}
    if variant:
        kw['variant'] = variant
    return kw


def record_cases(tree, rec, cases, logs_dir, cand_env, label, variant=None, losses_only=False, sketch=True):
    """One `numerics record` process per case (a second trainer in the same process would not get all GPU memory
    back). Returns None on success, else an error dict."""
    n = CFG['numerics']
    for case in cases:
        with scratch_dir('dmd-r-') as tmp:
            cases_file = Path(tmp) / 'case.json'
            cases_file.write_text(json.dumps([case]))
            env = base_env(tmp, cand_env, pythonpath=[ROOT])
            cmd = launch(['-m', 'numerics', 'record', '--adapter', n['adapter'], '--adapter-kwargs',
                          json.dumps(adapter_kwargs(tree, Path(tmp) / 'train', variant)), '--cases', str(cases_file),
                          *(['--losses-only'] if losses_only else
                            (['--sketch-dim', str(n['sketch_dim'])] if sketch else [])),
                          # no --dtype-trace: under its dispatch mode compiled regions fall back to eager
                          '--math-mode', CFG['math_mode'], '--deterministic', '--out', str(rec)])
            log_file = Path(logs_dir) / f'{label}_{case["name"]}.log'
            r = run_gpu(cmd, ROOT, env, log_file, f'{label}: case {case["name"]}',
                        TIMEOUT['trajectory' if losses_only else 'numerics'])
        if r['status'] != 'ok' or r['returncode']:
            return {'status': 'INVALID' if r['status'] == 'INVALID' else 'ERROR',
                    'reason': r.get('reason') or f'record exited {r["returncode"]}; see {log_file.name}'}
    return None


def missing_tensor_failures(report):
    """Judge errors that are really evidence of different work, turned into exact failures.

    The recorder only records a gradient for a parameter that has one. A candidate that drops a loss term's
    gradient (so some parameters get none) therefore produces a strict subset of the reference's tensors, which the
    judge reports as an error ("the records do not line up") rather than as a wrong result. Here a step whose
    tensor set is missing reference tensors and adds none of its own is an exact failure: the candidate did not do
    the work that produced them. A step with extra or renamed tensors stays an error -- that is an adapter/layout
    problem, not a verdict.
    """
    out = []
    for err in report.get('errors') or []:
        if isinstance(err, dict) and err.get('missing') and not err.get('unexpected'):
            out.append({'where': err['where'], 'check': 'missing_tensors',
                        'note': 'the candidate produced none of these tensors (e.g. no gradient for parameters the '
                                'reference computes one for): it did not do that work',
                        'missing': err['missing'][:20]})
    return out


def numerics(tree, out, cand_env):
    n = CFG['numerics']
    rec = out / 'records'
    shutil.rmtree(rec, ignore_errors=True)
    cases = json.loads(path(n['cases']).read_text())
    failed = record_cases(tree, rec, cases, out, cand_env, 'record')
    if failed:
        return failed
    report = out / 'compare.json'
    env = base_env(out, pythonpath=[ROOT])
    r = subprocess.run([str(PY), '-m', 'numerics', 'compare', '--ref', str(path(n['ref'])), '--cand', str(rec),
                        '--thresholds', str(path(n['thresholds'])),
                        *[a for x in n.get('exclude', []) for a in ('--exclude', f'{x["regex"]}={x["reason"]}')],
                        '--out', str(report)], cwd=ROOT, env=env, capture_output=True, text=True)
    (out / 'compare.log').write_text(r.stdout + r.stderr)
    shutil.rmtree(out / 'caches', ignore_errors=True)
    if not report.exists():
        return {'status': 'ERROR', 'reason': f'compare exited {r.returncode}; see compare.log'}
    data = json.loads(report.read_text())
    if not n.get('keep_records'):
        shutil.rmtree(rec, ignore_errors=True)
    data['exact_failures'] = (data.get('exact_failures') or []) + missing_tensor_failures(data)
    summary = {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in data.items()
               if k not in ('status', 'measured_tier')}
    bitwise = (data.get('checked') == data.get('bitwise_identical') and not data.get('dtype_changes')
               and not data.get('precision_mode_changes') and data.get('bitwise_trusted', False))
    return {'status': data.get('status'), 'measured_tier': data.get('measured_tier'), 'report': str(report),
            'errors': len(data.get('errors') or []), 'exact_failures': data.get('exact_failures') or [],
            'bitwise': bitwise, 'summary': summary}


# ---------------------------------------------------------------- gate 3: trajectories

def traj_case(spec, seed):
    return dict(spec['case'], name=f'traj_s{seed}', seed=seed)


def trajectories(tree, out, cand_env):
    tj = CFG['trajectory']
    spec = json.loads(path(tj['spec']).read_text())
    frozen_path = path(tj['frozen'])
    seeds = json.loads(frozen_path.read_text())['reference_seeds']
    rec, done, report = out / 'trajectory', [], None
    schedule = tj['seed_schedule']
    for idx, n in enumerate(schedule):
        final = idx == len(schedule) - 1
        new = [x for x in seeds[:n] if x not in done]
        failed = record_cases(tree, rec, [traj_case(spec, s) for s in new], out, cand_env, 'traj', losses_only=True)
        if failed:
            return {'verdict': 'error', **failed}
        done += new
        report_path = out / f'traj_report_{n}.json'
        env = base_env(out, pythonpath=[ROOT])
        subprocess.run([str(PY), '-m', 'numerics', 'traj-compare', '--frozen', str(frozen_path), '--cand', str(rec),
                        *(['--final'] if final else []), '--out', str(report_path)],
                       cwd=ROOT, env=env, capture_output=True, text=True)
        shutil.rmtree(out / 'caches', ignore_errors=True)
        if not report_path.exists():
            return {'verdict': 'error', 'reason': 'traj-compare produced no report'}
        report = json.loads(report_path.read_text())
        report['report'] = str(report_path)
        if report['verdict'] != 'inconclusive':
            break
    return report


# ---------------------------------------------------------------- gate 1: timing

def time_tree(tree, out, label, cand_env=None, profile=None, profile_iters=5):
    t = CFG['timing']
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f'timing_{label}'
    for f in out.glob(f'timing_{label}.r*.json'):
        f.unlink()
    with scratch_dir('dmd-t-') as tmp:
        train_args = recipe.argv(Path(tmp) / 'train', path(CFG['fixture']['prompts']), path(CFG['fixture']['latents']),
                                 path(CFG['fixture']['ckpt']))
        hook_args = ['--warmup', str(t['warmup']), '--window', str(t['window']), '--repeats', str(t['repeats']),
                     *(['--profile', profile, '--profile-iters', str(profile_iters)] if profile else [])]
        argv = launch([str(BENCH / 'timing.py'), '--out', str(prefix), *hook_args, '--', *train_args])
        stable_argv = [f'torchrun --nproc-per-node={CFG["world_size"]}', 'timing.py', *hook_args, '--',
                       *[x.replace(str(tmp), '<scratch>') for x in train_args]]
        env = base_env(tmp, cand_env)
        if profile == 'nsys':
            nsys = shutil.which('nsys') or CFG.get('nsys')
            if not nsys or not Path(nsys).exists():
                raise SystemExit('nsys not found')
            argv = [nsys, 'profile', '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                    '--trace=cuda,nvtx,osrt,cublas,cudnn', '--sample=none', '--cuda-memory-usage=false',
                    '--force-overwrite=true', '-o', str(out / f'nsys_{label}'), *argv]
        log_file = out / f'timing_{label}.log'
        what = f'timing {label}{f" ({profile})" if profile else ""}'
        r = run_gpu(argv, tree, env, log_file, what, TIMEOUT['profile' if profile else 'timing'])
    plan = [t['warmup'], t['window'], t['repeats']] if not profile else [t['warmup'], profile_iters, 1]
    fingerprint = {'argv': stable_argv, 'candidate_environment': cand_env or {},
                   'math_mode': CFG['math_mode'], 'gpus': [(g['index'], g['uuid']) for g in r.get('gpus_before') or []],
                   'compile_cache': 'cold (fresh per run)', 'instrumentation': profile or 'none',
                   'timing_plan': plan, 'versions': versions()}
    run ={'run_status': r['status'], 'wall_s': r.get('wall_s'), 'fingerprint': fingerprint, 'log': str(log_file),
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
    return {**summarize_timing(ranks, r['wall_s'], profile, profile_iters), **run, '_ranks': ranks}


def summarize_timing(ranks, wall, profile=None, profile_iters=5):
    t = CFG['timing']
    ratio = recipe.UPDATE_RATIO
    repeats = 1 if profile else t['repeats']
    size = profile_iters if profile else t['window']
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
        first = t['warmup'] + w['k'] * size
        expect = ['gen' if (first + i) % ratio == 0 else 'guidance' for i in range(size)]
        if w['kinds'] != expect:
            problems.append(f'window {w["k"]}: iteration kinds {w["kinds"]} differ from the update schedule '
                            f'(a generator update every {ratio} iterations)')
    secs = [w['seconds'] for w in ws]
    total = sum(secs) if secs else None
    frozen = None
    ref_file = ROOT / 'task' / 'timing_ref.json'
    rng_problem = None
    if not profile and ref_file.exists():
        frozen = json.loads(ref_file.read_text())
        n_check = t['warmup'] + t['window'] * t['repeats']
        for x in ranks:
            want = frozen['rng_fingerprints'][str(x['rank'])][:n_check]
            if x['rng_fingerprints'][:n_check] != want:
                k = next((i for i, (a, b) in enumerate(zip(x['rng_fingerprints'], want)) if a != b), None)
                rng_problem = (f'rank {x["rank"]}: the random-number stream differs from the reference from iteration '
                               f'{k} on (random numbers drawn in a different amount or order change the work of '
                               f'every later iteration)')
                break
    if rng_problem:
        problems.append(rng_problem)
    res = {
        'status': 'ok' if not problems and secs else 'ERROR', 'problems': problems,
        'window_s': [round(s, 3) for s in secs], 'total_s': round(total, 3) if total else None,
        'iterations_timed': size * repeats,
        's_per_iteration': round(total / (size * len(secs)), 4) if secs else None,
        's_per_cycle': round(total / (size * len(secs)) * ratio, 3) if secs else None,
        'peak_memory_allocated_gib': round(max(x['peak_memory_allocated_bytes'] for x in ranks) / 2**30, 2),
        'peak_memory_reserved_gib': round(max(x['peak_memory_reserved_bytes'] for x in ranks) / 2**30, 2),
        'startup_to_first_iteration_s': round(max(x['startup_to_first_iteration_s'] or 0 for x in ranks), 1),
        'process_wall_s': round(wall, 1),
    }
    if len(secs) == 2:
        res['window_ratio'] = round(secs[0] / secs[1], 4)
        if frozen:
            res['window_ratio_vs_frozen'] = round(res['window_ratio'] / frozen['window_ratio'] - 1, 4)
    return res


def strip(run):
    return {k: v for k, v in run.items() if k != '_ranks'} if run else run


def benefit(tree, out, cand_env):
    t = CFG['timing']
    frozen = json.loads((ROOT / 'task' / 'timing_ref.json').read_text())
    ref_src = check_reference()
    with scratch_dir('dmd-ref-') as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / 'reference')
        ref_a = time_tree(ref_tree, out, 'ref_a')
        cand = time_tree(tree, out, 'cand', cand_env=cand_env) if ref_a['status'] == 'ok' else None
        ref_b = time_tree(ref_tree, out, 'ref_b') if cand and cand['status'] == 'ok' else None
    runs = {'ref_a': strip(ref_a), 'cand': strip(cand), 'ref_b': strip(ref_b)}
    bad = {k: v for k, v in runs.items() if v is None or v['status'] != 'ok'}
    if bad:
        status = 'INVALID' if any(v and v['status'] in ('INVALID', 'TIMEOUT') for v in bad.values()) else 'ERROR'
        return status, {'speedup': None, 'runs': runs}
    diffs = {k: fingerprint_diff(ref_a.get('fingerprint'), v.get('fingerprint')) for k, v in runs.items() if k != 'ref_a'}
    for k in diffs:   # the candidate's switches are what the comparison varies
        diffs[k] = [d for d in diffs[k] if d != 'candidate_environment']
    badfp = {k: d for k, d in diffs.items() if d}
    if badfp:
        return 'ERROR', {'speedup': None, 'runs': runs, 'fingerprint_diffs': diffs,
                         'reason': f'timing runs not comparable: fingerprint differs from ref_a in {badfp}'}
    ra, rb, c = ref_a['total_s'], ref_b['total_s'], cand['total_s']
    ref = (ra + rb) / 2
    perf = {'reference_s_per_iteration': round(ref / ref_a['iterations_timed'], 4),
            'candidate_s_per_iteration': cand['s_per_iteration'], 'speedup': round(ref / c, 4),
            'reference_spread': round(abs(ra - rb) / ref, 4), 'frozen_reference_s_per_iteration':
            frozen['s_per_iteration'], 'runs': runs, 'fingerprint_diffs': diffs}
    problems = []
    for name, run in (('ref_a', ref_a), ('ref_b', ref_b)):
        off = run['total_s'] / frozen['total_s'] - 1
        if abs(off) > t['frozen_band']:
            problems.append(f'{name} took {run["total_s"]:.1f}s, {off:+.1%} off the frozen reference '
                            f'{frozen["total_s"]:.1f}s (> {t["frozen_band"]:.0%})')
    pair = [abs(a - b) / ((a + b) / 2) for a, b in zip(ref_a['window_s'], ref_b['window_s'])]
    perf['reference_pair_window_spread'] = [round(x, 4) for x in pair]
    if max(pair) > t['pair_band']:
        problems.append(f'the two reference runs differ by {max(pair):.1%} in a window (> {t["pair_band"]:.0%})')
    for name, run in (('ref_a', ref_a), ('cand', cand), ('ref_b', ref_b)):
        band = t['window_band'] if name != 'cand' else t['window_band_candidate']
        dev = run.get('window_ratio_vs_frozen')
        if dev is not None and abs(dev) > band:
            problems.append(f'{name}: window ratio {run["window_ratio"]} is {dev:+.1%} off the frozen reference '
                            f'ratio {frozen["window_ratio"]} (> {band:.0%}): one window was disturbed')
    perf['problems'] = problems
    if problems:
        return 'INVALID', perf
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * perf['reference_spread'])
    perf['benefit'] = {'speedup': perf['speedup'], 'speedup_bar': round(bar, 4),
                       'peak_memory_allocated_gib': cand['peak_memory_allocated_gib'],
                       'reference_peak_memory_allocated_gib': round((ref_a['peak_memory_allocated_gib'] +
                                                                     ref_b['peak_memory_allocated_gib']) / 2, 2),
                       'passed': perf['speedup'] > bar}
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf


# ---------------------------------------------------------------- verdict

def evaluate(tree, out, skip_timing=False, cand_env=None, quick=False):
    """Gates 0 -> 1 -> 2 -> 3. Returns (status, correctness, performance)."""
    cand_env = cand_env or {}
    failed = coverage(tree)
    if failed:
        return failed[0], failed[1], None
    if quick:
        num = numerics(tree, out, cand_env)
        # exact failures first: a candidate that does different work is INCORRECT, even when the records then also
        # fail to line up (a missing forward call removes the tensors it would have produced)
        if num.get('exact_failures'):
            return 'INCORRECT', {'passed': False, 'stage': 'structure', 'verdict': 'structure', 'numerics': num}, None
        if num.get('errors') or num.get('status') not in ('PASS', 'INCORRECT'):
            return 'ERROR', {'passed': False, 'stage': 'structure', 'numerics': num}, None
        return 'PASS', {'passed': True, 'stage': 'structure (quick: gates 1 and 3 not run)',
                        'bitwise': num['bitwise'], 'measured_tier': num.get('measured_tier'), 'numerics': num}, None
    performance = None
    if not skip_timing:
        status, performance = benefit(tree, out, cand_env)
        if status in ('ERROR', 'INVALID'):
            return status, {'passed': False, 'stage': 'benefit',
                            'note': (performance or {}).get('reason') or '; '.join((performance or {}).get('problems')
                                                                                 or []) or 'timing failed'}, performance
        if status == 'UNAVAILABLE':
            performance['measured_speedup'] = performance.pop('speedup')
            return 'UNAVAILABLE', {'passed': None, 'stage': 'benefit', 'verdict': 'no_benefit',
                                   'note': 'not faster than the reference: rejected without a precision evaluation'}, \
                performance
    num = numerics(tree, out, cand_env)
    tier = num.get('measured_tier')
    correctness = {'stage': 'bitwise', 'measured_tier': tier, 'numerics': num}
    if num.get('exact_failures'):
        return 'INCORRECT', dict(correctness, passed=False, verdict='structure',
                                 note='exact checks failed: the candidate does different work'), performance
    if num.get('errors') or num.get('status') not in ('PASS', 'INCORRECT'):
        return 'ERROR', dict(correctness, passed=False, note='numerics could not compare the records'), performance
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
        return ('INVALID' if traj.get('status') == 'INVALID' else 'ERROR'), \
            dict(correctness, passed=False, verdict='error'), performance
    note = ('an endpoint deviates from the reference beyond 2 x its seed spread' if traj['verdict'] == 'different'
            else 'no verdict after the full seed schedule')
    return 'INCORRECT', dict(correctness, passed=False, verdict=traj['verdict'], note=note), performance


def guarded(out, fn, tree_src=None):
    """Inputs checked before (nothing measured on a mismatch) and after; the tree is copied first and its digest
    checked again at the end (the code must not write into its own tree)."""
    out.mkdir(parents=True, exist_ok=True)
    start = verify_inputs('start')
    if not start['ok']:
        return {'status': 'ERROR', 'stage': 'inputs', 'inputs': [start],
                'note': 'a frozen input differs from task/formulation.json; nothing was measured'}
    if tree_src is None:
        res = fn(None)
        digest = None
    else:
        with scratch_dir('dmd-src-') as tmp:
            tree = materialize(tree_src, Path(tmp) / 'tree')
            digest = tree_digest(tree)
            try:
                res = fn(tree)
            except SystemExit as exc:
                res = {'status': 'ERROR', 'note': f'the evaluation stopped: {exc}'}
            res['source_digest'] = digest
            if tree_digest(tree) != digest:
                res['status'] = 'ERROR'
                res['note'] = 'the tree changed while it was being measured (the code wrote into its own tree)'
    end = verify_inputs('end')
    res['inputs'] = [start, end]
    if not end['ok']:
        res['status'] = 'ERROR'
        res['note'] = 'a frozen input changed during the run'
    return res


def finish(out, name, res):
    res['cost'] = cost()
    res['evaluator_sha256'] = evaluator_sha256()
    res['finished'] = datetime.now().isoformat(timespec='seconds')
    (out / name).write_text(json.dumps(res, indent=1, default=str) + '\n')
    log(f'wrote {out / name}: status {res.get("status")}')
    summary = {'status': res.get('status')}
    perf = res.get('performance')
    if isinstance(perf, dict):
        summary['speedup'] = perf.get('speedup', perf.get('measured_speedup'))
    for k in ('speedup_vs_frozen_reference', 's_per_iteration'):
        if k in res:
            summary[k] = res[k]
    print(json.dumps(summary))
    return 0 if res.get('status') in ('PASS', 'ok') else 1


def cmd_check(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        status, correctness, performance = evaluate(tree, out, skip_timing=a.skip_timing, cand_env=cand_env,
                                                    quick=a.quick)
        return {'status': status, 'tree': str(Path(a.tree).resolve()), 'config': a.config, 'environment': cand_env,
                'correctness': correctness, 'performance': performance}
    return finish(out, 'check.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_time(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        res = strip(time_tree(tree, out, 'tree', cand_env=cand_env))
        ref_file = ROOT / 'task' / 'timing_ref.json'
        if ref_file.exists():
            frozen = json.loads(ref_file.read_text())
            if res['status'] == 'ok':
                res['speedup_vs_frozen_reference'] = round(frozen['total_s'] / res['total_s'], 4)
            res['frozen_reference_s_per_iteration'] = frozen['s_per_iteration']
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)
    return finish(out, 'time.json', guarded(out, run, Path(a.tree).resolve()))


CATEGORIES = [
    ('nccl', r'nccl'),
    ('attention', r'fmha|flash|attention|mem_eff|cutlass.*Attention'),
    ('gemm', r'gemm|cutlass|sm90_xmma|sm80_xmma|cublas|ampere_|hopper_|s16816|nvjet'),
    ('conv', r'conv|implicit_gemm|winograd|cudnn|dgrad|wgrad|fprop'),
    ('norm', r'norm'),
    ('optimizer', r'adam|multi_tensor|foreach'),
    ('memcpy/cast', r'copy|memcpy|cast|convert|_to_copy'),
    ('triton', r'^triton_'),
    ('elementwise/reduce', r'elementwise|reduce|vectorized|unrolled|index|cat|silu|gelu|softmax|mul|add'),
]


def categorize(kernels):
    cats = {}
    for k in kernels:
        n = k['name'].lower()
        cat = next((c for c, rx in CATEGORIES if re.search(rx, n)), 'other')
        cats[cat] = cats.get(cat, 0.0) + k['total_ms']
    return {k: round(v, 1) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}


def cmd_profile(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)
    mode = 'nsys' if a.nsys else 'torch'

    def run(tree):
        res = time_tree(tree, out, 'profile', cand_env=cand_env, profile=mode, profile_iters=a.iters)
        ranks = res.pop('_ranks', None)
        if res['status'] != 'ok':
            return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)
        res['note'] = (f'{a.iters} iterations after a warmup of {CFG["timing"]["warmup"]} '
                       f'({a.iters // recipe.UPDATE_RATIO} generator iteration(s) per {recipe.UPDATE_RATIO}); '
                       f'instrumented: the time is not comparable with `time`')
        r0 = ranks[0]
        ends = r0['iteration_end_s']
        first = CFG['timing']['warmup']
        res['iteration_s_host'] = [round(ends[i] - ends[i - 1], 3) for i in range(first, min(len(ends), first + a.iters))]
        res['iteration_kinds'] = r0['kinds'][first:first + a.iters]
        if mode == 'torch':
            summ = json.loads((out / 'timing_profile.torchprof.json').read_text())
            summ['categories_ms'] = categorize(summ['kernels'])
            res['torch_profile'] = summ
            res['trace'] = str(out / 'timing_profile.trace.json')
            md = ['# profile', '', f'tree {a.tree}, config {a.config}', '', res['note'], '',
                  f'window {summ["span_s"]:.2f} s for {summ["iterations"]} iterations; GPU busy '
                  f'{summ["gpu_busy_s"]:.2f} s ({summ["gpu_busy_fraction"]:.0%}); kernel time {summ["kernel_time_s"]:.2f} s '
                  f'(rank 0)', '', 'host time per iteration (s): ' + ', '.join(map(str, res['iteration_s_host'])),
                  'kinds: ' + ', '.join(res['iteration_kinds']), '',
                  '## kernel time by category (ms, rank 0)', '', '| category | ms |', '|---|---|']
            md += [f'| {k} | {v} |' for k, v in summ['categories_ms'].items()]
            md += ['', '## GPU time inside model calls (forward only; ranges nest: guidance_model contains real_unet, '
                   'fake_unet, gan_head)', '', '| call | GPU ms | calls | CPU ms |', '|---|---|---|---|']
            md += [f'| {k} | {v["cuda_ms"]} | {v["calls"]} | {v["cpu_ms"]} |' for k, v in summ['annotations'].items()]
            md += ['', '## top kernels', '', '| ms | calls | share | kernel |', '|---|---|---|---|']
            md += [f'| {k["total_ms"]} | {k["calls"]} | {k["share_of_kernel_time"]} | `{k["name"][:140]}` |'
                   for k in summ['kernels'][:40]]
            md += ['', '## top CPU ops (self time)', '', '| ms | calls | op |', '|---|---|---|']
            md += [f'| {o["self_cpu_ms"]} | {o["calls"]} | {o["name"]} |' for o in summ['cpu_ops_self_time'][:25]]
            (out / 'profile.md').write_text('\n'.join(md) + '\n')
        else:
            res['trace'] = str(out / 'nsys_profile.nsys-rep')
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)
    return finish(out, 'profile.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_baseline(a):
    out = Path(a.out).resolve()

    def run(_):
        ref_src = check_reference()
        with scratch_dir('dmd-ref-') as tmp:
            tree = materialize(ref_src, Path(tmp) / 'reference')
            runs = {k: time_tree(tree, out, k) for k in ('ref_a', 'ref_b', 'ref_c')}
        ok = [v for v in runs.values() if v['status'] == 'ok']
        tot = [v['total_s'] for v in ok]
        res = {'status': 'ok' if len(ok) == len(runs) else 'ERROR', 'runs': {k: strip(v) for k, v in runs.items()},
               'total_s': tot, 'spread': round((max(tot) - min(tot)) / statistics.mean(tot), 4) if tot else None}
        if ok:
            r = ok[0]['_ranks']
            res['rng_fingerprints'] = {str(x['rank']): x['rng_fingerprints'] for x in r}
            res['rng_identical_across_runs'] = all(
                [x['rng_fingerprints'] for x in v['_ranks']] == [x['rng_fingerprints'] for x in r] for v in ok)
        return res
    return finish(out, 'baseline.json', guarded(out, run))


# ---------------------------------------------------------------- formulator commands

def cmd_record(a):
    """Formulator: record cases (default bench cases) of a tree into a directory (reference records, variants)."""
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(a.cases).read_text()) if a.cases else json.loads(path(CFG['numerics']['cases']).read_text())
    tree = Path(a.tree).resolve()
    if a.traj_seeds:
        spec = json.loads(path(CFG['trajectory']['spec']).read_text())
        cases = [traj_case(spec, int(s)) for s in a.traj_seeds.split(',')]
    failed = record_cases(tree, out, cases, out, candidate_environment(a.config), 'formulate', variant=a.variant,
                          losses_only=bool(a.traj_seeds))
    print(json.dumps({'status': 'ok' if not failed else failed, 'out': str(out), 'cost': cost()}))
    return 1 if failed else 0


def cmd_freeze_inputs(a):
    inputs = freeze_inputs(a.paths)
    f = ROOT / 'task' / 'formulation.json'
    data = json.loads(f.read_text()) if f.exists() else {}
    data['inputs'] = inputs
    f.write_text(json.dumps(data, indent=1) + '\n')
    print(json.dumps({'files': len(inputs), 'sha256': sum(v['kind'] == 'sha256' for v in inputs.values())}))
    return 0


def cmd_loader(a):
    got, err = loader_digest(Path(a.tree).resolve())
    print(json.dumps(got) if got else err)
    return 0 if got else 1


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
    f.add_argument('--nsys', action='store_true')
    f.add_argument('--iters', type=int, default=recipe.UPDATE_RATIO)
    b = sub.add_parser('baseline')
    b.add_argument('--out', required=True)
    r = sub.add_parser('record', help='formulator: record numerics cases or reference trajectories')
    r.add_argument('--tree', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--cases')
    r.add_argument('--variant')
    r.add_argument('--config')
    r.add_argument('--traj-seeds')
    ld = sub.add_parser('loader', help='formulator: print the dataset digest of a tree')
    ld.add_argument('--tree', required=True)
    fi = sub.add_parser('freeze-inputs', help='formulator: write the frozen input fingerprints')
    fi.add_argument('paths', nargs='+')
    a = p.parse_args()
    return {'freeze-inputs': cmd_freeze_inputs, 'check': cmd_check, 'time': cmd_time, 'profile': cmd_profile, 'baseline': cmd_baseline,
            'record': cmd_record, 'loader': cmd_loader}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
