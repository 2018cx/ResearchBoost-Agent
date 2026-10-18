"""Per-rank load report. Observation only: it reads artefacts another run already wrote and
reports measured quantities, exactly like `bench.py profile` does. It runs no training, writes
nothing into task/, and is not part of any gate.

Two independent views of the same step:

  GPU   from the per-rank PyTorch profiler traces that `bench.py profile --detail` writes
        (<out>/detail_<label>.r<rank>.json, one per rank). Per rank it reports kernel time split
        into NCCL and non-NCCL kernels. Note on reading it: a rank that reaches a collective
        early stays inside the NCCL kernel until the last rank arrives, so a rank's NCCL kernel
        time grows as its own share of the compute shrinks.

  DATA  from the fixture, by replaying the sampler the run uses (modality_lengths ->
        get_modality_length_grouped_indices -> split_to_even_chunks, megabatch
        per_device * world_size * grad_accum, rank r taking micro-batches r, r+world, ...).
        Per rank it reports the tokens in its micro-batches and the token slots after the
        collator pads each micro-batch to its longest sample.

usage: python3 bench/rank_report.py [--traces <dir>] [--label reference] [--json out.json]
                                    [--no-gpu] [--no-data] [--seed 42]
"""
import argparse, collections, json, statistics, sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
BENCH = json.loads((WS / 'bench' / 'bench.json').read_text())
PER_DEV = BENCH['per_rank_batch']
WORLD = BENCH['world_size']
ACCUM = BENCH['grad_accum']
IMG_TOKENS = 576          # CLIP 336px / patch 14 -> 24*24 patches, replacing one placeholder token
MAXLEN = 2048             # --model_max_length


# ---------------------------------------------------------------- GPU view
def gpu_view(traces):
    rows = {}
    for t in sorted(traces):
        rank = int(str(t).rsplit('.r', 1)[1].split('.')[0])
        events = json.loads(Path(t).read_text()).get('traceEvents', [])
        nccl = other = 0
        first = last = None
        per_kernel = collections.Counter()
        for e in events:
            if e.get('cat') != 'kernel':
                continue
            dur, name = e.get('dur', 0), e.get('name', '')
            if 'nccl' in name.lower():
                nccl += dur
            else:
                other += dur
                per_kernel[name[:70]] += dur
            ts = e.get('ts')
            if ts is not None:
                first = ts if first is None else min(first, ts)
                last = ts + dur if last is None else max(last, ts + dur)
        rows[rank] = {'compute_ms': other / 1e3, 'nccl_ms': nccl / 1e3,
                      'span_ms': (last - first) / 1e3 if first is not None else None,
                      'top_kernel': per_kernel.most_common(1)[0][0] if per_kernel else None}
    return rows


# ---------------------------------------------------------------- DATA view
def split_to_even_chunks(indices, lengths, num_chunks):
    """Verbatim from the candidate's llava/train/llava_trainer.py."""
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]
    per = len(indices) // num_chunks
    chunks, cl = [[] for _ in range(num_chunks)], [0] * num_chunks
    for i in indices:
        k = cl.index(min(cl))
        chunks[k].append(i)
        cl[k] += lengths[i]
        if len(chunks[k]) == per:
            cl[k] = float('inf')
    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator):
    """Verbatim from the candidate's llava/train/llava_trainer.py, including the mixed tail:
    the last megabatch of each modality is concatenated, index-sorted and appended as one
    block that split_to_even_chunks never sees."""
    import torch
    def length_grouped(sub_lengths, gen):
        idx = torch.randperm(len(sub_lengths), generator=gen).tolist()
        mb = world_size * batch_size
        megabatches = [idx[i:i + mb] for i in range(0, len(sub_lengths), mb)]
        megabatches = [sorted(m, key=lambda i: sub_lengths[i], reverse=True) for m in megabatches]
        megabatches = [split_to_even_chunks(m, sub_lengths, world_size) for m in megabatches]
        return [i for m in megabatches for batch in m for i in batch]

    assert all(l != 0 for l in lengths)
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return length_grouped([abs(l) for l in lengths], generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])
    mm_shuffle = [mm_indices[i] for i in length_grouped(list(mm_lengths), None)]
    lang_shuffle = [lang_indices[i] for i in length_grouped(list(lang_lengths), None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i:i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i:i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]
    additional_batch = mm_megabatches[-1] + lang_megabatches[-1]
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatches = [megabatches[i] for i in torch.randperm(len(megabatches), generator=generator).tolist()]
    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))
    return [i for megabatch in megabatches for i in megabatch]


def token_lengths(data, tokenizer_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False, model_max_length=MAXLEN)
    sysprompt = len(tok('A chat between a curious user and an artificial intelligence assistant. '
                        'The assistant gives helpful, detailed, and polite answers to the '
                        "user's questions.").input_ids)
    out = []
    for s in data:
        n = sysprompt + sum(len(tok(c['value']).input_ids) - 1 + 5 for c in s['conversations'])
        if 'image' in s:
            n += IMG_TOKENS - 1
        out.append(min(n, MAXLEN))
    return out


def data_view(fixture, tokenizer_dir, seed):
    import torch
    data = json.loads(Path(fixture).read_text())
    real = token_lengths(data, tokenizer_dir)
    modality = [sum(len(c['value'].split()) for c in s['conversations']) * (1 if 'image' in s else -1)
                for s in data]
    # Upstream passes generator=None for the two per-modality shuffles, so those draw from the
    # global RNG that HF's set_seed(training_args.seed) seeds; only the megabatch shuffle takes the
    # Trainer's generator. Seed both here, or the replay is not reproducible.
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    order = get_modality_length_grouped_indices(modality, PER_DEV, WORLD * ACCUM, gen)
    micro = [order[i:i + PER_DEV] for i in range(0, len(order), PER_DEV)]

    rows = {r: {'real_tokens': 0, 'padded_tokens': 0, 'micro_batches': 0, 'longest': 0} for r in range(WORLD)}
    waste, per_step = [], []
    for s in range(len(micro) // (WORLD * ACCUM)):
        step = {r: 0 for r in range(WORLD)}
        for c in range(WORLD * ACCUM):
            mb = micro[s * WORLD * ACCUM + c]
            r = c % WORLD
            mx, sm = max(real[i] for i in mb), sum(real[i] for i in mb)
            rows[r]['real_tokens'] += sm
            rows[r]['padded_tokens'] += PER_DEV * mx
            rows[r]['micro_batches'] += 1
            rows[r]['longest'] = max(rows[r]['longest'], mx)
            step[r] += PER_DEV * mx
            waste.append(1 - sm / (PER_DEV * mx))
        mean = statistics.fmean(step.values())
        per_step.append({'step': s, 'max_over_mean': max(step.values()) / mean,
                         'heaviest_rank': max(step, key=step.get)})
    srt = sorted(real)
    stats = {'samples': len(data), 'micro_batches': len(micro), 'seed': seed,
             'token_len_p10': srt[len(srt) // 10], 'token_len_p50': srt[len(srt) // 2],
             'token_len_p90': srt[9 * len(srt) // 10], 'token_len_max': max(srt),
             'pad_share_per_micro_batch_mean': statistics.fmean(waste),
             'per_step_max_over_mean_mean': statistics.fmean(p['max_over_mean'] for p in per_step),
             'per_step_max_over_mean_min': min(p['max_over_mean'] for p in per_step),
             'per_step_max_over_mean_max': max(p['max_over_mean'] for p in per_step),
             'heaviest_rank_counts': dict(collections.Counter(p['heaviest_rank'] for p in per_step))}
    return rows, stats, per_step


def table(title, head, rows):
    w = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    out = [title, '  ' + '  '.join(str(h).ljust(w[i]) for i, h in enumerate(head))]
    out += ['  ' + '  '.join(str(c).ljust(w[i]) for i, c in enumerate(r)) for r in rows]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--traces', help='directory holding detail_<label>.r<rank>.json from `profile --detail`')
    p.add_argument('--label', default=None, help='restrict to one profile label')
    p.add_argument('--fixture', default=None, help='default: the timing fixture named in bench.json')
    p.add_argument('--seed', type=int, default=42, help='sampler seed to replay (training default)')
    p.add_argument('--json', help='also write the report as json')
    p.add_argument('--no-gpu', action='store_true')
    p.add_argument('--no-data', action='store_true')
    a = p.parse_args()

    report, lines = {}, []
    if not a.no_gpu:
        if not a.traces:
            sys.exit('--traces is required for the GPU view (or pass --no-gpu); '
                     'produce it with: bench.py profile --detail --out <dir>')
        pat = f'detail_{a.label}.r*.json*' if a.label else 'detail_*.r*.json*'
        traces = list(Path(a.traces).glob(pat))
        if not traces:
            sys.exit(f'no traces matching {pat} under {a.traces}')
        g = gpu_view(traces)
        report['gpu'] = g
        tot = {k: sum(v[k] for v in g.values()) for k in ('compute_ms', 'nccl_ms')}
        rows = []
        for r in sorted(g):
            v = g[r]
            mean_c = tot['compute_ms'] / len(g)
            rows.append([r, f"{v['compute_ms']:.0f}", f"{v['compute_ms'] / mean_c:.4f}",
                         f"{v['nccl_ms']:.0f}", f"{v['span_ms']:.0f}" if v['span_ms'] else '-'])
        lines += table('GPU kernel time per rank (profiler traces)',
                       ['rank', 'compute ms', 'vs mean', 'nccl ms', 'span ms'], rows) + ['']

    if not a.no_data:
        fixture = a.fixture or BENCH['assets']['fixture'] + '/timing.json'
        d, stats, per_step = data_view(fixture, BENCH['assets']['vicuna'], a.seed)
        report['data'] = {'per_rank': d, 'stats': stats, 'per_step': per_step}
        mean_p = statistics.fmean(v['padded_tokens'] for v in d.values())
        mean_r = statistics.fmean(v['real_tokens'] for v in d.values())
        rows = []
        for r in sorted(d):
            v = d[r]
            rows.append([r, f"{v['real_tokens'] / 1e6:.3f}M", f"{v['real_tokens'] / mean_r:.4f}",
                         f"{v['padded_tokens'] / 1e6:.3f}M", f"{v['padded_tokens'] / mean_p:.4f}",
                         v['longest'], v['micro_batches']])
        lines += table(f"Tokens per rank over the fixture (sampler replay, seed {a.seed})",
                       ['rank', 'real', 'vs mean', 'padded', 'vs mean', 'longest mb', 'micro-batches'], rows)
        lines += ['', f"  token length over the {stats['samples']} samples: "
                      f"p10 {stats['token_len_p10']}  p50 {stats['token_len_p50']}  "
                      f"p90 {stats['token_len_p90']}  max {stats['token_len_max']}",
                  f"  padded token slots that hold padding, per micro-batch: "
                  f"{stats['pad_share_per_micro_batch_mean']:.1%} mean",
                  f"  per optimizer step, heaviest rank's padded tokens over the 4-rank mean: "
                  f"{stats['per_step_max_over_mean_mean']:.4f} mean  "
                  f"{stats['per_step_max_over_mean_min']:.4f} min  "
                  f"{stats['per_step_max_over_mean_max']:.4f} max",
                  f"  which rank was heaviest, over the {len(per_step)} steps: "
                  f"{stats['heaviest_rank_counts']}",
                  "  (the sampler sorts and balances on word counts; the padded cost above is in "
                  "tokens, and the two orderings differ)", '']

    print('\n'.join(lines))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1) + '\n')
        print(f'wrote {a.json}')


if __name__ == '__main__':
    main()
