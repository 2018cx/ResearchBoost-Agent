"""Gate 0 data check (CPU): the tree's data pipeline must train every optimizer step on the reference's samples.

  python bench/loader_probe.py --tree <dir> --batches K        (prints one JSON line)

Uses the tree's own argument dataclasses, conversation templates, LazySupervisedDataset, collator, LLaVATrainer
sampler and Trainer.get_train_dataloader on the timing fixture, with the bench's training arguments. It re-creates
the part of train() that sets up the tokenizer and the image processor (train() itself is held equal to the
reference by the setup guard). One process, world size 1 and gradient_accumulation_steps x world_size, so that the
modality/length-grouped sampler sees the same megabatch size (world x accumulation x per-device batch) as the real
run; global micro-batch k goes to rank k % world in step k // (world x accumulation). Seeded like the Trainer.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tree', required=True)
    p.add_argument('--batches', type=int, default=16)
    a = p.parse_args()
    tree = Path(a.tree).resolve()
    sys.path.insert(0, str(tree))
    os.chdir(tree)
    import torch
    import transformers
    from bench.llava_adapter import CFG, batch_digest, train_argv
    from llava import conversation as conversation_lib
    from llava.train import train as T
    from llava.train.llava_trainer import LLaVATrainer

    world, accum = CFG['world_size'], CFG['grad_accum']
    out = tempfile.mkdtemp(prefix='llava-probe-')
    argv = train_argv({'--deepspeed': None, '--bf16': 'False', '--tf32': 'False', '--gradient_checkpointing': 'False',
                       '--gradient_accumulation_steps': str(world * accum), '--dataloader_num_workers': '0',
                       '--use_cpu': 'True'}, out)
    argv = [x for i in range(0, len(argv), 2) for x in argv[i:i + 2] if argv[i + 1] != 'None']
    parser = transformers.HfArgumentParser((T.ModelArguments, T.DataArguments, T.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(argv)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=training_args.model_max_length, padding_side='right',
        use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    data_args.image_processor = transformers.CLIPImageProcessor.from_pretrained(model_args.vision_tower)
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    module = T.make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LLaVATrainer(model=torch.nn.Linear(1, 1), tokenizer=tokenizer, args=training_args, **module)
    transformers.set_seed(training_args.seed)
    loader = trainer.get_train_dataloader()
    rows = []
    for k, batch in enumerate(loader):
        if k >= a.batches:
            break
        d = batch_digest(batch, rows=True)
        d.update(step=k // (world * accum), micro=(k // world) % accum, rank=k % world)
        rows.append(d)
    print(json.dumps({'batches': rows, 'dataset_len': len(module['train_dataset'])}))


if __name__ == '__main__':
    main()
