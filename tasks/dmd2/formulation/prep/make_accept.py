"""Acceptance trees: copies of the reference with one known change each (python make_accept.py <ref> <dir>)."""
import shutil, sys
from pathlib import Path

ref, base = Path(sys.argv[1]), Path(sys.argv[2])
EDITS = {
    # (b) setup change in __main__ -> gate 0 ERROR
    'setup_main': ('main/train_sd.py',
                   '    trainer = Trainer(args)\n',
                   '    torch.backends.cudnn.benchmark = True\n    trainer = Trainer(args)\n'),
    # (c) destructive: the GAN classifier never looks at the real images at all -- one fake_unet forward, one
    # classifier head call and one noising draw fewer per iteration -> the exact checks (call counters, RNG
    # fingerprints) must fail -> INCORRECT
    'skip_real_cls': ('main/sd_guidance.py',
                      '        pred_realism_on_real = self.compute_cls_logits(\n'
                      '            real_image.detach(), \n'
                      '            text_embedding=real_text_embedding,\n'
                      '            unet_added_conditions=real_unet_added_conditions\n'
                      '        )\n',
                      '        pred_realism_on_real = torch.zeros(\n'
                      '            real_image.shape[0], 1, device=real_image.device, dtype=torch.float32\n'
                      '        )\n'),
    # (d) math-equivalent rewrite of the DMD gradient -> not bitwise, statistical gate PASS
    'dmd_reorder': ('main/sd_guidance.py',
                    '            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True) \n',
                    '            scale = torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)\n'
                    '            grad = p_real / scale - p_fake / scale\n'),
    # (e) drop one loss term's gradient: GAN classification loss detached in the guidance update -> statistical gate INCORRECT
    'detach_guidance_cls': ('main/train_sd.py',
                            '            guidance_loss += guidance_loss_dict["guidance_cls_loss"] * self.guidance_cls_loss_weight\n',
                            '            guidance_loss += guidance_loss_dict["guidance_cls_loss"].detach() * self.guidance_cls_loss_weight\n'),
    # (d2) math-equivalent rewrite of the CFG combination, on the real UNet's bf16 outputs: equal in exact
    # arithmetic, different in floating point (and the difference survives into the fp32 DMD loss) -> not bitwise,
    # the statistical gate must call it equivalent
    'cfg_reorder': ('main/sd_guidance.py',
                    '        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) \n',
                    '        noise_pred = guidance_scale * noise_pred_text - (guidance_scale - 1.0) * noise_pred_uncond\n'),
    # (f) wrong training, nothing missing: the GAN classification loss in the guidance update is weighted 20%
    # higher. Every tensor is still there and the RNG stream is unchanged, so the exact checks hold and only the
    # statistical gate can reject it.
    'cls_weight_120pct': ('main/train_sd.py',
                          '        self.guidance_cls_loss_weight = args.guidance_cls_loss_weight \n',
                          '        self.guidance_cls_loss_weight = args.guidance_cls_loss_weight * 1.2\n'),
    # (e') alternative: GAN generator loss detached in the generator update
    'detach_gen_cls': ('main/train_sd.py',
                       '                generator_loss += generator_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight\n',
                       '                generator_loss += generator_loss_dict["gen_cls_loss"].detach() * self.gen_cls_loss_weight\n'),
}
for name, (rel, old, new) in EDITS.items():
    if len(sys.argv) > 3 and name not in sys.argv[3:]:
        continue
    dest = base / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ref, dest, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    f = dest / rel
    text = f.read_text()
    assert text.count(old) == 1, (name, text.count(old))
    f.write_text(text.replace(old, new))
    print('made', dest)
