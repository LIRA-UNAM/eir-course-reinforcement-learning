# Pretrained policies

One trained checkpoint per example, so you can watch every example without a W&B account:

```bash
uv run play exampleN --checkpoint models/exampleN.pt
```

`uv run play exampleN` with no `--checkpoint` or `--wandb-run-path` also loads `models/exampleN.pt`.

Each file is the last checkpoint of the latest finished training run on W&B:

| Example | File | W&B run | Source file |
| --- | --- | --- | --- |
| example1 | `example1.pt` (Q-table) | `Pumanoids/furuta_pendulum/8x3jp7l1` | `model_9999.pt` |
| example2 | `example2.pt` (Q-table) | `Pumanoids/furuta_pendulum/er237686` | `model_2999.pt` |
| example3 | `example3.pt` | `Pumanoids/eir_course/r8hv1j40` | `model_9999.pt` |
| example4 | `example4.pt` | `Pumanoids/eir_course/662p6zjr` | `model_9999.pt` |
| example5 | `example5.pt` | `Pumanoids/eir_course/21hfr4sx` | `model_9999.pt` |
| example6 | `example6.pt` | `Pumanoids/eir_course/ekj637ur` | `model_9999.pt` |
| example7 | `example7.pt` | `Pumanoids/eir_course/ezslvh8x` | `model_9999.pt` |
| example8 | `example8.pt` | `Pumanoids/eir_course/j2m1w35o` | `model_9999.pt` |
| example9 | `example9.pt` | `Pumanoids/eir_course/ocwbrcsu` | `model_9999.pt` |
| example10 | `example10.pt` | `Pumanoids/eir_course/lnys1zxk` | `model_9999.pt` |

To replace one with your own policy, copy `logs/rsl_rl/<example>/<run>/model_<iteration>.pt` here as `<example>.pt`.
