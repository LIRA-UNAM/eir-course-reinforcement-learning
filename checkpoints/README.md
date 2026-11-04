# Pretrained policies

`uv run play <example>` loads the policy in this folder when no `--wandb-run-path` or `--checkpoint-file` is given.

| Example | File | From |
| --- | --- | --- |
| example1, example2 | `<example>/q_table.npz` | `logs/qlearning/<example>/<run>/q_table.npz` |
| example3-example10 | `<example>/model.pt` | `logs/rsl_rl/<example>/<run>/model_9999.pt` |
