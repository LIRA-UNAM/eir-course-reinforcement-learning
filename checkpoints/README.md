# Pretrained policies

`uv run play <example>` loads the policy in this folder when no `--wandb-run-path` or `--checkpoint-file` is given.

| Example | File | From |
| --- | --- | --- |
| example1 | `example1/model.pt` (Q-table) | `logs/rsl_rl/example1/<run>/model_299.pt` |
| example2 | `example2/model.pt` (Q-table) | `logs/rsl_rl/example2/<run>/model_2999.pt` |
| example3-example10 | `<example>/model.pt` | `logs/rsl_rl/<example>/<run>/model_9999.pt` |
