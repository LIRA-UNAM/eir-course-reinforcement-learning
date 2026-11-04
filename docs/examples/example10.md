# example10: the full K1 velocity task

**Goal:** train the Booster K1 policy as it is trained for the real robot: all 14 reward terms from [example9](example9.md), plus rough terrain, domain randomization, pushes and a terrain curriculum. The configuration is `Pumas-Velocity-Rough-Booster-K1-PPO` from the research repository this course is built from, with one change: instead of K1's speed curriculum (v_x widening from [−1, 1] to [−2, 3] m/s), commands come from fixed ranges for the whole run: v_x ∈ [−1, 1] m/s, v_y ∈ [−0.5, 0.5] m/s, ω_z ∈ [−1.2, 1.2] rad/s.

**Lecture:** sections 1 (why RL), 3 (non-stationary, randomized MDPs) and 8.7 (terminations and curricula).

## What changes from example9

The reward is identical. Everything else is what makes a policy survive the jump from simulation to the robot:

| Mechanism | example9 | example10 |
| --- | --- | --- |
| Terrain | Flat plane | Generated terrain mixing flat ground, pyramid stairs (up and down), slopes (up and down), random roughness and waves; 10 difficulty levels with a curriculum: level up after walking > 4 m, level down after covering < half the commanded distance |
| Terrain sensing | — | The critic sees a 17 × 11 height scan (187 values; critic total 271). The actor stays blind: it must infer the terrain from proprioception |
| Pushes | — | Random velocity kicks every 1–3 s |
| Friction | Fixed | Foot friction 0.3–1.2 |
| Sensors | Observation noise | + encoder bias ±0.015 rad |
| Mass properties | Nominal | Trunk CoM offset, pseudo-inertia of trunk and limbs randomized |
| Actuators | Nominal | PD gains scaled 0.8–1.2 |
| Ground contact | Nominal | Terrain `solref`/`solimp` randomized (contact stiffness and damping) |
| Commands | v_x ∈ [−1, 1] m/s, v_y ∈ [−0.5, 0.5] m/s, ω_z ∈ [−1.2, 1.2] rad/s | Same fixed ranges (no speed curriculum in the course) |
| Iterations | 10 000 | 10 000 (same budget) |

Each randomized parameter is sampled once per environment at start-up, so the 4096 environments are 4096 slightly different robots. The policy has to work on all of them; that is the "family of MDPs" from lecture section 3, point 6.

## Run it

Training needs a Linux machine with an NVIDIA GPU.

```bash
uv run train example10 --env.scene.num-envs 4096
uv run play example10                                     # pretrained policy from checkpoints/example10/
uv run play example10 --wandb-run-path <entity>/eir_course/<run_id>
```

In play mode the terrain is randomized at every reset and episodes do not time out. Use the Viser joystick to drive the robot over obstacles.

## What to watch for (hypotheses to check against your run)

- Learning is slower than example9 at first: harder terrain and randomization make the same reward harder to collect. `Curriculum/terrain_levels/mean` shows how far up the terrain ladder the robots get.
- Compare example9 and example10 policies in play on flat ground: example10 typically walks with a more conservative, robust gait.
