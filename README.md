# Biped Locomotion

Bipedal locomotion reinforcement learning using Isaac Lab with a **manager-based workflow** and RSL-RL. 
sim2sim deployment in Mujoco.

## Architecture

This project uses the **manager-based workflow** for Isaac Lab. The manager-based design organizes the simulation into modular managers (e.g. `ObservationManager`, `ActionManager`, `CommandManager`, `EventManager`) that handle observations, actions, commands, and domain randomization, providing a structured and extensible environment setup.

## Available Environments

| Task | Description |
|------|-------------|
| `biped_walk_flat` | Flat terrain locomotion (training) |
| `biped_walk_flat_play` | Flat terrain locomotion (evaluation) |
| `biped_walk_rough` | Rough terrain locomotion (training) |
| `biped_walk_rough_play` | Rough terrain locomotion (evaluation) |
| `biped_walk_stairs` | Stair climbing (training) |
| `biped_walk_stairs_play` | Stair climbing (evaluation) |

All environments use `ManagerBasedRLEnv` and the `PointFootPPORunnerCfg` agent.

## Scripts

### Training (RSL-RL)

Train a policy with PPO:

```bash
python scripts/rsl_rl/train.py --task=biped_walk_flat
```

**Useful options:**
- `--task` — Task name (e.g. `biped_walk_flat`, `biped_walk_rough`, `biped_walk_stairs`)
- `--num_envs` — Number of parallel environments
- `--max_iterations` — Training iterations
- `--resume` — Resume from latest checkpoint
- `--load_run RUN_NAME` — Resume from a specific run
- `--experiment_name` — Custom experiment folder name

### Play

Run a trained policy (fixed/random commands):

```bash
python scripts/rsl_rl/play.py --task=biped_walk_flat_play --load_run RUN_NAME
```
```bash
python scripts/list_envs.py
```

Or with a specific checkpoint file:

```bash
python scripts/rsl_rl/play.py --task=biped_walk_flat_play --checkpoint path/to/model.pt
```

#### Play with Keyboard Teleport

Control the robot with W/A/S/D during evaluation:

```bash
python scripts/rsl_rl/keyboard_teleport.py --task=biped_walk_flat_play --load_run RUN_NAME
```

#### Play with Gamepad Teleport

Control the robot with a gamepad during evaluation:

```bash
python scripts/rsl_rl/gamepad_teleport.py --task=biped_walk_flat_play --load_run RUN_NAME
```

### Debug Policy

Inspect policy inputs and outputs (observation terms, actions):

```bash
python scripts/rsl_rl/debug_policy.py --task=biped_walk_flat_play --checkpoint path/to/model.pt
```

To also view live observation and action data:

```bash
python scripts/rsl_rl/debug_policy.py --task=biped_walk_flat_play --checkpoint path/to/model.pt --obs_data_view
```

****### Sim2Sim Deployment (MuJoCo)

Deploy trained policies in MuJoCo for sim-to-sim validation. Update `relative_policy_path` in each script to point to your exported policy.

#### Basic Deployment

Run the policy with fixed velocity commands (edit `cmd_vel` and `relative_policy_path` in the script):

```bash
python sim2sim_mujoco/mujoco_basic_deploy.py
```

#### Keyboard Teleport

Control the robot with keyboard (arrow keys, etc.) during MuJoCo simulation:

```bash
python sim2sim_mujoco/mujoco_keyb_teleport.py
```

---

**Note:** Run scripts from the project root with Isaac Lab properly installed and the `biped` extension on your `PYTHONPATH`. If using the Isaac Lab launcher, use `./isaaclab.sh -p scripts/rsl_rl/train.py ...` from your Isaac Lab installation directory.
