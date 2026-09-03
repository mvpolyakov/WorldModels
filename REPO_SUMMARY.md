# GRI Repository Handoff Summary

## Repository state

- Working directory: `/home/mvp/git/gri`
- Git has no commits yet; the project files currently appear as untracked files.
- No test files or test configuration are present.
- Python dependencies are listed in `requirements.txt`, including `genesis-world[render]==1.3.3`, PyTorch, NumPy, SciPy (implicitly used by the code), and RL-related packages.

## Project purpose

This is an early Genesis-based scene/simulation prototype. It currently builds a configurable ramp-and-ball scene, runs it in an evaluation loop, and provides a keyboard-driven editor for interactively adjusting object poses and the viewer camera.

## Main files and flow

### `src/sim/scene_config.py`

- Defines `ObjectConfig`, `CameraConfig`, and `SceneConfig` dataclasses.
- `SceneConfig.from_json()` loads `landscape`, `mobiles`, camera settings, gravity, and `show_viewer` from JSON.
- `SceneConfig.to_json()` serializes dataclasses with `dataclasses.asdict()`.
- `load_scene_config(key)` first treats an existing `.json` path as a file, then checks `src/sim/configs/<key>.json`, then falls back to the in-code `_base_config()` generator.
- The generated fallback scene computes a 25-degree ramp and a ball placement mathematically. The checked-in JSON is a separate, edited 15-degree scene.

### `src/sim/configs/base.json`

- Current file-backed scene configuration.
- Contains one fixed `Box` ramp and one mobile `Sphere` ball.
- The ramp is positioned around `(0.1, 0.2, 0.4476)` with Euler rotation near `(0, -15, 0)` degrees.
- The ball is positioned around `(-0.2563, 0.05, 0.5802)` with radius `0.2`.
- Viewer camera values have been saved from an editor session; observation camera values remain defaults.
- Friction values are `0.5` for the ramp and `2.9` for the ball.

### `src/sim/build_world.py`

- Maps config strings `Box`, `Sphere`, `Mesh`, `URDF`, and `MJCF` to Genesis morph classes.
- Adds all landscape objects—including the second, ground `Box` component—then mobile objects. The ground is an ordinary landscape entry so its contact friction is configurable.
- Passes each object's `morph_params` directly as keyword arguments to the selected Genesis morph constructor.
- Applies sliding, torsional, and rolling friction from each object config. Rolling friction is enabled in the rigid solver so rolling bodies can lose energy and come to rest.
- Supports optional per-object diffuse textures.
- Adds a non-colliding Genesis checkerboard plane slightly above the configured ground box for visual context; the box remains the physics surface.
- Adds an observation camera, builds the scene with `n_envs` and `env_spacing=(1.5, 1.5)`, and returns a `World` dataclass-like bundle.
- Mobile entities are returned in a name-to-entity dictionary. Landscape entities are not stored in `World`; the editor relies on their creation order in `scene.entities`.

### `src/sim/dataclasses.py`

- Contains `World`, which wraps a Genesis scene, mobile entities, and an optional agent.
- `World.step()` steps the scene and optional agent.
- `World.reset()` resets the scene; randomization is currently a placeholder.
- `add_scene_controls()` registers `T` for reset and Shift+`T` for randomized reset.

### `src/eval.py`

- CLI options: `--n-envs` (default `1`), `--steps` (default `1000`), and `--scene-config` (default `base`).
- Initializes Genesis with the GPU backend, loads the scene config, builds the world, registers reset controls, performs initial steps, then keeps the viewer alive in an infinite stepping loop.
- Model loading and rollout collection are not implemented yet.
- The evaluation loop can pre-create translucent, non-colliding ball ghosts, place one every `--ghost-interval` steps, and save an image using the live viewer camera pose with `--output-image` (default `artifacts/ball_path.png`).
- `TrajectoryRecorder` owns the ghost entities, capture state, render camera, and image export; `World` only exposes it as an optional `trajectory_recorder` member.

### `src/editor.py`

- CLI option: `--config`, defaulting to `src/sim/configs/base.json`.
- Initializes Genesis with the GPU backend and loads the specified config.
- Ensures objects without an existing `fixed` morph parameter are fixed while editing, so gravity does not move them.
- Selects landscape objects by their creation order, with the ramp first and ground second; mobiles come from `world.mobiles`.
- `TAB` cycles selection; `I/K`, `J/L`, and `U/O` translate on X/Y/Z; number keys `1/3`, `4/6`, and `7/9` rotate around X/Y/Z; `R` resets the selected object to its initial pose; `ENTER` saves.
- Saving writes current positions and either Euler angles or quaternions back into each object's `morph_params`, removes temporary `fixed` only from mobile objects, saves viewer camera position/look-at, and calls `SceneConfig.to_json()`.

## Current scene-configuration concerns for the next task

1. `base.json` is now authoritative whenever it exists, so edits to `_base_config()` are ignored by the default loader.
2. The second `landscape` component is an explicit fixed ground `Box`, represented like the ramp, so its sliding and rolling friction can be tuned independently in `base.json`.
3. `friction` controls sliding resistance. For a rolling sphere, tune `friction_rolling` (in metres) because rolling resistance is separate from sliding friction.
4. `build_world()` forwards JSON morph parameters unchanged. Any Genesis API mismatch or desired normalization must be handled at this boundary or during config loading.
5. The editor's landscape lookup depends on Genesis entity ordering and may be brittle if additional entities are inserted before or among landscape objects.
6. The editor's comments/documentation are partly stale: the implementation uses `R` to restore the original pose, number keys for rotation, and does not implement Shift-based rotation.
7. Pose serialization chooses Euler angles when `euler` exists or `quat` is absent, otherwise it writes quaternions. This preserves the existing representation in most cases but needs validation for each Genesis morph type.
8. `src/sim/scene_config.py` imports `Tuple` and uses JSON arrays interchangeably with tuple annotations; this works through constructor assignment but `to_json()` emits arrays.

## Useful commands

```bash
python src/editor.py
python src/eval.py --scene-config base
python src/eval.py --scene-config src/sim/configs/base.json
```

Genesis/GPU availability and the installed dependency versions should be verified before attempting to run either executable. There is currently no automated test suite.

## Relevant prior notes

The `scratch/` directory contains the earlier editor plan and walkthrough. `scratch/EDITOR_CONTROLS.md` describes the intended keyboard workflow, while `scratch/implementation_plan.md` records the original proposal to migrate from the mathematical fallback scene to JSON.
