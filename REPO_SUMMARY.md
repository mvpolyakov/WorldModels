# GRI Repository Handoff

## Purpose

GRI is an early, configuration-driven Python prototype built on Genesis. It provides an interactive scene editor and a command-line evaluation loop for simulated worlds.

The long-term goal is an object-centric world model that infers latent physical properties from observations and interventions, predicts outcomes in unseen scenarios, and eventually supports goal-directed action selection and generation. The repo currently contains simulation code, a basic evaluation script, and a dummy world model; training code is the next implementation step. The roadmap below describes planned work.

## Research goals

1. **Infer and predict object physics.** Learn representations of motion and latent properties such as mass, friction, and restitution. Use interactions to infer properties (for example, mass), then predict their effects in other interactions. Longer-term predictions include object presence and motion under occlusion, and changes of state such as ice melting. Success means accurate predictions in unseen scenarios, including unseen combinations of properties.
2. **Predict counterfactual outcomes.** Given the same state and different supplied actions, imagine the resulting trajectories. Begin with simple actions that do not require a trained action generator.
3. **Generate purposeful actions.** For a target-reaching task such as hitting a ping-pong ball at a target, generate candidate paddle angles and velocities, evaluate their outcomes with the world model, and execute the selected action in simulation. First establish action selection over a grid, then learn to generate candidates.

## Planned architecture and training approach

The initial learned pipeline is **object encoder → dynamics model → decoder**. The encoder forms object-centric latent states from observations; the dynamics model predicts `s_t → s_{t+1}`; the decoder extracts predicted observable quantities and object properties, such as velocity.

To validate the dynamics design, initially bypass the encoder and provide ground-truth simulator observations. Phase 1.0 uses only five consecutive XYZ positions to predict the next position with a small MLP; overfitting is acceptable while establishing the Genesis collection, training, and evaluation loop. Phase 1.1 will investigate persistent object state and inference of physical properties. Once dynamics works, replace simulator observations with a trained encoder. This bootstrap stage can be skipped in later runs after the design is validated.

See [TRAINING_PLAN.md](TRAINING_PLAN.md) for the detailed phase 1.0 implementation plan and phase 1.1 follow-up.

The rolling, bouncing, paddle-contact, and multi-object stages should share the same architecture, potentially with richer object states or greater capacity. Counterfactual action experiments then add explicit action conditioning to the dynamics model, `(s_t, a_t) → s_{t+1}`. Learned action generation is a separate, later module.

## Phased roadmap

| Stage | Environment / task | Main learning or evaluation objective |
| --- | --- | --- |
| 1 | Ball rolling | Velocity and friction. Start dynamics training with ground-truth inputs. |
| 2 | Ball bouncing | Velocity, restitution, and surface properties. |
| 3 | Paddle hitting | Contact dynamics, spin, and effects of paddle angle and impact velocity. |
| 3B | Multiple balls colliding | Track multiple objects and model their interactions. |
| Transfer check | Real videos, before adding action conditioning | Evaluate whether learned representations capture real-world physical properties such as friction. |
| 4 | Counterfactual hitting | From the same state, predict distinct ball trajectories for several supplied actions. |
| 4A | Planning over an action grid | Use predicted outcomes to choose the paddle action that reaches target `T`; verify in simulation. |
| 5 | Learned action generation | Train an action model to propose paddle angle/velocity candidates, score them with world-model rollouts, and execute the best candidate. |

Evaluation should distinguish prediction on held-out scenarios and property combinations, counterfactual rollout accuracy, and target-reaching success in simulation. Occlusion and material state changes remain longer-term goals beyond the initial ball-interaction curriculum.

## Architecture

- `src/sim/scene_config.py` defines the generic configuration dataclasses and JSON loading.
- `src/sim/build_world.py` defines the `WorldBuilder` interface and its generic `scaffold_world()` method. `SimpleRollingWorldBuilder` owns `simple_rolling.json`, rolling-world randomization, and prediction visualization setup.
- `src/sim/dataclasses.py` contains the `World` wrapper and viewer controls.
- `src/eval.py` is the rollout/evaluation entry point.
- `src/editor.py` is the interactive scene editor and config saver.
- `src/world_models/` contains model implementations. The model interface currently accepts and returns position arrays shaped `(B, K, 3)`: environments, mobiles, XYZ coordinates.
- `src/sim/utilities/` contains simulation support and visualization helpers. Visualization logic should remain separate from world construction and physics state.

## Development conventions

- Keep scene data and physical/visual properties in the config model or JSON files.
- Use semantic names and domain-focused encapsulation.
- Represent scene components uniformly in their appropriate config lists.
- Make only task-related edits; avoid cosmetic reformatting.
- Genesis APIs and typings are version-sensitive. Verify installed source or signatures before changing Genesis calls.

## Validation and execution

The intended interpreter is the pyenv environment `ai`:

```bash
PYENV_VERSION=ai pyenv exec python src/editor.py
PYENV_VERSION=ai pyenv exec python src/eval.py --world-builder simple-rolling
PYENV_VERSION=ai pyenv exec python -m py_compile <changed-files>
PYENV_VERSION=ai pyenv exec python -m json.tool src/sim/configs/simple_rolling.json
```

There is no automated test suite. Genesis execution requires compatible GPU/display support; the installed Genesis package may also trigger a Numba cache-location error in restricted environments.
