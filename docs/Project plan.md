# Learning an Object-Centric Predictive World Model for Physical Interaction

## Motivation

Many robotic systems in development today are focused on perfecting action control in a (largely) static environment.  Clothes folding is a task often taken as representative of required capabilities for a general-purpose consumer home robot -- trivial for humans but challenging to learn for modern robots.  However, a true general purpose autonomous robot must not only be competent at specific tasks but also be able to adequately respond to its environment. 

This project approaches general purpose robotics from the bottom up, starting with learning basic world modeling to eventually allow for arbitrary interaction and graceful degradation.  Among other things, this goal requires an internal model of basic physics.  video-based World Foundation Models such as Nvidia Cosmos, are still [quite weak at physics intuition](https://arxiv.org/abs/2506.09849).  They can predict plausible but not necessarily accurate interactions, and prediction errors quickly compound during rollout.

A latent state Object World Model, on the other hand, that first, explicitly decouples world state prediction from action generation and, second, organizes that world state around a smaller number of objects with composable, general properties could be far more effective. Accordingly, this project intends to progress from predicting simple motion of individual mobiles to using an object world model to select and carry out actions by a simple robot that interacts with the world (in sim). The advantages of this approach are that 
- such a capability is useful to any robotic system in an open environment.  
- it can be trained to high fidelity in simulation (access to training data is not a concern)
- it allows for progressive development all the way to a useful real-world system, with each stage motivated by a specific ability gain and backed by existing research.

This effort is philosophically inspired by [Dreamer family models](https://arxiv.org/abs/2301.04104) that jointly train a world model and an action generation model.  See [Related Research](#related-research) for additional references.


## Long-term research goals

The long-term objective for this project is to develop: 
- A world model that learns various physical properties (ultimately object presence and motion under occlusion, cross-interaction inference of properties such as mass, and material state changes such as melting. The plan below, however, is limited to motion and related properties, e.g., friction, mass and restitution). Success means accurate predictions in unseen scenarios, including with interactions and with unseen combinations of properties.

- A robot action generation model that uses the world model to both generate actions and evaluate alternative generated courses of action based on their projected outcomes.  Success means generating and carrying out such actions; for example, for a task like hitting a ping-pong ball at a target, generate candidate paddle angles and velocities, evaluate their outcomes with the world model, and execute the selected action in simulation.

## Phased plan

These are the phased research objectives, green indicating phase completion (with phase code changes detailed in branches.md)


<table>
  <thead>
    <tr>
      <th style="width: 3em; text-align: center;">Stage</th>
      <th>Phase</th>
      <th>Goals / hypotheses tested</th>
      <th>Enabling architecture / inputs</th>
      <th>Scenes / tasks</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #dcfce7;">
      <td rowspan="5" style="background-color: #dbeafe; color: #111827; text-align: center; vertical-align: middle; white-space: nowrap; width: 3em;"><strong>1a</strong></td>
      <td><strong>1.0 · Dynamics bootstrap</strong></td>
      <td>Establish the simulation → training → prediction loop; build visual evaluation tool. Can a short position history predict the next position?</td>
      <td>Small MLP: five consecutive XYZ positions → next XYZ position. Ground-truth simulator observations bypass the encoder.</td>
      <td>Ball rolling on a fixed ramp</td>
    </tr>
    <tr style="background-color: #dcfce7;">
      <td><strong>1.1 · Object state, dynamics learning</strong></td>
      <td>Inference of latent physical properties (gravity/acceleration)</td>
      <td>Recurrent object state with a simple RNN. Basic encoder (passthrough) + updater + dynamics + decoder architecture</td>
      <td>Rolling trajectories on multiple varied ramps</td>
    </tr>
    <tr>
      <td><strong>1.2 · Learning multiple latent properties</strong></td>
      <td>Vary friction: can the model learn this additional latent property from the warmup window?</td>
      <td>?</td>
      <td>Same as 1.1 + varying friction property of the ball</td>
    </tr>
    <tr>
      <td><strong>2 · Extrapolation<br>(optional)</strong></td>
      <td>Extrapolation to friction/acceleration/motion values far outside its training range</td>
      <td>If above architecture does not, may explore neural arithmetic research OR add a GLM module</td>
      <td>Same as 1.2 with latent properties outside training range</td>
    </tr>
    <tr>
      <td><strong>3 · Object properties' long term persistence</strong></td>
      <td>Learn persistent latents (restitution)</td>
      <td>RNN+, may require object centric pipeline (currently phase 4.2)</td>
      <td>Ball bouncing with varying restitution values (between scenes) --&gt; model must learn bounciness of the ball from one event to predict future bounces much later</td>
    </tr>
    <tr>
      <td rowspan="3" style="background-color: #bfdbfe; color: #111827; text-align: center; vertical-align: middle; white-space: nowrap; width: 3em;"><strong>1b</strong></td>
      <td><strong>4.0 · More complex dynamics: Ball bouncing</strong></td>
      <td>Predict motion at surface switching, and on impact</td>
      <td>Upgrade dynamics model to simple transformer</td>
      <td>Ball bouncing on surfaces with varied angles</td>
    </tr>
    <tr>
      <td><strong>4.1 · Paddle hitting</strong></td>
      <td>Learn contact dynamics and the effects of paddle angle, impact velocity (mobile vs. static surface -&gt; mobile vs. moving surface)</td>
      <td>Same as 4.0 ?</td>
      <td>Paddle–ball impacts under varied contact conditions.</td>
    </tr>
    <tr>
      <td><strong>4.2 · Multi-object interaction</strong></td>
      <td>Track several objects and predict their coupled motion through collisions?</td>
      <td>Shared object-centric pipeline with state for multiple objects and interaction between them</td>
      <td>Multiple balls colliding.</td>
    </tr>
    <tr>
      <td rowspan="2" style="background-color: #fef3c7; color: #111827; text-align: center; vertical-align: middle; white-space: nowrap; width: 3em;"><strong>2a</strong></td>
      <td><strong>5.0 · Counterfactual action/interaction</strong></td>
      <td>Given the same state, can the model predict different outcomes under different supplied interventions?</td>
      <td>Add action-conditioned dynamics: (object state, action) → next state. Supply simple actions without a trained action generator.</td>
      <td>Repeat a ball–paddle starting state with different paddle angles and velocities.</td>
    </tr>
    <tr>
      <td><strong>5.1 · Planning over a grid</strong></td>
      <td>Can accurate imagined outcomes support purposeful action selection?</td>
      <td>Action-conditioned world model (5.0) + grid search and target-based scoring of rollouts.</td>
      <td>Select a paddle action that sends the ball toward target T; execute it in simulation.</td>
    </tr>
    <tr>
      <td rowspan="1" style="background-color: #fde68a; color: #111827; text-align: center; vertical-align: middle; white-space: nowrap; width: 3em;"><strong>2b</strong></td>
      <td><strong>6.0 · Learned action generation</strong></td>
      <td>Train model to generate actions based on simple goals using inverse dynamics in simulation (5.0); goal input mode tbd</td>
      <td>Add an action-generating model proposing paddle angle/velocity candidates, using action vocabulary introduced in 5.0</td>
      <td>Same as 5.1</td>
    </tr>
    <tr>
      <td rowspan="2" style="background-color: #f3e8ff; color: #111827; text-align: center; vertical-align: middle; white-space: nowrap; width: 3em;"><strong>2c</strong></td>
      <td><strong>6.1 · Action sequence learning (tentative)</strong></td>
      <td>Use RL to train generation of progressively longer sequences of actions</td>
      <td>Largely same as 6.0</td>
      <td>? Possibly something like a ping-pong set up, requiring acquisition of multi-step action sequences</td>
    </tr>
    <tr>
      <td><strong>6.2 · Skill discovery / refinement (tentative)</strong></td>
      <td>Can multimodal action generation discover distinct, reusable strategies, potentially including different spins?</td>
      <td>Extend learned action generation to multiple strategies; the skill representation and training method remain open.</td>
      <td>Ball–paddle tasks with multiple successful strategies.</td>
    </tr>
    <tr>
      <td rowspan="2" style="background-color: #fee2e2; color: #111827; text-align: center; vertical-align: middle; white-space: nowrap; width: 3em;"><strong>3</strong></td>
      <td><strong>7.0 · Video input</strong></td>
      <td>Switch to video frame input (sim cameras) from ground-truth simulator observations. Replicate phases 1-6 with video input</td>
      <td>Add video encoder; probably something like SAVI++ incorporating object tracking</td>
      <td>Same scenes as above - with input solely provided from camera(s and possibly depth)</td>
    </tr>
    <tr>
      <td><strong>7.1 · Transfer check: Real videos</strong></td>
      <td>Do learned representations capture real-world motion and properties such as friction?</td>
      <td>Same as 7.0, encoder may need fine-tuning</td>
      <td>Real videos of physical interactions analogous to above -- ping pong balls, billard balls -- modulo not modeling more complex effects like spin</td>
    </tr>
  </tbody>
</table>




## Major decisions
1. Simulation environment.  
Chosen to start: Genesis (having also considered maniskill, mujoco, Isaac) for its
    - GPU-native support for both rendering and processing in windows/WSL
    - enormous parallelism 
    - good Python support
    - camera sensors 
    - URDF/MJCF for asset import 
    - built-in GUI (though limited)
    - increasingly good RL abstractions (not as good as maniskill but enough for what I want)
    - reasonable learning curve 


## Related Research
In putting together this plan, the following papers were among many serving as guideposts.

- [Dreamer V3](https://arxiv.org/pdf/2301.04104) - Demonstrates a world state model with action generation
- [Dreamweaver](https://arxiv.org/abs/2501.14174) - Demonstrates an object world model with composable properties
- [PhysDreamer](https://arxiv.org/abs/2404.13026) - Demonstrates learning action-conditioned dynamics in 3d space with 3d objects
- [SlotFormer](https://arxiv.org/abs/2210.05861) - The canonical transformer-based object world model
- [SAVI++](https://arxiv.org/abs/2206.07764) - Demonstrates slot-based representations over video
- [Pi0.5](https://arxiv.org/abs/2504.16054) - A premier VLA with hierarchical action generation
