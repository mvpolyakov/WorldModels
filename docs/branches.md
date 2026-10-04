# Historical branches descriptions

## frozen/phase-0
First functioning branch -- model learns to copy a single fixed trajectory.
Contains train/eval/editor scripts. Can create a single fixed trajectory, have the model
learn it, then demonstrate it with eval script (2 trajectory traces shown side by side)

## frozen/phase-1 
Learning latent physics (friction, inertia), also adds scalable training infrastructure.

Model is the same as phase-0 but training data contains different friction values
and directions.  Also adds multi-frame history capture to predict motion - which 
also requires a warm up period before predictions can be generated.

Adds parallelized training using genesis's multiple envs support. Also refactors
scene-world relation, creating an extensible inheritance scheme and adds a new construct "builder" in addition to Scene and World.  

