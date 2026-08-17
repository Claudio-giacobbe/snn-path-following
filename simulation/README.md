# CoppeliaSim Simulation

## Scene

Open:

`path_planning_robot1.ttt`

The scene contains the BM_Bot mobile robot, its four rolling joints, the camera, the Floor and the child script responsible for procedural trail generation.

Expected object names:

```text
/camera
/rollingJoint_fl
/rollingJoint_rl
/rollingJoint_rr
/rollingJoint_fr
```

## Procedural trail

`path_generator.lua` is the non-threaded child script associated with the Floor.

It hides the original path and generates a closed red trail starting from the robot position. The path is procedurally deformed and represented by red static segments.

The script is intended for CoppeliaSim 4.10.
