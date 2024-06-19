# hippopt
### HIgh Performance* Planning and OPTimization framework

hippopt is an open-source framework for generating whole-body trajectories for legged robots, with a focus on direct transcription of optimal control problems solved with multiple-shooting methods. The framework takes as input the robot model and generates optimized trajectories that include both kinematic and dynamic quantities.

*supposedly

## Features

- [X] Direct transcription of optimal control problems with multiple-shooting methods
- [X] Support for floating-base robots, including humanoids ...
  - [ ] ... and quadrupeds
- [X] Integration with CasADi library for efficient numerical optimization
- [X] Generation of optimized trajectories that include both kinematic and dynamic quantities
- [ ] Extensive documentation
- [X] examples to help you get started

## Installation
It is suggested to use [``mamba``](https://github.com/conda-forge/miniforge).
```bash
conda install -c conda-forge -c robotology python=3.11 casadi pytest liecasadi adam-robotics idyntree meshcat-python ffmpeg-python matplotlib resolve-robotics-uri-py hdf5storage
pip install --no-deps -e .[all]
```

## Examples
### Turnkey planners
The folder [``turnkey_planners``](src/hippopt/turnkey_planners) contains examples of whole-body trajectory optimization for legged robots.
In this folder it is possible to find the following examples:
- [``humanoid_pose_finder/main.py``](src/hippopt/turnkey_planners/humanoid_pose_finder/main.py): generates a static pose for the humanoid robot ``ergoCub`` given desired foot and center of mass positions.
- [``humanoid_kinodynamic/main_single_step_flat_ground.py``](src/hippopt/turnkey_planners/humanoid_kinodynamic/main_single_step_flat_ground.py): generates a kinodynamic trajectory for the humanoid robot ``ergoCub`` to perform a single step motion with no a-priori guess or terminal constraint.
- [``humanoid_kinodynamic/main_periodic_step.py``](src/hippopt/turnkey_planners/humanoid_kinodynamic/main_periodic_step.py): generates a kinodynamic trajectory for the humanoid robot ``ergoCub`` to perform a periodic walking motion.
- [``humanoid_kinodynamic/main_walking_on_stairs.py``](src/hippopt/turnkey_planners/humanoid_kinodynamic/main_walking_on_stairs.py): generates a kinodynamic trajectory for the humanoid robot ``ergoCub`` to perform a walking motion on stairs.

> [!IMPORTANT]  
> For the tests to run, it is necessary to clone [``ergocub-software``](https://github.com/icub-tech-iit/ergocub-software) and extend the ``GAZEBO_MODEL_PATH`` environment variable to include the ``ergocub-software/urdf/ergoCub/robots`` and ``ergocub-software/urdf`` folders.

> [!NOTE]
> It is necessary to launch the examples from a folder with write permissions, as the examples will generate several files (ground meshes, output videos, ...).

## Citing this work

If you find the work useful, please consider citing:

```bib
@ARTICLE{dafarra2022dcc,
  author={Dafarra, Stefano and Romualdi, Giulio and Pucci, Daniele},
  journal={IEEE Transactions on Robotics}, 
  title={Dynamic Complementarity Conditions and Whole-Body Trajectory Optimization for Humanoid Robot Locomotion}, 
  year={2022},
  volume={38},
  number={6},
  pages={3414-3433},
  doi={10.1109/TRO.2022.3183785}}
```



## Maintainer

This repository is maintained by:

|                                                              |                                                      |
| :----------------------------------------------------------: | :--------------------------------------------------: |
| [<img src="https://github.com/S-Dafarra.png" width="40">](https://github.com/S-Dafarra) | [@S-Dafarra](https://github.com/S-Dafarra) |
