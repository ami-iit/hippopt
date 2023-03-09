# hippopt
### HIgh Performance* Planning and OPTimization framework

hippopt is an open-source framework for generating whole-body trajectories for legged robots, with a focus on direct transcription of optimal control problems solved with multiple-shooting methods. The framework takes as input the robot model and generates optimized trajectories that include both kinematic and dynamic quantities.

*supposedly

## Features

- [ ] Direct transcription of optimal control problems with multiple-shooting methods
- [ ] Support for floating-base robots, including humanoids and quadrupeds
- [ ] Integration with CasADi library for efficient numerical optimization
- [ ] Generation of optimized trajectories that include both kinematic and dynamic quantities
- [ ] Extensive documentation and examples to help you get started

## Installation
It is suggested to use [``conda``](https://docs.conda.io/en/latest/).
```bash
conda install -c conda-forge casadi pytest
pip install --no-deps -e .[all]
```

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
