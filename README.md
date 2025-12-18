# Adapting PAD to Trifinger Robotic Manipulation in Causal World

This repository contains the modified [codebase](https://github.com/nicklashansen/policy-adaptation-during-deployment) from the paper 
```
@article{hansen2020deployment,
  title={Self-Supervised Policy Adaptation during Deployment},
  author={Nicklas Hansen and Rishabh Jangir and Yu Sun and Guillem Alenyà and Pieter Abbeel and Alexei A. Efros and Lerrel Pinto and Xiaolong Wang},
  year={2020},
  eprint={2007.04309},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

The main changes involved adapting the framework to a different simulation and task. In particular, the original model used image sequences as input, which was converted to state vectors by replacing the CNN encoder with a simple MLP. 

Links: [Project page](https://alestaubin.github.io/publication/202512-CPSC532X) 
