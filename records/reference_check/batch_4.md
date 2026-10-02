# Reference batch 4 (8 references)
## [4] schumacher2023
Bib entry:
```
@inproceedings{schumacher2023,
  author    = {Pierre Schumacher and Daniel F. B. Haeufle and Dieter B{\"u}chler and Syn Schmitt and Georg Martius},
  title     = {{DEP-RL}: Embodied exploration for reinforcement learning in overactuated and musculoskeletal systems},
  booktitle = {Proc. Int. Conf. Learning Representations (ICLR)},
  year      = {2023}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- Deep reinforcement learning on such bodies is slow and is mostly model-free \cite{berg2023,chiappa2023,schumacher2023}.

## [12] janner2019
Bib entry:
```
@inproceedings{janner2019,
  author    = {Michael Janner and Justin Fu and Marvin Zhang and Sergey Levine},
  title     = {When to trust your model: Model-based policy optimization},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {32},
  year      = {2019}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- Imagined rollouts add data whatever the model contains, and an inaccurate model can make model-based learning worse than model-free \cite{janner2019,voelcker2022}; how much of a babble-trained model's effect is due to what it has learned, and how that depends on the body, has not been isolated.
- MBPO \cite{janner2019} branches short rollouts from real states.
- Model error can make model-based learning worse than model-free, more so in higher dimension \cite{janner2019,voelcker2022}.

## [20] soderstrom2015
Bib entry:
```
@article{soderstrom2015,
  author  = {Nicholas C. Soderstrom and Robert A. Bjork},
  title   = {Learning versus performance: An integrative review},
  journal = {Perspectives on Psychological Science},
  volume  = {10},
  number  = {2},
  pages   = {176--199},
  year    = {2015}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- Performance while guided is not the same as what has been learned \cite{schmidt1992,soderstrom2015}.

## [28] yuan2020
Bib entry:
```
@inproceedings{yuan2020,
  author    = {Li Yuan and Francis E. H. Tay and Guilin Li and Tao Wang and Jiashi Feng},
  title     = {Revisiting knowledge distillation via label smoothing regularization},
  booktitle = {Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- Supervised distillation uses the student itself or a random network as the teacher to separate what the term does from what the teacher knows \cite{yuan2020,sarnthein2023}; a term toward the learner's own earlier policy stabilises fine-tuning \cite{li2023proto,luo2023,zhao2022}, and a cloned policy fine-tuned by off-policy reinforcement often degrades when the constraint is removed \cite{nair2020,uchendu2023,zhang2023}.

## [36] lungarella2003
Bib entry:
```
@article{lungarella2003,
  author  = {M. Lungarella and G. Metta and R. Pfeifer and G. Sandini},
  title   = {Developmental robotics: a survey},
  journal = {Connection Science},
  volume  = {15}, number = {4}, pages = {151--190}, year = {2003},
  doi     = {10.1080/09540090310001655110}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- \textbf{Body models.} Developmental robotics \cite{lungarella2003,asada2009,cangelosi2015} treats motor babbling as a way to learn forward models of the self \cite{dearden2005}, and intrinsic motivation as what drives it \cite{oudeyer2007}; self-modelling \cite{bongard2006} and goal babbling \cite{rolf2010} learn body models by directed exploration.

## [44] hafner2021
Bib entry:
```
@inproceedings{hafner2021,
  author    = {Danijar Hafner and Timothy Lillicrap and Mohammad Norouzi and Jimmy Ba},
  title     = {Mastering {Atari} with discrete world models},
  booktitle = {Proc. Int. Conf. Learning Representations (ICLR)},
  year      = {2021}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- In model-based reinforcement learning, World Models \cite{ha2018} trains a model once from random rollouts; the Dreamer series \cite{hafner2020,hafner2021,hafner2025} and TD-MPC2 \cite{hansen2024} keep the model learning online;

## [52] feng2023
Bib entry:
```
@inproceedings{feng2023,
  author    = {Yusen Feng and Xiyan Xu and Libin Liu},
  title     = {{MuscleVAE}: Model-based controllers of muscle-actuated characters},
  booktitle = {Proc. SIGGRAPH Asia Conference Papers},
  year      = {2023},
  doi       = {10.1145/3610548.3618137}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- Muscle world models are co-learned with the skill \cite{feng2023,kim2025}, and frozen models learned from reward-free actuation drive soft and muscle-driven bodies by planning or model-only learning \cite{berdica2024,wu2024modex,zheng2025}.

## [60] sutton1991
Bib entry:
```
@article{sutton1991,
  author  = {Richard S. Sutton},
  title   = {Dyna, an integrated architecture for learning, planning, and reacting},
  journal = {ACM SIGART Bulletin},
  volume  = {2},
  number  = {4},
  pages   = {160--163},
  year    = {1991},
  doi     = {10.1145/122344.122377}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- It is used for Dyna \cite{sutton1991} rollouts of three steps branched from 200 real states every 100 environment steps; the reward at an imagined state is the environment's analytic reach reward, computed with the simulator's forward kinematics.

