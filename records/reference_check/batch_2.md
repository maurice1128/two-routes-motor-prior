# Reference batch 2 (8 references)
## [2] berg2023
Bib entry:
```
@inproceedings{berg2023,
  author    = {Cameron H. Berg and Vittorio Caggiano and Vikash Kumar},
  title     = {{SAR}: Generalization of physiological dexterity via synergistic action representation},
  booktitle = {Proc. Robotics: Science and Systems (RSS)},
  year      = {2023},
  doi       = {10.15607/RSS.2023.XIX.007}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- Deep reinforcement learning on such bodies is slow and is mostly model-free \cite{berg2023,chiappa2023,schumacher2023}.

## [10] schmitt2018
Bib entry:
```
@misc{schmitt2018,
  author       = {Simon Schmitt and Jonathan J. Hudson and Augustin Zidek and Simon Osindero and Carl Doersch and Wojciech M. Czarnecki and Joel Z. Leibo and Heinrich Kuttler and Andrew Zisserman and Karen Simonyan and S. M. Ali Eslami},
  title        = {Kickstarting deep reinforcement learning},
  howpublished = {arXiv:1803.03835},
  year         = {2018}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- Motor-learning research holds that guidance can leave a learner dependent on it \cite{salmoni1984,schmidt1992}, and teacher--student pipelines in robotics cut a distillation term to make the student independent \cite{schmitt2018,agarwal2022}.
- In teacher--student learning, a teacher that sees privileged state is distilled into a student on the student's own rollouts \cite{chen2019lbc,lee2020,miki2022,kumar2021}, and the distillation term is annealed or cut to make the student independent \cite{schmitt2018,agarwal2022}.

## [18] mckay2022
Bib entry:
```
@article{mckay2022,
  author  = {Brad McKay and Julia Hussien and Mary-Anne Vinh and Alexandre Mir-Orefice and Hugh Brooks and Diane M. Ste-Marie},
  title   = {Meta-analysis of the reduced relative feedback frequency effect on motor learning and performance},
  journal = {Psychology of Sport and Exercise},
  volume  = {61}, pages = {102165}, year = {2022},
  doi     = {10.1016/j.psychsport.2022.102165}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- The evidence is mixed: one meta-analysis finds the degrading effect limited to laboratory tasks, parameter learning and extensive practice \cite{marschall2007}, a later one finds no reliable benefit of reduced feedback frequency \cite{mckay2022}, and children given reduced feedback learn worse \cite{sullivan2008}.

## [26] miki2022
Bib entry:
```
@article{miki2022,
  author  = {Takahiro Miki and Joonho Lee and Jemin Hwangbo and Lorenz Wellhausen and Vladlen Koltun and Marco Hutter},
  title   = {Learning robust perceptive locomotion for quadrupedal robots in the wild},
  journal = {Science Robotics},
  volume  = {7},
  number  = {62},
  pages   = {eabk2822},
  year    = {2022},
  doi     = {10.1126/scirobotics.abk2822}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- In teacher--student learning, a teacher that sees privileged state is distilled into a student on the student's own rollouts \cite{chen2019lbc,lee2020,miki2022,kumar2021}, and the distillation term is annealed or cut to make the student independent \cite{schmitt2018,agarwal2022}.

## [34] uchendu2023
Bib entry:
```
@inproceedings{uchendu2023,
  author    = {Ikechukwu Uchendu and Ted Xiao and Yao Lu and Banghua Zhu and Mengyuan Yan and Jos{\'e}phine Simon and Matthew Bennice and Chuyuan Fu and Cong Ma and Jiantao Jiao and Sergey Levine and Karol Hausman},
  title     = {Jump-start reinforcement learning},
  booktitle = {Proc. 40th Int. Conf. Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {202},
  pages     = {34556--34583},
  year      = {2023}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- Supervised distillation uses the student itself or a random network as the teacher to separate what the term does from what the teacher knows \cite{yuan2020,sarnthein2023}; a term toward the learner's own earlier policy stabilises fine-tuning \cite{li2023proto,luo2023,zhao2022}, and a cloned policy fine-tuned by off-policy reinforcement often degrades when the constraint is removed \cite{nair2020,uchendu2023,zhang2023}.

## [42] ha2018
Bib entry:
```
@inproceedings{ha2018,
  author    = {David Ha and J{\"u}rgen Schmidhuber},
  title     = {Recurrent world models facilitate policy evolution},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {31},
  year      = {2018}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- In model-based reinforcement learning, World Models \cite{ha2018} trains a model once from random rollouts; the Dreamer series \cite{hafner2020,hafner2021,hafner2025} and TD-MPC2 \cite{hansen2024} keep the model learning online;

## [50] yu2020
Bib entry:
```
@inproceedings{yu2020,
  author    = {Tianhe Yu and Garrett Thomas and Lantao Yu and Stefano Ermon and James Y. Zou and Sergey Levine and Chelsea Finn and Tengyu Ma},
  title     = {{MOPO}: Model-based offline policy optimization},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {33},
  year      = {2020}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- Plan2Explore \cite{sekar2020} and APV \cite{seo2022} pretrain a model without task reward; offline methods \cite{kidambi2020,yu2020,yu2021} fit a model to logged, rewarded data.

## [58] haarnoja2018
Bib entry:
```
@inproceedings{haarnoja2018,
  author    = {Tuomas Haarnoja and Aurick Zhou and Pieter Abbeel and Sergey Levine},
  title     = {Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor},
  booktitle = {Proc. 35th Int. Conf. Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {80},
  pages     = {1861--1870},
  year      = {2018}
}
```
Sentences in the paper that cite it (\cite{...} shows all keys cited together):
- One soft actor-critic (SAC) \cite{haarnoja2018,haarnoja2018b} learner is used in every condition: two hidden layers of 128 units, update-to-data ratio 2, batch 128, learning rate $3\times10^{-4}$, automatic entropy temperature.

