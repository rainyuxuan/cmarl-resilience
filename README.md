# Resilience of Cooperative Multi-Agent Reinforcement Learning

## Setup

### Conda Environment

```bash
conda create -n cmarl python=3.9
conda activate cmarl
pip install -r requirements.txt
```

### Install MARLlib

```bash
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib
pip install --upgrade pip
pip install -r requirements.txt

# we recommend the gym version between 0.20.0~0.22.0.
pip install gym>=0.20.0,<0.22.0

# add patch files to MARLlib
python patch/add_patch.py -y
```

### Reset Conda Environment

```bash
conda deactivate
conda remove --name cmarl --all
```

## References

```bibtex
@inproceedings{zheng2018magent,
  title={MAgent: A many-agent reinforcement learning platform for artificial collective intelligence},
  author={Zheng, Lianmin and Yang, Jiacheng and Cai, Han and Zhou, Ming and Zhang, Weinan and Wang, Jun and Yu, Yong},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}
```