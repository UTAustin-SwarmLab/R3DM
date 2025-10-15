# R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning

## Overview

This repository contains the official implementation for R3DM accepted at the International Conference on Machine Learning (ICML) 2025. It includes the source code for the ACORM and R3DM algorithms, as well as the baseline models used for comparison in the StarCraft Multi-Agent Challenge (SMAC and SMACv2) environments.

[![Paper](https://img.shields.io/badge/paper-ICML%202025-blue.svg)](https://openreview.net/forum?id=VSIjdKPGp8)
[![Poster](https://img.shields.io/badge/poster-PDF-blue.svg)](https://icml.cc/media/PosterPDFs/ICML%202025/45066.png?t=1753041938.8506515)
[![Tweet](https://img.shields.io/badge/Tweet-X-black.svg)](https://x.com/hgoel1000/status/1944217308001710216)
[![License](https://img.shields.io/badge/License-Apache_2.0-red.svg)](LICENSE)

## Abstract

Multi-agent reinforcement learning (MARL) has achieved significant progress in large-scale traffic control, autonomous vehicles, and robotics. Drawing inspiration from biological systems where roles naturally emerge to enable coordination, role-based MARL methods have been proposed to enhance cooperation learning for complex tasks. However, existing methods exclusively derive roles from an agent's past experience during training, neglecting their influence on its future trajectories. This paper introduces a key insight: an agent's role should shape its future behavior to enable effective coordination. Hence, we propose Role Discovery and Diversity through Dynamics Models (R3DM), a novel role-based MARL framework that learns emergent roles by maximizing the mutual information between agents' roles, observed trajectories, and expected future behaviors. R3DM optimizes the proposed objective through contrastive learning on past trajectories to first derive intermediate roles that shape intrinsic rewards to promote diversity in future behaviors across different roles through a learned dynamics model. Benchmarking on SMAC and SMACv2 environments demonstrates that R3DM outperforms state-of-the-art MARL approaches, improving multi-agent coordination to increase win rates by up to 20%.

## Repository Structure
This repository is organized to facilitate the reproduction of experiments and further research. The structure is designed to be logical and self-contained, with distinct directories for the main algorithms and each baseline comparison. A brief overview of the directory structure is provided below:   

```
.
├── ACORM_QMIX/         # Source code for the proposed ACORM and R3DM algorithms.
├── CIA/                # Source code for the CIA baseline algorithm.
├── CDS/                # Source code for the CDS baseline algorithm.
├── GoMARL/             # Source code for the GoMARL baseline algorithm.
├── EMC/                # Source code for the EMC baseline algorithm.
├── llm2vec/            # Source code for the llm2vec dependency module.
├── plot.py             # Python script for plotting experimental results from wandb.
├── requirements.txt    # A list of required Python packages for the project.
└── README.md           # This documentation file.
```

The code is based on the ACORM paper and repository.

## Environment Setup and Installation
This section provides a comprehensive, step-by-step guide to setting up the necessary environment for running all experiments. Following these instructions carefully is crucial for ensuring reproducibility.   

### Prerequisites:

1. A Linux-based operating system.
2. NVIDIA GPU with CUDA 11.8 support.
3. Anaconda or Miniconda for managing Python environments.
4. Fit for cloning the repository.


### Step 1: Clone the Repository
First, clone this repository to your local machine using the following command:


git clone https://github.com/your-username/ACORM.git
cd ACORM

### Step 2: Install StarCraft II and SMAC Maps

1. The experiments are conducted using the StarCraft II Learning Environment. Both the game and the specific map packs must be installed correctly.
2. Download StarCraft II: Download the Linux version 4.10 of StarCraft II from the official Blizzard repository:(https://github.com/Blizzard/s2client-proto#downloads).
3. Install StarCraft II: By default, the game must be unzipped and placed in your home directory at ~/StarCraftII/.

Important Note: If you install the game in a different location, you must set the SC2PATH environment variable to point to your installation directory. For example:

```
export SC2PATH="/path/to/your/StarCraftII/"
```

This variable is required by some of the baseline scripts to locate the game executable.

Install SMACv1 Maps: Download the original SMAC maps and place them in the Maps directory of your StarCraft II installation.
Install SMACv2 Maps: For experiments on SMACv2 environments, additional maps are required.

Install the smacv2 package:

```
pip install git+https://github.com/oxwhirl/smacv2.git
```
Download the SMACv2 maps from the official release:(https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip).

Unzip the downloaded file and place the contents into the SMAC_Maps directory within your StarCraft II installation (e.g., ~/StarCraftII/Maps/SMAC_Maps/).

### Step 3: Create and Activate the Conda Environment
A dedicated Conda environment is used to manage dependencies and ensure a consistent setup.   

```
conda create -n acorm python=3.9.16 -y
conda activate acorm
```

### Step 4: Install Python Dependencies
Install the required Python packages using the provided requirements.txt file and additional pip commands. The commands should be run from the root of the cloned repository.

```
# Install packages from the requirements file
pip install -r requirements.txt


# Install flash-attention
pip install flash-attn --no-build-isolation

```

### Step 5: Install PyTorch with CUDA Support
To ensure full compatibility with the required CUDA version and to prevent common driver-related issues, it is highly recommended to install PyTorch and its related libraries through Conda. This method correctly handles the CUDA toolkit dependencies within the environment.
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 6: Troubleshooting Common Issues
This section addresses known issues that may arise during installation.

Moviepy Errors with wandb
If you encounter errors related to moviepy during installation or when logging experiment videos to Weights & Biases (wandb), it is likely due to a version incompatibility. To resolve this, run the following commands to install specific, compatible versions of wandb[media] and moviepy:

```
pip install --no-deps wandb[media]==0.18.7
pip install --no-deps --upgrade "moviepy<2.0.0"
```

## Reproducing Experimental Results

This section provides the necessary commands to run the experiments for ACORM, R3DM, and all baseline algorithms.

A Note on Environment Variables
For the scripts to execute correctly, the PYTHONPATH must be set to include the relevant project directories. The provided commands handle this by prepending PYTHONPATH=$(pwd)/<SUBDIR> to the execution command. This approach replaces hardcoded, user-specific paths with dynamic ones, making the commands universally executable from the root of the cloned repository.   

All experimental results, including performance metrics and training curves, are logged to Weights & Biases (wandb). Ensure you have an account and have logged in via the command line before starting any experiments.

## Quick Start: Command Reference
For experienced users, the following table provides a quick reference to the primary execution commands for a representative map in each algorithm and environment combination. Detailed explanations and further options are provided in the subsequent sections.

Algorithm	Environment	Example Map / Config	Execution Command (from repository root)

| Algorithm      | Environment | Example Map / Config | Execution Command (from repository root)                                                                                                                              |
| :------------- | :---------- | :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ACORM/R3DM** | SMACv1      | `MMM2`               | `python ./ACORM_QMIX/main.py --algorithm R3DM --env_name MMM2 --cluster_num 3 --max_train_steps 3050000`                                                              |
| **CIA** | SMACv1      | `3s5z_vs_3s6z`       | `PYTHONPATH=$(pwd)/CIA python CIA/src/main.py --config=cia_grad_qmix_3s5z_vs_3s6z --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=3`                  |
| **CDS** | SMACv1      | `3s5z_vs_3s6z`       | `PYTHONPATH=$(pwd)/CDS/CDS_SMAC/QPLEX-master-SC2/pymarl-master/ SC2_PATH=$SC2PATH python CDS/CDS_SMAC/QPLEX-master-SC2/pymarl-master/src/main.py --config=qplex_qatten_sc2 --env-config=sc2_3s5z_vs_3s6z with env_args.map_name=3s5z_vs_3s6z env_args.seed=3` |
| **GoMARL** | SMACv1      | `3s5z_vs_3s6z`       | `PYTHONPATH=$(pwd)/GoMARL/ python GoMARL/src/main.py --config=group --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=3`                                  |
| **EMC** | SMACv2      | `protoss_5_vs_5`     | `PYTHONPATH=$(pwd)/EMC/pymarl python EMC/src/main.py --config=sc2v2_protoss_5vs_5 --env-config=sc2v2_protoss_5_vs_5 with env_args.seed=3`                                |


Running ACORM and R3DM (Main Algorithms)
The main algorithms, ACORM and R3DM, are executed using the main.py script located in the ACORM_QMIX directory.

Example Command:

```
python3 ./ACORM_QMIX/main.py --algorithm R3DM --env_name MMM2 --cluster_num 3 --max_train_steps 3050000
```

Key Arguments:
Providing explanations for key arguments empowers other researchers to not only replicate results but also to adapt the code for their own research questions, such as testing the algorithm on new maps or with different hyperparameters. This practice elevates the repository from a static artifact to a dynamic tool for the community.

```
--algorithm: Specifies the algorithm to run. Options are ACORM or R3DM.

--env_name: The name of the SMAC map to use for the experiment (e.g., MMM2, 8m, 3s5z).

--cluster_num: A key hyperparameter for the algorithm, defining the number of clusters.

--max_train_steps: The total number of training timesteps for the experiment.
```

We also provide the scripts for launching experiments all seeds here.
```
sh ACORM_QMIX/r3dm.sh
sh ACORM_QMIX/acorm.sh
```

## Running Baseline Algorithms
The following subsections provide the commands for running each of the baseline algorithms. Note the distinct command structures and configurations for SMACv1 and SMACv2 environments.

### CIA
Run experiments within CIA folder using the following:
```
PYTHONPATH=/home/hg22723/projects/ACORMLLM/CIA/ python src/main.py --config=cia_grad_qmix_3s5z_vs_3s6z --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=3
```

For SMACV2 environments that start with protoss, terran or zerg as sample command is as follows (dont specify map_name)
```
PYTHONPATH=/home/hg22723/projects/ACORMLLM/CIA/ python src/main.py --config=cia_grad_qmix_3s5z_vs_3s6z --env-config=sc2_protoss_5_vs_5 with env_args.seed=3
```

### CDS
1. 
Run experiments within the CDS/CDS_SMAC/QPLEX-master-SC2/pymarl-master folder with the following command:
2. Copy the StarcraftII installation done after following earlier steps and place it in pymarl-master/3rdparty 
3.  Following is the sample command to then run the experiments for CDS
```
PYTHONPATH=/home/hg22723/projects/ACORMLLM/CDS/CDS_SMAC/QPLEX-master-SC2/pymarl-master/ SC2_PATH=/home/hg22723/ python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_3s5z_vs_3s6z with env_args.map_name=3s5z_vs_3s6z env_args.seed=3
```

For SMACV2 environments that start with protoss, terran or zerg as sample command is as follows (dont specify map_name)
```
PYTHONPATH=/home/hg22723/projects/ACORMLLM/CDS/CDS_SMAC/QPLEX-master-SC2/pymarl-master/ SC2_PATH=/home/hg22723/ python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_protoss_5_vs_5 with env_args.seed=3
```

### GoMARL
Run experiments within GoMARL folder using the following from SMACV1 environments
```
PYTHONPATH=/home/hg22723/projects/ACORMLLM/GoMARL/ python src/main.py --config=group --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=3
```

For SMACV2 environments that start with protoss, terran or zerg as sample command is as follows (dont specify map_name)
```
PYTHONPATH=/home/hg22723/projects/ACORMLLM/GoMARL/ python src/main.py --config=group --env-config=sc2v2_protoss_5_vs_5 with env_args.seed=3
```

### EMC
```
PYTHONPATH=/home/hg22723/projects/ACORMLLM/EMC/pymarl python src/main.py --config=sc2v2_protoss_5vs_5 --env-config=sc2v2_protoss_5_vs_5 with env_args.seed=3
```


## License
Code licensed under the Apache License v2.0.

## Citation

If you find our work useful, please use this citation.

```
@inproceedings{goelr3dm,
  title={R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning},
  author={Goel, Harsh and Omama, Mohammad and Chalaki, Behdad and Tadiparthi, Vaishnav and Pari, Ehsan Moradi and Chinchali, Sandeep P},
  booktitle={Forty-second International Conference on Machine Learning}
}
```


