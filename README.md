
## **Instructions**

Code based on the ACORM paper and repository.

## **Experiment instructions**

### **Installation instructions**
Download the Linux version 4.10 of StarCraft II from the Blizzard's [repository](https://github.com/Blizzard/s2client-proto#downloads). By default, the game is expected to be in `~/StarCraftII/` directory. 

Make sure you install the SMAC Maps.

See `requirments.txt` file for more information about how to install the dependencies.
```python
conda create -n acorm python=3.9.16 -y
conda activate acorm
pip install -r requirements.txt

# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 kmeans-pytorch
cd llm2vec
pip install -e .
pip install flash-attn --no-build-isolation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Run an experiment

You can execute the following command to run ACORM or R3DM based on QMIX with a map config, such as `MMM2`:

```python
python ./ACORM_QMIX/main.py --algorithm R3DM --env_name MMM2 --cluster_num 3 --max_train_steps 3050000
```

All results will be stored on wandb.

You can plot the curve with `seaborn`:

```python
python plot.py --algorithm 'R3DM' or 'ACORM'
```

## License

Code licensed under the Apache License v2.0.

## Run the baselines
We provide the example code
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

### Setting up SMACV2

Install SMACV2 from github
'''
pip install git+https://github.com/oxwhirl/smacv2.git
'''

Download the maps from [here](https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip) and place it inside SMAC_Maps within your StarcraftII installation.

After downloading SMACV2 maps, update scripts accordingly for SMACv2. For example, for GoMARL:
```
PYTHONPATH=/home/hg22723/projects/ACORMLLM/GoMARL/ python src/main.py --config=group --env-config=sc2v2_protoss_5_vs_5 with env_args.map_name=10gen_protoss env_args.seed=3
```



### If some moviepy errors come out please do this
'''
pip install --no-deps wandb[media]==0.18.7
pip install --no-deps --upgrade "moviepy<2.0.0"
'''
