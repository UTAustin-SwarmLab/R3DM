#!/bin/bash
CUDA_VISIBLE_DEVICES=4 python src/main.py --config=EMC_sc2_2c_vs_64zg --env-config=sc2 with env_args.map_name=2c_vs_64zg env_args.seed=1 &
CUDA_VISIBLE_DEVICES=7 python src/main.py --config=EMC_sc2_2c_vs_64zg --env-config=sc2 with env_args.map_name=2c_vs_64zg env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=EMC_sc2_2c_vs_64zg --env-config=sc2 with env_args.map_name=2c_vs_64zg env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=EMC_sc2_2c_vs_64zg --env-config=sc2 with env_args.map_name=2c_vs_64zg env_args.seed=4 &
CUDA_VISIBLE_DEVICES=5 python src/main.py --config=EMC_sc2_2c_vs_64zg --env-config=sc2 with env_args.map_name=2c_vs_64zg env_args.seed=5 &


CUDA_VISIBLE_DEVICES=4 python src/main.py --config=EMC_sc2_6h_vs_8z --env-config=sc2 with env_args.map_name=6h_vs_8z env_args.seed=1 &
CUDA_VISIBLE_DEVICES=7 python src/main.py --config=EMC_sc2_6h_vs_8z --env-config=sc2 with env_args.map_name=6h_vs_8z env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=EMC_sc2_6h_vs_8z --env-config=sc2 with env_args.map_name=6h_vs_8z env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=EMC_sc2_6h_vs_8z --env-config=sc2 with env_args.map_name=6h_vs_8z env_args.seed=4 &
CUDA_VISIBLE_DEVICES=5 python src/main.py --config=EMC_sc2_6h_vs_8z --env-config=sc2 with env_args.map_name=6h_vs_8z env_args.seed=5 &


CUDA_VISIBLE_DEVICES=4 python src/main.py --config=EMC_sc2_MMM2 --env-config=sc2 with env_args.map_name=MMM2 env_args.seed=1 &
CUDA_VISIBLE_DEVICES=7 python src/main.py --config=EMC_sc2_MMM2 --env-config=sc2 with env_args.map_name=MMM2 env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=EMC_sc2_MMM2 --env-config=sc2 with env_args.map_name=MMM2 env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=EMC_sc2_MMM2 --env-config=sc2 with env_args.map_name=MMM2 env_args.seed=4 &
CUDA_VISIBLE_DEVICES=5 python src/main.py --config=EMC_sc2_MMM2 --env-config=sc2 with env_args.map_name=MMM2 env_args.seed=5 &


CUDA_VISIBLE_DEVICES=4 python src/main.py --config=EMC_sc2_corridor --env-config=sc2 with env_args.map_name=corridor env_args.seed=1 &
CUDA_VISIBLE_DEVICES=7 python src/main.py --config=EMC_sc2_corridor --env-config=sc2 with env_args.map_name=corridor env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=EMC_sc2_corridor --env-config=sc2 with env_args.map_name=corridor env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=EMC_sc2_corridor --env-config=sc2 with env_args.map_name=corridor env_args.seed=4 &
CUDA_VISIBLE_DEVICES=5 python src/main.py --config=EMC_sc2_corridor --env-config=sc2 with env_args.map_name=corridor env_args.seed=5 &

CUDA_VISIBLE_DEVICES=4 python src/main.py --config=EMC_sc2_3s5z_vs_3s6z --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=1 &
CUDA_VISIBLE_DEVICES=7 python src/main.py --config=EMC_sc2_3s5z_vs_3s6z --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=EMC_sc2_3s5z_vs_3s6z --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=EMC_sc2_3s5z_vs_3s6z --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=4 &
CUDA_VISIBLE_DEVICES=5 python src/main.py --config=EMC_sc2_3s5z_vs_3s6z --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=5 &

CUDA_VISIBLE_DEVICES=4 python src/main.py --config=EMC_sc2_5m_vs_6m --env-config=sc2 with env_args.map_name=5m_vs_6m env_args.seed=1 &
CUDA_VISIBLE_DEVICES=7 python src/main.py --config=EMC_sc2_5m_vs_6m --env-config=sc2 with env_args.map_name=5m_vs_6m env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=EMC_sc2_5m_vs_6m --env-config=sc2 with env_args.map_name=5m_vs_6m env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=EMC_sc2_5m_vs_6m --env-config=sc2 with env_args.map_name=5m_vs_6m env_args.seed=4 &
CUDA_VISIBLE_DEVICES=5 python src/main.py --config=EMC_sc2_5m_vs_6m --env-config=sc2 with env_args.map_name=5m_vs_6m env_args.seed=5 &