#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python src/main.py --config=group --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=1 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --config=group --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=group --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=group --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=4 &
CUDA_VISIBLE_DEVICES=4 python src/main.py --config=group --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z env_args.seed=5 &


CUDA_VISIBLE_DEVICES=0 python src/main.py --config=group --env-config=sc2 with env_args.map_name=6h_vs_8z env_args.seed=1 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --config=group --env-config=sc2 with env_args.map_name=6h_vs_8z env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=group --env-config=sc2 with env_args.map_name=6h_vs_8z env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=group --env-config=sc2 with env_args.map_name=6h_vs_8z env_args.seed=4 &
CUDA_VISIBLE_DEVICES=4 python src/main.py --config=group --env-config=sc2 with env_args.map_name=6h_vs_8z env_args.seed=5 &


CUDA_VISIBLE_DEVICES=0 python src/main.py --config=group --env-config=sc2 with env_args.map_name=MMM2 env_args.seed=1 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --config=group --env-config=sc2 with env_args.map_name=MMM2 env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=group --env-config=sc2 with env_args.map_name=MMM2 env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=group --env-config=sc2 with env_args.map_name=MMM2 env_args.seed=4 &
CUDA_VISIBLE_DEVICES=4 python src/main.py --config=group --env-config=sc2 with env_args.map_name=MMM2 env_args.seed=5 &


CUDA_VISIBLE_DEVICES=0 python src/main.py --config=group --env-config=sc2 with env_args.map_name=27m_vs_30m env_args.seed=1 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --config=group --env-config=sc2 with env_args.map_name=27m_vs_30m env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=group --env-config=sc2 with env_args.map_name=27m_vs_30m env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=group --env-config=sc2 with env_args.map_name=27m_vs_30m env_args.seed=4 &
CUDA_VISIBLE_DEVICES=4 python src/main.py --config=group --env-config=sc2 with env_args.map_name=27m_vs_30m env_args.seed=5 &