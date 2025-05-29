#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_3s5z_vs_3s6z with env_args.seed=1 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_3s5z_vs_3s6z with env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_3s5z_vs_3s6z with env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_3s5z_vs_3s6z with env_args.seed=4 &
CUDA_VISIBLE_DEVICES=4 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_3s5z_vs_3s6z with env_args.seed=5 &


CUDA_VISIBLE_DEVICES=0 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_6h_vs_8z with env_args.seed=1 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_6h_vs_8z with env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_6h_vs_8z with env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_6h_vs_8z with env_args.seed=4 &
CUDA_VISIBLE_DEVICES=4 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_6h_vs_8z with env_args.seed=5 &


CUDA_VISIBLE_DEVICES=0 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_MMM2 with env_args.seed=1 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_MMM2 with env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_MMM2 with env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_MMM2 with env_args.seed=4 &
CUDA_VISIBLE_DEVICES=4 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_MMM2 with env_args.seed=5 &


CUDA_VISIBLE_DEVICES=0 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_27m_vs_30m with env_args.seed=1 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_27m_vs_30m with env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_27m_vs_30m with env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_27m_vs_30m with env_args.seed=4 &
CUDA_VISIBLE_DEVICES=4 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_27m_vs_30m with env_args.seed=5 &

CUDA_VISIBLE_DEVICES=0 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_corridor with env_args.seed=1 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_corridor with env_args.seed=2 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_corridor with env_args.seed=3 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_corridor with env_args.seed=4 &
CUDA_VISIBLE_DEVICES=4 python src/main.py --config=qplex_qatten_sc2 --env-config=sc2_corridor with env_args.seed=5 &