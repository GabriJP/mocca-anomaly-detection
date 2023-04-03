#!/usr/bin/env bash

python3 main_shanghaitech.py -dp data/UCSDped2 --end-to-end-training --code-length 1024 --epochs 10 --test --load-lstm --seed 3 --wandb_group seeds --wandb_name seed3_1
python3 main_shanghaitech.py -dp data/UCSDped2 --end-to-end-training --code-length 1024 --epochs 10 --test --load-lstm --seed 3 --wandb_group seeds --wandb_name seed3_2
python3 main_shanghaitech.py -dp data/UCSDped2 --end-to-end-training --code-length 1024 --epochs 10 --test --load-lstm --seed 3 --wandb_group seeds --wandb_name seed3_3

python3 main_shanghaitech.py -dp data/UCSDped2 --end-to-end-training --code-length 1024 --epochs 10 --test --load-lstm --seed 50 --wandb_group seeds --wandb_name seed50_1
python3 main_shanghaitech.py -dp data/UCSDped2 --end-to-end-training --code-length 1024 --epochs 10 --test --load-lstm --seed 50 --wandb_group seeds --wandb_name seed50_2
python3 main_shanghaitech.py -dp data/UCSDped2 --end-to-end-training --code-length 1024 --epochs 10 --test --load-lstm --seed 50 --wandb_group seeds --wandb_name seed50_3
