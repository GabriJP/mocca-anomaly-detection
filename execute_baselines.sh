#!/usr/bin/env bash

cd "${HOME}/PycharmProjects/mocca-anomaly-detection/" || exit
git pull
client_command="python main_shanghaitech2.py --load-lstm --bidirectional --clip-length=16 --code-length=512 --dropout=0.3 --epochs=300 --idx-list-enc=3,4,5,6 --wandb_group baseline --es_initial_patience_epochs 1 --es_patience 10 --seed=3"

# shellcheck disable=SC2087
ssh almogrote <<EOC
cd mocca-anomaly-detection
git pull
export PATH="\${HOME}/miniconda3/condabin:$PATH"
eval "\$(conda shell.bash hook)"
export FLWR_TELEMETRY_ENABLED=0
conda activate mocca
nohup ${client_command} -dp data/shanghaitech  --wandb_name shang_earlytest_seed --batch-size 16  >shang_earlytest_seed.out 2>&1 </dev/null &
EOC

# shellcheck disable=SC2087
ssh platano <<EOC
cd mocca-anomaly-detection
git pull
export PATH="\${HOME}/miniconda3/condabin:$PATH"
eval "\$(conda shell.bash hook)"
export FLWR_TELEMETRY_ENABLED=0
conda activate mocca
nohup sh -c "${client_command} -dp data/UCSDped12  --wandb_name ped12_earlytest_seed --batch-size 8  >ped12_earlytest_seed.out 2>&1 ; ${client_command} -dp data/UCSDped2  --wandb_name ped2_earlytest_seed --batch-size 8 >ped2_earlytest_seed.out 2>&1" </dev/null &
EOC

# shellcheck disable=SC2087
ssh citic <<EOC
cd PycharmProjects/mocca-anomaly-detection
git pull
export PATH="\${HOME}/miniconda3/condabin:$PATH"
eval "\$(conda shell.bash hook)"
export FLWR_TELEMETRY_ENABLED=0
conda activate mocca
nohup ${client_command} -dp data/UCSDped1  --wandb_name ped1_earlytest_seed --batch-size 4  >ped1_earlytest_seed.out 2>&1 </dev/null &
EOC
