#!/usr/bin/env bash

#cd "${HOME}/mocca-anomaly-detection/" || exit
#git pull
client_command='python main_shanghaitech2.py --load-lstm --bidirectional --clip-length=16 --code-length=512 --dropout=0.3 --epochs=300 --idx-list-enc=3,4,5,6 --wandb_group baseline --es_initial_patience_epochs 5 --es_patience 10 --compile_net'

# shellcheck disable=SC2087
ssh -T almogrote <<EOC
cd mocca-anomaly-detection
git pull
export PATH="\${HOME}/miniconda3/condabin:$PATH"
eval "\$(conda shell.bash hook)"
export FLWR_TELEMETRY_ENABLED=0
conda activate mocca
nohup ${client_command} -dp data/shanghaitech --wandb_name shang_earlytest --batch-size 16 >shang_earlytest.out 2>&1 </dev/null &
EOC

# shellcheck disable=SC2087
ssh -T platano <<EOC
cd mocca-anomaly-detection
git pull
export PATH="\${HOME}/miniconda3/condabin:$PATH"
eval "\$(conda shell.bash hook)"
export FLWR_TELEMETRY_ENABLED=0
conda activate mocca
nohup sh -c '${client_command} -dp data/UCSDped12 --wandb_name ped12_earlytest --batch-size 4 >ped12_earlytest.out 2>&1 </dev/null ; ${client_command} -dp data/UCSDped2 --wandb_name ped2_earlytest --batch-size 8 >ped2_earlytest.out 2>&1 </dev/null' >/dev/null 2>&1 </dev/null &
EOC

# shellcheck disable=SC2087
ssh -T citic <<EOC
cd PycharmProjects/mocca-anomaly-detection
git pull
export PATH="\${HOME}/miniconda3/condabin:$PATH"
eval "\$(conda shell.bash hook)"
export FLWR_TELEMETRY_ENABLED=0
conda activate mocca
nohup ${client_command} -dp data/UCSDped1 --wandb_name ped1_earlytest --batch-size 4 >ped1_earlytest.out 2>&1 </dev/null &
EOC

#ssh almogrote "pkill -9 python" ; ssh platano "pkill -9 python" ; pkill -9 python
