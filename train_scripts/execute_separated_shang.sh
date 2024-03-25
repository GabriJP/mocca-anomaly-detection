#!/usr/bin/env bash

#cd "${HOME}/mocca-anomaly-detection/" || exit
#git pull
subset=${1:-'separated'}
client_command="python main_shanghaitech2.py --load-lstm --bidirectional --clip-length=16 --code-length=512 --dropout=0.3 --epochs=300 --idx-list-enc=3,4,5,6 --wandb-group=${subset} --es_initial_patience-epochs=5 --es-patience=10 --compile-net"

# shellcheck disable=SC2087
ssh -T almogrote <<EOC
cd mocca-anomaly-detection
git pull
export PATH="\${HOME}/miniconda3/condabin:$PATH"
eval "\$(conda shell.bash hook)"
export FLWR_TELEMETRY_ENABLED=0
conda activate mocca
nohup sh -c '\
${client_command} -dp data/shanghaitech/${subset}/shang01 --wandb-name=shang01 --batch-size=16 >shang01.out 2>&1 </dev/null ; \
${client_command} -dp data/shanghaitech/${subset}/shang02 --wandb-name=shang02 --batch-size=16 >shang02.out 2>&1 </dev/null ; \
${client_command} -dp data/shanghaitech/${subset}/shang03 --wandb-name=shang03 --batch-size=16 >shang03.out 2>&1 </dev/null ; \
${client_command} -dp data/shanghaitech/${subset}/shang04 --wandb-name=shang04 --batch-size=16 >shang04.out 2>&1 </dev/null ; \
' >/dev/null 2>&1 </dev/null &
EOC

# shellcheck disable=SC2087
ssh -T platano <<EOC
cd mocca-anomaly-detection
git pull
export PATH="\${HOME}/miniconda3/condabin:$PATH"
eval "\$(conda shell.bash hook)"
export FLWR_TELEMETRY_ENABLED=0
conda activate mocca
nohup sh -c '\
${client_command} -dp data/shanghaitech/${subset}/shang05 --wandb-name=shang05 --batch-size=8 >shang05.out 2>&1 </dev/null ; \
${client_command} -dp data/shanghaitech/${subset}/shang06 --wandb-name=shang06 --batch-size=8 >shang06.out 2>&1 </dev/null ; \
${client_command} -dp data/shanghaitech/${subset}/shang07 --wandb-name=shang07 --batch-size=8 >shang07.out 2>&1 </dev/null ; \
${client_command} -dp data/shanghaitech/${subset}/shang08 --wandb-name=shang08 --batch-size=8 >shang08.out 2>&1 </dev/null ; \
' >/dev/null 2>&1 </dev/null &
EOC

# shellcheck disable=SC2087
ssh -T citic <<EOC
cd PycharmProjects/mocca-anomaly-detection
git pull
export PATH="\${HOME}/miniconda3/condabin:$PATH"
eval "\$(conda shell.bash hook)"
export FLWR_TELEMETRY_ENABLED=0
conda activate mocca
nohup sh -c '\
${client_command} -dp data/shanghaitech/${subset}/shang10 --wandb-name=shang10 --batch-size=4 >shang10.out 2>&1 </dev/null ; \
${client_command} -dp data/shanghaitech/${subset}/shang11 --wandb-name=shang11 --batch-size=4 >shang11.out 2>&1 </dev/null ; \
${client_command} -dp data/shanghaitech/${subset}/shang12 --wandb-name=shang12 --batch-size=4 >shang12.out 2>&1 </dev/null ; \
${client_command} -dp data/shanghaitech/${subset}/shang13 --wandb-name=shang13 --batch-size=4 >shang13.out 2>&1 </dev/null ; \
' >/dev/null 2>&1 </dev/null &
EOC

#ssh almogrote "pkill -9 python" ; ssh platano "pkill -9 python" ; pkill -9 python
