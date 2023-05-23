#!/usr/bin/env bash

cd "${HOME}/PycharmProjects/mocca-anomaly-detection/" || exit
git pull
#export PATH="${HOME}/miniconda3/bin:$PATH"
#eval "$(conda shell.bash hook)"
#conda activate mocca
#gid=$(python -c "import wandb ; print(wandb.util.generate_id())")
gid="not_so_large_fed_ped"
client_command="./execute.sh --load-lstm --bidirectional --clip-length=16 --code-length=512 --dropout=0.3 --idx-list-enc=3,4,5,6 --wandb_group ${gid}"

ssh xavier "cd mocca-anomaly-detection && git pull"
ssh platano "cd mocca-anomaly-detection && git pull"
ssh almogrote "cd mocca-anomaly-detection && git pull"

# shellcheck disable=SC2029
ssh xavier "nohup ./execute.sh --num_rounds 50 --epochs 2 --warm_up_n_epochs=0 --batch_size 8 --proximal_mu 1 >${gid}.log 2>&1 </dev/null &"
sleep 10
# shellcheck disable=SC2029
ssh platano "nohup ${client_command} --data-path data/UCSDped2 --wandb_name=platano >${gid}.log 2>&1 </dev/null &"
# shellcheck disable=SC2029
ssh almogrote "nohup ${client_command} --data-path data/UCSDped1 --wandb_name=almogrote >${gid}.log 2>&1 </dev/null &"

# ffmpeg -http_persistent 0 -protocol_whitelist file,http,https,tcp,tls,crypto -i url.m3u8 -c copy video.mp4
