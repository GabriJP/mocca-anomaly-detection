#!/usr/bin/env bash

client_command="python main_shanghaitech2.py --load-lstm --bidirectional --clip-length=16 --code-length=512 --dropout=0.3 --epochs=21 --test-chk 5,10,15,20 --idx-list-enc=3,4,5,6 --wandb_group shang_avo --es_initial_patience_epochs 51 --es_patience 51"

for subset in data/shanghaitech/avo/*
do
  [ -e "$subset" ] || continue
  ${client_command} -dp "$subset" --wandb_name "${subset##*/}" --batch-size 16 >"avo_${subset##*/}.out" 2>&1 </dev/null
done
