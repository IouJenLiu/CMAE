#!/usr/bin/env bash
cd ..

i=0
EXP_BATCH_NAME=CMAE
EXP=CMAE_ROOM_${i}
ENV=room30
python main.py --env ${ENV} --exp_mode active_cen --multilevel --tree_subspace  --mixed_explore --stochastic_select_subspace  --exp_name ${EXP} --seed ${i} --exp_batch_name ${EXP_BATCH_NAME}