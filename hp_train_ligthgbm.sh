#!/usr/bin/env bash
source config_hp_train.sh
MODEL_CONFIG=$CONFIG_DIR/model_configs/hp_lightgbm.py
CMD="python bin/train.py $VARIABLE_CONFIG $MODEL_CONFIG $DATASET_CONFIG $TRAINING_CONFIG $EXPERIMENT_DIR  $@"
echo $CMD
$CMD