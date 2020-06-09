#!/usr/bin/env bash
source config_hp_train.sh
DATASET_CONFIG=$CONFIG_DIR/dataset_configs/dataset_config.py
MODEL_CONFIG=$CONFIG_DIR/model_configs/hp_ridge_regression.py
CMD="python bin/train.py $VARIABLE_CONFIG $MODEL_CONFIG $DATASET_CONFIG $TRAINING_CONFIG $EXPERIMENT_DIR  $@"
echo $CMD
$CMD