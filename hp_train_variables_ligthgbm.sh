#!/usr/bin/env bash
source config_hp_train.sh
VARIABLE_CONFIG=$CONFIG_DIR/variable_configs/experiment_variables_hp_search.py
MODEL_CONFIG=$CONFIG_DIR/model_configs/lightgbm.py
EXPERIMENT_DIR=$EXPERIMENT_DIR/hp_variables
CMD="python bin/train.py $VARIABLE_CONFIG $MODEL_CONFIG $DATASET_CONFIG $TRAINING_CONFIG $EXPERIMENT_DIR  $@"
echo $CMD
$CMD