#!/usr/bin/env bash
source config_hp_train.sh
DATASET_CONFIG=$CONFIG_DIR/dataset_configs/dataset_config.py
MODEL_CONFIG=$CONFIG_DIR/model_configs/hp_random_forest.py
VARIABLE_CONFIG=$CONFIG_DIR/variable_configs/non_one_hot_experiment_variables.py
CMD="python bin/train.py $VARIABLE_CONFIG $MODEL_CONFIG $DATASET_CONFIG $TRAINING_CONFIG $EXPERIMENT_DIR  $@"
echo $CMD
$CMD