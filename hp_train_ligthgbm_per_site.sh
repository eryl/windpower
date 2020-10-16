#!/usr/bin/env bash
EXPERIMENT_DIR=/data/models/vindel/hp_lightgbm_per_site
CONFIG_DIR=configs
DATASET_CONFIG=$CONFIG_DIR/dataset_configs/dataset_config.py
TRAINING_CONFIG=$CONFIG_DIR/training_configs/single_epoch_hp_train_config_single_iter.py
VARIABLE_CONFIG=$CONFIG_DIR/variable_configs/non_one_hot_experiment_variables.py
MODEL_CONFIG=$CONFIG_DIR/model_configs/lightgbm.py
CMD="python scripts/train.py $VARIABLE_CONFIG $MODEL_CONFIG $DATASET_CONFIG $TRAINING_CONFIG $EXPERIMENT_DIR  $@"
echo $CMD
$CMD
