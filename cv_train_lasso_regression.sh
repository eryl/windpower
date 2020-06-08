#!/usr/bin/env bash
source config_hp_train.sh
DATASET_CONFIG=$CONFIG_DIR/dataset_configs/uncached_dataset_config.py
MODEL_CONFIG=$CONFIG_DIR/model_configs/lasso_regression.py
TRAINING_CONFIG=$CONFIG_DIR/training_configs/single_epoch_single_cv_loop.py
CMD="python bin/train.py $VARIABLE_CONFIG $MODEL_CONFIG $DATASET_CONFIG $TRAINING_CONFIG $EXPERIMENT_DIR  $@"
echo $CMD
$CMD