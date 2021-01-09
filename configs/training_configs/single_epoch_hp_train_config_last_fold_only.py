from windpower.train_ensemble import TrainConfig
import windpower.mltrain.train

do_not_save = False
outer_folds = 10
inner_folds = 10
outer_fold_idx = [9]
inner_fold_idx = [9]
hp_search_iterations = 1
fold_padding = 54
train_kwargs = mltrain.train.TrainingConfig(max_epochs=1, keep_snapshots=False)

training_config = TrainConfig(outer_folds=outer_folds,
                              inner_folds=inner_folds,
                              outer_xval_loop_idxs=outer_fold_idx,
                              inner_xval_loop_idxs=inner_fold_idx,
                              hp_search_iterations=hp_search_iterations,
                              fold_padding=fold_padding,
                              train_kwargs=train_kwargs)