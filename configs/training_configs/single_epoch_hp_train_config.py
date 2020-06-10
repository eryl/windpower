from windpower.train_ensemble import TrainConfig
import mltrain.train

do_not_save = False
outer_folds = 10
outer_xval_loops = None
inner_folds = 10
inner_xval_loops = None
hp_search_iterations = 5
fold_padding = 42
train_kwargs = mltrain.train.TrainingConfig(max_epochs=1, keep_snapshots=False)

training_config = TrainConfig(outer_folds=outer_folds,
                              outer_xval_loops=outer_xval_loops,
                              inner_folds=inner_folds,
                              inner_xval_loops=inner_xval_loops,
                              hp_search_iterations=hp_search_iterations,
                              fold_padding=fold_padding,
                              train_kwargs=train_kwargs)