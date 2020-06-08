do_not_save = False
outer_folds = 10
outer_xval_loops = None
inner_folds = 0
inner_xval_loops = None
hp_search_iterations = 1
train_kwargs = dict(max_epochs=1,  # We're not using an iterative model here
                    keep_snapshots=False)
