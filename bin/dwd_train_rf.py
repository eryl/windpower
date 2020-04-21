import argparse
import csv
import json
import pickle
from collections import defaultdict, deque

from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
np.seterr(all='warn')

from vindel.dwd_dataset import SiteDataset

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics

from mltrain.train import DiscreteHyperParameter, HyperParameterTrainer, train, BaseModel, LowerIsBetterMetric, HigherIsBetterMetric


class DatasetWrapper(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        yield self.dataset

    def __len__(self):
        return len(self.dataset)


class EnsembleModelWrapper(BaseModel):
    def __init__(self, *args, model, **kwargs):
        self.model = model(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def fit(self, batch):
        x = batch['x']
        y = batch['y']
        self.model.fit(x, y)

    def evaluate(self, batch):
        x = batch['x']
        y = batch['y']
        y_hats = self.model.predict(x)

        mse = sklearn.metrics.mean_squared_error(y, y_hats)
        rmse = np.sqrt(mse)
        mae = sklearn.metrics.mean_absolute_error(y, y_hats)
        mad = sklearn.metrics.median_absolute_error(y, y_hats)
        r_squared = sklearn.metrics.r2_score(y, y_hats)
        return {'mean_squared_error': mse,
                'root_mean_squared_error': rmse,
                'mean_absolute_error': mae,
                'median_absolute_deviance': mad,
                'r_squared': r_squared}

    def get_metadata(self):
        return dict(model=self.model.__class__.__name__, args=self.args, kwargs=self.kwargs)

    def evaluation_metrics(self):
        mse_metric = LowerIsBetterMetric('mean_squared_error')
        rmse_metric = LowerIsBetterMetric('root_mean_squared_error')
        mae_metric = LowerIsBetterMetric('mean_absolute_error')
        mad_metric = LowerIsBetterMetric('median_absolute_deviance')
        r_squared_metric = HigherIsBetterMetric('r_squared')
        return [mse_metric, mae_metric, rmse_metric, mad_metric, r_squared_metric]

    def save(self, save_path):
        with open(save_path, 'wb') as fp:
            pickle.dump(self.model, fp)

def train_on_site(datasets, experiment_dir: Path, n_estimator_values, max_depth_values, save_models=True):
    best_model = None
    best_mse = np.inf

    try:
        train_input_data = datasets['train_x']
        train_positions = datasets['train_pos']
        train_positions = np.array(train_positions)[:, np.newaxis]
        # one_hots = np.array(to_one_hot(positions))
        train_input_data = np.concatenate([train_input_data, train_positions], axis=-1)
        dev_input_data = datasets['dev_x']
        dev_positions = datasets['dev_pos']
        dev_positions = np.array(dev_positions)[:, np.newaxis]
        # dev_one_hots = np.array(to_one_hot(dev_positions))
        dev_input_data = np.concatenate([dev_input_data, dev_positions], axis=-1)
        test_input_data = datasets['test_x']
        test_positions = datasets['test_pos']
        test_positions = np.array(test_positions)[:, np.newaxis]
        test_targets = datasets['test_y']
        # test_one_hots = np.array(to_one_hot(test_positions))
        test_input_data = np.concatenate([test_input_data, test_positions], axis=-1)

        with open(experiment_dir / 'training_results.csv', 'w') as fp:
            csv_writer = csv.DictWriter(fp, fieldnames=['n_estimators', 'max_depth', 'mse', 'rmse', 'r_squared'])
            csv_writer.writeheader()

            for n_estimators in tqdm(n_estimator_values):
                for max_depth in tqdm(max_depth_values):
                    model = sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                                   n_jobs=-1)
                    model.fit(train_input_data, datasets['train_y'])

                    y_hat = model.predict(dev_input_data)
                    mse = sklearn.metrics.mean_squared_error(datasets['dev_y'], y_hat)
                    if mse < best_mse:
                        best_model = model
                        best_mse = mse
                    r_squared = model.score(dev_input_data, datasets['dev_y'])
                    csv_writer.writerow(dict(n_estimators=n_estimators, max_depth=max_depth, mse=mse, rmse=np.sqrt(mse), r_squared=r_squared))
                    fp.flush()

        if best_model is not None:
            if save_models:
                with open(experiment_dir / 'best_model.pkl', 'wb') as fp:
                    pickle.dump(best_model, fp)

            y_hats = best_model.predict(test_input_data)

            pos_predictions = defaultdict(list)
            for i, pos in enumerate(test_positions):
                pos_predictions[int(pos)].append((y_hats[i], test_targets[i]))

            with open(experiment_dir / 'test_results.csv', 'w') as fp:
                csv_writer = csv.DictWriter(fp, fieldnames=['pos', 'n_train', 'n_dev', 'n_test', 'mse', 'rmse', 'mae', 'mad', 'r_squared'])
                csv_writer.writeheader()
                for pos, predictions in pos_predictions.items():
                    ys, y_hats = zip(*predictions)
                    mse = sklearn.metrics.mean_squared_error(ys, y_hats)
                    rmse = np.sqrt(mse)
                    mae = sklearn.metrics.mean_absolute_error(ys, y_hats)
                    mad = sklearn.metrics.median_absolute_error(ys, y_hats)
                    r_squared = sklearn.metrics.r2_score(ys, y_hats)
                    csv_writer.writerow(dict(pos=pos, mse=mse, rmse=rmse,
                                             mae=mae, mad=mad, r_squared=r_squared, n_train=len(train_input_data),
                                             n_dev=len(dev_input_data),
                                             n_test=len(test_input_data)))
    except ValueError as e:
        print("Error {}".format(e))
        return


def timestamp():
    """
    Generates a timestamp.
    :return:
    """
    import datetime
    t = datetime.datetime.now().replace(microsecond=0)
    #Since the timestamp is usually used in filenames, isoformat will be invalid in windows.
    #return t.isoformat()
    # We'll use another symbol instead of the colon in the ISO format
    # YYYY-MM-DDTHH:MM:SS -> YYYY-MM-DDTHH.MM.SS
    time_format = "%Y-%m-%dT%H.%M.%S"
    return t.strftime(time_format)

def main():
    parser = argparse.ArgumentParser(description='Train random forrest on sites')
    parser.add_argument('site_files', help="NetCDF files to use", nargs='+', type=Path)
    parser.add_argument('experiment_dir', help="Directory to output results to", type=Path)
    parser.add_argument('--window-length', help="Length of windows to consider", type=int, default=7)
    parser.add_argument('--target-lag',
                        help="Lag between last window element and target. 0 means that the target will "
                             "be the same lead time as the last window element, window_length - 1 will "
                             "be the first window element. Window_length//2 is the middle window element.",
                        type=int, default=3)
    parser.add_argument('--weather-variables', help="What NWP variables to predict on",
                        nargs='+', default=('T', 'U', 'V', 'phi', 'r'))
    parser.add_argument('--include-site-id', help="If true, site id will be a feature", action='store_true')
    parser.add_argument('--include-lead-time', help="If true, lead time will be added as a feature",
                        action='store_true')
    parser.add_argument('--include-time-of-day', help="If true, time of day of window start will be added as a feature",
                        action='store_true')
    parser.add_argument('--n-estimator-values', help="The values to check for n_estimators", nargs='+',
                        type=int, default=[20, 50, 100, 200, 300])
    parser.add_argument('--max-depth-values', help="Values to consider for max_depth", nargs='+',
                        type=int, default=[None, 2, 5, 10, 14, 17, 20, 30, 50])
    parser.add_argument('--do-not-save', help="If given, models will not be written to disk", action='store_true')
    parser.add_argument('--outer-folds', help="Do this many number of folds in the outer cross validation", type=int,
                        default=10)
    parser.add_argument('--outer-xval-loops', help="If set, at most do this many loops of the outer cross validation, "
                                                   "regardless of how many folds there are", type=int)
    parser.add_argument('--inner-folds', help="Do this many number of folds in the inner cross validation", type=int,
                        default=10)
    parser.add_argument('--inner-xval-loops', help="If set, at most do this many loops of the inner cross validation, "
                                                   "regardless of how many folds there are", type=int)

    parser.add_argument('--hp-search-iterations', help="Do this many iterations of hyper parameter search per setting",
                        type=int,
                        default=5)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir / timestamp()
    experiment_dir.mkdir(parents=True)
    metadata = dict(window_length=args.window_length,
                    target_lag=args.target_lag,
                    n_estimator_values=args.n_estimator_values,
                    max_depth_values=args.max_depth_values,
                    weather_variables=args.weather_variables,
                    include_lead_time=args.include_lead_time,
                    include_time_of_day=args.include_time_of_day
                    )
    with open(experiment_dir / 'metadata.json', 'w') as fp:
        json.dump(metadata, fp)

    for site_dataset_path in tqdm(args.site_files):
        site_dataset = SiteDataset(dataset_path=site_dataset_path,
                                   window_length=args.window_length,
                                   production_offset=args.target_lag,
                                   weather_variables=args.weather_variables,
                                   include_lead_time=args.include_lead_time,
                                   include_time_of_day=args.include_time_of_day,
                                   use_cache=True)
        site_id = site_dataset.get_id()
        site_dir = experiment_dir / site_id
        site_dir.mkdir()

        base_model = EnsembleModelWrapper
        base_args = tuple()
        n_estimators = DiscreteHyperParameter(args.n_estimator_values)
        max_depth = DiscreteHyperParameter(args.max_depth_values)
        metadata['site_dataset_path'] = site_dataset_path
        base_kwargs = dict(model=RandomForestRegressor, n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
        train_kwargs = dict(max_epochs=1,  #The random forest regressor doesn't do epochs
                            metadata=metadata,
                            keep_snapshots=False,)

        for i, (test_dataset, train_dataset) in tqdm(enumerate(site_dataset.k_fold_split(args.outer_folds)),
                                                     total=args.outer_folds):
            if args.outer_xval_loops is not None and i >= args.outer_xval_loops:
                break
            fold_dir = site_dir / f'outer_fold_{i:02}'
            fold_dir.mkdir(parents=True)
            test_reference_times = test_dataset.get_reference_times()
            train_reference_time = train_dataset.get_reference_times()
            np.savez(fold_dir / 'fold_reference_times.npz', train=train_reference_time, test=test_reference_times)

            with HyperParameterTrainer(base_model=base_model,
                                       base_args=base_args,
                                       base_kwargs=base_kwargs) as hp_trainer:
                for j, (validation_dataset, fit_dataset) in tqdm(enumerate(train_dataset.k_fold_split(args.inner_folds)),
                                                             total=args.inner_folds):
                    if args.inner_xval_loops is not None and j >= args.inner_xval_loops:
                        break

                    output_dir = fold_dir / f'inner_fold_{j:02}'
                    output_dir.mkdir()
                    fit_reference_times = fit_dataset.get_reference_times()
                    validation_reference_times = validation_dataset.get_reference_times()
                    np.savez(output_dir / 'fold_reference_times.npz', train=fit_reference_times,
                             test=validation_reference_times)
                    fit_dataset = DatasetWrapper(fit_dataset[:])   # This will return the whole dataset as a numpy array
                    validation_dataset = DatasetWrapper(validation_dataset[:])
                    hp_trainer.train(n=args.hp_search_iterations,
                        training_dataset=fit_dataset,
                        evaluation_dataset=validation_dataset,
                        output_dir=output_dir,
                        **train_kwargs)
                best_args, best_kwargs = hp_trainer.get_best_hyper_params()
            model = base_model(*best_args, **best_kwargs)
            train_dataset = DatasetWrapper(train_dataset[:])
            test_dataset = DatasetWrapper(test_dataset[:])
            best_performance, best_model_path = train(model=model,
                                                      training_dataset=train_dataset,
                                                      evaluation_dataset=test_dataset,
                                                      output_dir=fold_dir,
                                                      **train_kwargs)


if __name__ == '__main__':
    main()



