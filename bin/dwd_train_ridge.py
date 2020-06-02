import argparse
from pathlib import Path
import numpy as np
np.seterr(all='warn')

from windpower.train_ensemble import train
from sklearn.linear_model import Ridge
from mltrain.train import DiscreteHyperParameter, LowerIsBetterMetric, HigherIsBetterMetric, BaseModel
from windpower.dataset import Variable, CategoricalVariable, DiscretizedVariableEvenBins

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
    parser.add_argument('--horizon',
                        help="What horizon to forecast over",
                        type=int, default=30)
    parser.add_argument('--weather-variables', help="What NWP variables to predict on",
                        nargs='+', default=('T', 'U', 'V', 'phi', 'r'))
    parser.add_argument('--include-site-id', help="If true, site id will be a feature", action='store_true')
    parser.add_argument('--include-lead-time', help="If true, lead time will be added as a feature",
                        action='store_true')
    parser.add_argument('--include-time-of-day', help="If true, time of day of window start will be added as a feature",
                        action='store_true')
    parser.add_argument('--one-hot-encode', help="If true, categorical variables will be one-hot encoded, "
                                                 "if false they will be integer encoded",
                        action='store_true')
    parser.add_argument('--do-not-save', help="If given, models will not be written to disk", action='store_true')
    parser.add_argument('--outer-folds', help="Do this many number of folds in the outer cross validation", type=int,
                        default=10)
    parser.add_argument('--outer-xval-loops', help="If set, at most do this many loops of the outer cross validation, "
                                                   "regardless of how many folds there are", type=int)
    parser.add_argument('--inner-folds', help="Do this many number of folds in the inner cross validation", type=int,
                        default=0)
    parser.add_argument('--inner-xval-loops', help="If set, at most do this many loops of the inner cross validation, "
                                                   "regardless of how many folds there are", type=int)

    parser.add_argument('--hp-search-iterations', help="Do this many iterations of hyper parameter search per setting",
                        type=int,
                        default=1)
    args = parser.parse_args()

    variable_definitions = {'T': Variable('T'),
                            'U': Variable('U'),
                            'V': Variable('V'),
                            'phi': DiscretizedVariableEvenBins('phi', (-np.pi, np.pi), 64,
                                                               one_hot_encode=args.one_hot_encode),
                            'r': Variable('r'),
                            'site_production': Variable('site_production'),
                            }

    metadata = dict(window_length=args.window_length,
                    target_lag=args.target_lag,
                    horizon=args.horizon,
                    weather_variables=args.weather_variables,
                    #variable_definitions=variable_definitions,
                    include_lead_time=args.include_lead_time,
                    include_time_of_day=args.include_time_of_day
                    )

    base_args = tuple()
    #n_estimators = DiscreteHyperParameter(args.n_estimator_values)
    #max_depth = DiscreteHyperParameter(args.max_depth_values)

    base_kwargs = dict(model=Ridge, scaling=True)
    train_kwargs = dict(max_epochs=1,  # The random forest regressor doesn't do epochs
                        metadata=metadata,
                        keep_snapshots=False, )

    train(site_files=args.site_files, experiment_dir=args.experiment_dir, window_length=args.window_length,
          target_lag=args.target_lag, weather_variables=args.weather_variables,
          horizon=args.horizon,
          include_lead_time=args.include_lead_time, include_time_of_day=args.include_time_of_day, one_hot_encode=args.one_hot_encode,
          outer_folds=args.outer_folds, outer_xval_loops=args.outer_xval_loops,
          inner_folds=args.inner_folds, inner_xval_loops=args.inner_xval_loops,
          hp_search_iterations=args.hp_search_iterations, metadata=metadata, train_kwargs=train_kwargs,
          base_args=base_args, base_kwargs=base_kwargs, variable_definitions=variable_definitions)


if __name__ == '__main__':
    main()



