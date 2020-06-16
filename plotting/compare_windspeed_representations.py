import argparse
from csv import DictReader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np
sns.set(color_codes=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('performance_files', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()

    performance = []
    for f in args.performance_files:
        with open(f) as fp:
            performance.extend(list(DictReader(fp)))

    model_weather_variable_performance = defaultdict(lambda: defaultdict(list))
    for perf in performance:
        weather_vars = frozenset(eval(perf['weather_variables']))
        nwp_model = perf['nwp_model']
        try:
            mean_absolute_error = float(perf['mean_absolute_error'])
            r2 = float(perf['r_squared'])

            model_weather_variable_performance[nwp_model][weather_vars].append((mean_absolute_error, r2))
        except ValueError:
            continue


    cartesian_performance = defaultdict(list)
    polar_performance = defaultdict(list)

    for nwp_model, var_performance in model_weather_variable_performance.items():
        for vars, performances in var_performance.items():
            if 'phi' in vars:
                cartesian_performance[nwp_model].extend(performances)
            else:
                polar_performance[nwp_model].extend(performances)

    models = set()
    models.update(cartesian_performance.keys())
    models.update(polar_performance.keys())

    # fig, axes = plt.subplots(len(models), 2, sharex='all', sharey='all')
    # for model, (ax_cart, ax_pol) in zip(sorted(models), axes):
    #     cartesian_perf = cartesian_performance[model]
    #     mae, r2 = zip(*cartesian_perf)
    #     mae = np.array(mae)
    #     r2 = np.array(r2)
    #     #sns.kdeplot(np.array(mae), np.array(r2), ax=ax_cart)
    #     sns.jointplot(x=mae, y=r2)
    #     ax_cart.set_xlabel('Mean Absolute Error')
    #     ax_cart.set_ylabel('$R^2$')
    #     ax_cart.set_title('Cartesian coordinates')
    #
    #     polar_perf = polar_performance[model]
    #     mae, r2 = zip(*polar_perf)
    #     mae = np.array(mae)
    #     r2 = np.array(r2)
    #     #sns.kdeplot(np.array(mae), np.array(r2), ax=ax_pol)
    #     sns.jointplot(x=mae, y=r2 )
    #     ax_pol.set_xlabel('Mean Absolute Error')
    #     ax_pol.set_ylabel('$R^2$')
    #     ax_pol.set_title('Polar coordinates')

    x_label = 'Mean Absolute Error'
    y_label = '$R^2$'

    fig, axes = plt.subplots(len(models), 1, sharex='all')

    for model, ax in zip(sorted(models), axes.flatten()):
        cartesian_perf = cartesian_performance[model]
        c_mae, c_r2 = zip(*cartesian_perf)
        c_mae = np.array(c_mae)
        c_r2 = np.array(c_r2)
        cart_df = pd.DataFrame({x_label: c_mae, y_label: c_r2})
        polar_perf = polar_performance[model]
        p_mae, p_r2 = zip(*polar_perf)
        p_mae = np.array(p_mae)
        p_r2 = np.array(p_r2)
        polar_df = pd.DataFrame({x_label: p_mae, y_label: p_r2})
        x_min = min(c_mae.min(), p_mae.min()) * 1.1
        x_max = min(c_mae.max(), p_mae.max()) * 1.1
        y_min = min(c_r2.min(), p_r2.min()) * 1.1
        y_max = min(c_r2.max(), p_r2.max()) * 1.1
        x_max += (x_max - x_min) * 0.05
        x_min -= (x_max - x_min) * 0.05
        y_max += (y_max - y_min) * 0.05
        y_min -= (y_max - y_min) * 0.05

        # sns.kdeplot(np.array(mae), np.array(r2), ax=ax_cart)
        sns.kdeplot(c_mae, label='Cartesian coordinates', ax=ax, shade=True)
        sns.kdeplot(p_mae, label='Polar coordinates', ax=ax, shade=True)
        ax.set_title(model)
        if args.output_dir is not None:
            save_path = args.output_dir / f'coordinates_{model}.png'
            plt.savefig(save_path)

    plt.suptitle('KDE plots of wind speed representation effect for different NWP providers')
    plt.xlabel('Mean squared error')

    plt.show()

    #
    # for model in sorted(models):
    #     cartesian_perf = cartesian_performance[model]
    #     c_mae, c_r2 = zip(*cartesian_perf)
    #     c_mae = np.array(c_mae)
    #     c_r2 = np.array(c_r2)
    #     cart_df = pd.DataFrame({x_label: c_mae, y_label: c_r2})
    #     polar_perf = polar_performance[model]
    #     p_mae, p_r2 = zip(*polar_perf)
    #     p_mae = np.array(p_mae)
    #     p_r2 = np.array(p_r2)
    #     polar_df = pd.DataFrame({x_label: p_mae, y_label: p_r2})
    #     x_min = min(c_mae.min(), p_mae.min()) * 1.1
    #     x_max = min(c_mae.max(), p_mae.max()) * 1.1
    #     y_min = min(c_r2.min(), p_r2.min()) * 1.1
    #     y_max = min(c_r2.max(), p_r2.max()) * 1.1
    #     x_max += (x_max - x_min) * 0.05
    #     x_min -= (x_max - x_min) * 0.05
    #     y_max += (y_max - y_min) * 0.05
    #     y_min -= (y_max - y_min) * 0.05
    #
    #     #sns.kdeplot(np.array(mae), np.array(r2), ax=ax_cart)
    #     joint = sns.jointplot(x=x_label, y=y_label, data=cart_df, alpha=.7, s=15)
    #     plt.subplots_adjust(top=0.92,
    #                         bottom=0.111,
    #                         left=0.136,
    #                         right=0.975,
    #                         hspace=0.2,
    #                         wspace=0.2)
    #     plt.suptitle(f'{model}, Cartesian Coordinates')
    #     joint.ax_joint.set_xlim(x_min, x_max)
    #     joint.ax_joint.set_ylim(y_min, y_max)
    #     if args.output_dir is not None:
    #         save_path = args.output_dir / f'cartesian_coords_{model}.png'
    #         plt.savefig(save_path)
    #     # sns.kdeplot(np.array(mae), np.array(r2), ax=ax_cart)
    #     joint = sns.jointplot(x=x_label, y=y_label, data=polar_df, alpha=.7, s=15)
    #     plt.subplots_adjust(top=0.92,
    #                         bottom=0.111,
    #                         left=0.136,
    #                         right=0.975,
    #                         hspace=0.2,
    #                         wspace=0.2)
    #     plt.suptitle(f'{model}, Polar Coordinates')
    #     joint.ax_joint.set_xlim(x_min, x_max)
    #     joint.ax_joint.set_ylim(y_min, y_max)
    #     if args.output_dir is not None:
    #         save_path = args.output_dir / f'polar_coords_{model}.png'
    #         plt.savefig(save_path)
    # plt.show()




if __name__ == '__main__':
    main()