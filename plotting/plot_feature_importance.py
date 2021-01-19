import argparse
import math
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

max_colors = 4
max_bars = 100



plot_groups={
        'DWD_ICON-EU':  [['T', 'U', 'V'], ['phi_U_V', 'r_U_V', 'lead_time', 'time_of_day']],
        "FMI_HIRLAM": [["Temperature", "WindUMS", "WindVMS"], ['phi', 'r', 'lead_time', 'time_of_day']],
        "NCEP_GFS": [['WindUMS_Height', 'WindVMS_Height', 'Temperature_Height'],
                     ['phi_WindUMS_Height_WindVMS_Height', 'r_WindUMS_Height_WindVMS_Height', 'lead_time', 'time_of_day']],
        "MetNo_MEPS": [["x_wind_10m", "y_wind_10m", "x_wind_z", "y_wind_z"],
                       ["air_pressure_at_sea_level","air_temperature_0m", "air_temperature_2m", "air_temperature_z"],
                       ['phi_10m', 'r_10m', 'phi_z', 'r_z', 'lead_time', 'time_of_day']],
        "ECMWF_EPS-CF": [["u10",
                         "v10",
                         "u100",
                         "v100",
                         #"u200",
                         #"v200",
                         "i10fg",
                         "t2m"],
                         ['phi_u10_v10',
                         'r_u10_v10',
                         'phi_u100_v100',
                         'r_u100_v100',
                         'lead_time',
                         'time_of_day']],
        "ECMWF_HRES": [["u10",
                       "v10",
                       "u100",
                       "v100",
                       # "u200",
                       # "v200",
                       "i10fg",
                       "t2m"],
                       ['phi_u10_v10',
                       'r_u10_v10',
                       'phi_u100_v100',
                       'r_u100_v100',
                       'lead_time',
                       'time_of_day']],
    'DWD_ICON-EU#ECMWF_EPS-CF#ECMWF_HRES#NCEP_GFS' :  [
        ['DWD_ICON-EU$T',
         'ECMWF_EPS-CF$t2m',
         'ECMWF_HRES$t2m',
         'NCEP_GFS$Temperature_Height',
         ],

         ['DWD_ICON-EU$U',
          'DWD_ICON-EU$V',
         'DWD_ICON-EU$r_U_V',
         'DWD_ICON-EU$phi_U_V',
         ],

        ['ECMWF_EPS-CF$i10fg',
         'ECMWF_EPS-CF$u10',
         'ECMWF_EPS-CF$u100',
         'ECMWF_EPS-CF$v10',
         'ECMWF_EPS-CF$v100',
         'ECMWF_EPS-CF$r_u10_v10',
         'ECMWF_EPS-CF$phi_u10_v10',
         'ECMWF_EPS-CF$r_u100_v100',
         'ECMWF_EPS-CF$phi_u100_v100',
         ],
        ['ECMWF_HRES$i10fg',

         'ECMWF_HRES$u10',
         'ECMWF_HRES$u100',
         'ECMWF_HRES$v10',
         'ECMWF_HRES$v100',
         'ECMWF_HRES$r_u10_v10',
         'ECMWF_HRES$phi_u10_v10',
         'ECMWF_HRES$r_u100_v100',
         'ECMWF_HRES$phi_u100_v100'
         ],
        ['NCEP_GFS$WindUMS_Height',
         'NCEP_GFS$WindVMS_Height',
         'NCEP_GFS$r_WindUMS_Height_WindVMS_Height',
         'NCEP_GFS$phi_WindUMS_Height_WindVMS_Height'],
         ['lead_time',
         'time_of_day'
         ]],
    }


def main():
    parser = argparse.ArgumentParser(description='Plot feature importance')
    parser.add_argument('importance_data', type=Path)
    parser.add_argument('--nwp-model', default='DWD_ICON-EU#ECMWF_EPS-CF#ECMWF_HRES#NCEP_GFS')
    args = parser.parse_args()

    data = pd.read_csv(args.importance_data)
    tick_pairs = []
    tick_labels = []
    start_tick = 0
    end_tick = start_tick
    current_feature_name = None
    for i, row in data.iterrows():
        feature_name = row['name']
        subvar_i = row['feature_subindex']
        feature_index = row['feature_index']

        if current_feature_name is None:
            current_feature_name = feature_name
        elif current_feature_name != feature_name:
            tick_labels.append(current_feature_name)
            tick_pairs.append((start_tick, feature_index))
            current_feature_name = feature_name
            start_tick = feature_index


    tick_labels.append(current_feature_name)
    tick_pairs.append((start_tick, start_tick + 1))


    ticks = [a for a, b in tick_pairs]
    tick_label_locs = []
    for (a,b), label in zip(tick_pairs, tick_labels):
        loc = int(a+b / 2)
        tick_label_locs.append(loc)

    varname_to_index = dict(zip(tick_labels, tick_pairs))

    plot_label = args.nwp_model
    if plot_label == 'DWD_ICON-EU#ECMWF_EPS-CF#ECMWF_HRES#NCEP_GFS':
        plot_label = 'Merged models'

    groups = plot_groups[args.nwp_model]

    gain_ci = data[['gain_ci_low','gain_ci_high']].values
    yerr_gain = gain_ci - data[['gain_mean']].values
    make_plots(data['gain_mean'].values, yerr_gain, groups, varname_to_index, f'Mean gain importance for {plot_label}', 'Mean gain importance')

    split_ci = data[['split_ci_low','split_ci_high']].values
    yerr_split = split_ci - data[['split_mean']].values
    make_plots(data['split_mean'].values, yerr_split, groups, varname_to_index, f'Mean split importance for {plot_label}', 'Mean split importance')
    plt.show()

def make_plots(data, yerr, groups, varname_to_index, plot_label, ylabel):
    nrows = 2
    ncols = int(math.ceil(len(groups) / nrows))
    fig_kwargs = dict(nrows=nrows, ncols=ncols, sharey='all', figsize=(16, 8))
    fig, axes = plt.subplots(**fig_kwargs)

    cmap = plt.get_cmap('tab10')

    for subplot_i, (var_group, ax) in enumerate(zip(groups, axes.flatten())):
        if subplot_i % ncols == 0:
            ax.set_ylabel(ylabel)
        x = 0
        handles = []
        for i, label in enumerate(var_group):
            a, b = varname_to_index[label]
            var_data = data[a:b]
            var_yerr = yerr[a:b]
            color = cmap(i/len(var_group))
            n_features = b - a
            ax.bar(x=np.arange(x,x+n_features), height=var_data, width=1, facecolor=color, label=label, yerr=var_yerr.T)
            x += n_features
            handles.append(mpatches.Patch(color=color, label=label))
        ax.legend(handles=handles)
        ax.get_xaxis().set_visible(False)

    plt.suptitle(plot_label)
    plt.tight_layout()
    plt.subplots_adjust(top=0.946,
    bottom=0.029,
    left=0.054,
    right=0.991,
    hspace=0.074,
    wspace=0.072)


if __name__ == '__main__':
    main()