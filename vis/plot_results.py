#! /usr/bin/env python
import os
import glob
import json
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(input_dir):
    """Simple function to load all json files in a pd.DataFrame"""
    pattern = os.path.join(input_dir, '**', '*.json')
    files = glob.glob(pattern, recursive=True)
    l = []
    for path in files:
        with open(path, 'r') as f:
            dc = json.load(f)
            dc["instance_type"] = os.path.basename(os.path.dirname(path))
            l.append(dc)

    df = pd.DataFrame(l)
    df['model_name'] = df.model_path.str.split('/').str.get(-1)
    return df


def plot_results(df, output_dir='figs'):
    os.makedirs(output_dir, exist_ok=True)
    groups = df.groupby(['model_name', 'instance_type'])
    for (model_name, instance_type), data in groups:
        data = data.sort_values(by='backend')
        idx = data.backend + '\n' + data.backend_meta
        tmp = data.data.apply(lambda x: pd.Series(x))
        tmp.index = idx
        tmp.sort_index(inplace=True)

        plt.figure(figsize=(10, 5))
        p = sns.barplot(data=[1 / vals.dropna() for col, vals in tmp.T.iteritems()])
        p.set_axisbelow(True)
        p.set_title(f'{model_name}\n{instance_type}')
        p.set_xticklabels(tmp.index)  # , rotation=30)
        p.set_ylabel('Samples per second')
        p.grid(axis='y')
        plt.savefig(f'{output_dir}/{model_name}-{instance_type}.png')


def main(input_dir, output_dir):
    df = load_data(input_dir)
    plot_results(df, output_dir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
