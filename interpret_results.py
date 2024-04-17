import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


ous_raw = pd.read_csv('csv/outcome_ous_interpret_raw.csv')
ous_smoothen = pd.read_csv('csv/outcome_ous_interpret_smoothen.csv')
maastro_raw = pd.read_csv('csv/outcome_maastro_interpret_raw.csv')
maastro_smoothen = pd.read_csv('csv/outcome_maastro_interpret_smoothen.csv')

ous_raw.columns

(ous_raw.ct_max > ous_raw.pt_max).sum() / ous_raw.shape[0]

(ous_smoothen.ct_max > ous_smoothen.pt_max).sum() / ous_smoothen.shape[0]


sns.violinplot(data=ous_raw[['hu_corr', 'suv_corr', 'dfs']].melt(
    ['dfs']), x='value', y='variable', hue='dfs')
plt.show()


sns.violinplot(data=ous_smoothen[['hu_corr', 'suv_corr', 'dfs']].melt(
    ['dfs']), x='value', y='variable', hue='dfs')
plt.show()

sns.violinplot(data=maastro_smoothen[['hu_corr', 'suv_corr', 'dfs']].melt(
    ['dfs']), x='value', y='variable', hue='dfs')
plt.show()

sns.violinplot(data=ous_smoothen.groupby('pid').max()[
               ['hu_corr', 'suv_corr', 'dfs']].melt(['dfs']), x='value', y='variable', hue='dfs')
plt.show()

sns.violinplot(data=ous_raw.groupby('pid').max()[
               ['hu_corr', 'suv_corr', 'dfs']].melt(['dfs']), x='value', y='variable', hue='dfs')
plt.show()


sns.violinplot(data=maastro_smoothen.groupby('pid').max()[
               ['hu_corr', 'suv_corr', 'dfs']].melt(['dfs']), x='value', y='variable', hue='dfs')
plt.show()

sns.violinplot(data=maastro_raw.groupby('pid').max()[
               ['hu_corr', 'suv_corr', 'dfs']].melt(['dfs']), x='value', y='variable', hue='dfs')
plt.show()

sns.violinplot(data=maastro_smoothen.groupby('pid').mean()[
               ['hu_corr', 'suv_corr', 'dfs']].melt(['dfs']), x='value', y='variable', hue='dfs')
plt.show()

sns.violinplot(data=maastro_raw.groupby('pid').mean()[
               ['hu_corr', 'suv_corr', 'dfs']].melt(['dfs']), x='value', y='variable', hue='dfs')
plt.show()

sns.violinplot(data=ous_smoothen[['ct_tumor_sum', 'pt_tumor_sum', 'dfs']].melt(
    ['dfs']), x='value', y='variable', hue='dfs')
plt.show()


(ous_smoothen.hu_corr > 0.25).sum()  # 13 / 556
(ous_smoothen.suv_corr > 0.25).sum()  # 133 / 556

(ous_smoothen.groupby('pid').mean().hu_corr > 0.25).sum()  # 0 / 139
(ous_smoothen.groupby('pid').mean().suv_corr > 0.25).sum()  # 19 / 139

(ous_smoothen.groupby('pid').max().hu_corr > 0.25).sum()  # 13 / 139
(ous_smoothen.groupby('pid').max().suv_corr > 0.25).sum()  # 94 / 139

(ous_smoothen.groupby('pid').median().hu_corr > 0.25).sum()  # 0 / 139
(ous_smoothen.groupby('pid').median().suv_corr > 0.25).sum()  # 16 / 139
