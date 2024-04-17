import os
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# sns.set_style('whitegrid')
plt.figure(figsize=(6, 6))

dfs_ous_smoothen_v2 = pd.read_csv(
    'P:/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/interpret_analyze_results/ous_smoothen_v2.csv')

dfs_ous_actual_res = pd.read_csv(
    'P:/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/interpret_analyze_results/ous_actual_res.csv'
)

dfs_maastro_smoothen_v2 = pd.read_csv(
    'P:/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/interpret_analyze_results/maastro_smoothen_v2.csv')
dfs_maastro_actual_res = pd.read_csv(
    'P:/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/interpret_analyze_results/maastro_actual_res.csv'
)


dfs_actual_res = pd.concat([dfs_ous_actual_res, dfs_maastro_actual_res])
dfs_smoothen_df = pd.concat([dfs_ous_smoothen_v2, dfs_maastro_smoothen_v2])
dfs_smoothen_df = dfs_smoothen_df.merge(
    dfs_actual_res, 'left', on=['pid', 'center', 'val_fold', 'test_fold'])
correct = dfs_smoothen_df.y == (dfs_smoothen_df.predicted > 0.5).astype(int)
dfs_smoothen_df['correct'] = correct
confusion_matrix = np.empty(len(correct), dtype=object)
pdfsitive_y = dfs_smoothen_df.y == 1
confusion_matrix[pdfsitive_y & correct] = 'TP'
confusion_matrix[(~pdfsitive_y) & correct] = 'TN'
confusion_matrix[pdfsitive_y & (~correct)] = 'FN'
confusion_matrix[(~pdfsitive_y) & (~correct)] = 'FP'
dfs_smoothen_df['confusion_matrix'] = confusion_matrix

# region Check SUV histogram
suv_regions = ['zeros', '0_2', '2_4', '4_6', '6_8', '8_10', '10_over']
suv_df = dfs_ous_smoothen_v2[dfs_ous_smoothen_v2['quantile'] == 0.95][[
    f'all_suv_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

sum_suv_df = suv_df.sum()
hist_vals = []
for suv in suv_regions:
    hist_vals.append(
        sum_suv_df[f'all_suv_{suv}_sum'] / sum_suv_df[f'all_suv_{suv}_area'])

maastro_suv_df = dfs_maastro_smoothen_v2[dfs_maastro_smoothen_v2['quantile'] == 0.95][[
    f'all_suv_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

maastro_sum_suv_df = maastro_suv_df.sum()
maastro_hist_vals = []
for suv in suv_regions:
    maastro_hist_vals.append(
        maastro_sum_suv_df[f'all_suv_{suv}_sum'] / maastro_sum_suv_df[f'all_suv_{suv}_area'])

plt.subplot(3, 2, 1)
ax = sns.lineplot(x=suv_regions, y=hist_vals,
                  marker='o', lw=3, label='OUS')
ax = sns.lineplot(x=suv_regions, y=maastro_hist_vals,
                  marker='o', lw=3, label='MAASTRO')
ax.set_xticklabels([0, 2, 4, 6, 8, 10, '>10'])
ax.set_xlabel('SUV')
ax.set_ylabel('Mean VarGrad')
ax.set_title('DFS M5 (PET/CT)', loc='left')
ax.set_ylim(0., 0.4)
# plt.show()

# endregion SUV histogram

# region Check HU histogram
suv_regions = ['zero', '0_16', '16_32', '32_48', '48_64',
               '64_80', '80_96', '96_112', '112_128',
               '128_144', '144_160', '160_176', '176_over']
suv_df = dfs_ous_smoothen_v2[dfs_ous_smoothen_v2['quantile'] == 0.95][[
    f'all_hu_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

sum_suv_df = suv_df.sum()
hist_vals = []
for suv in suv_regions:
    hist_vals.append(
        sum_suv_df[f'all_hu_{suv}_sum'] / sum_suv_df[f'all_hu_{suv}_area'])

maastro_suv_df = dfs_maastro_smoothen_v2[dfs_maastro_smoothen_v2['quantile'] == 0.95][[
    f'all_hu_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

maastro_sum_suv_df = maastro_suv_df.sum()
maastro_hist_vals = []
for suv in suv_regions:
    maastro_hist_vals.append(
        maastro_sum_suv_df[f'all_hu_{suv}_sum'] / maastro_sum_suv_df[f'all_hu_{suv}_area'])

plt.subplot(3, 2, 3)
ax = sns.lineplot(x=suv_regions, y=hist_vals,
                  marker='o', lw=3, label='OUS')
ax = sns.lineplot(x=suv_regions, y=maastro_hist_vals,
                  marker='o', lw=3, label='MAASTRO')
# ax.set_xticklabels([-30, -14, 2, 18, 34, 50, 66, 82, 98, 114, 130, 146, 162])
ax.set_xticklabels([-30, None, 2, None, 34, None, 66,
                    None, 98, None, 130, None, 162])
ax.set_xlabel('HU')
ax.set_ylabel('Mean VarGrad')
# ax.set_title('DFS M5', loc='left')
ax.set_ylim(0., 0.4)
# plt.show()

# endregion HU histogram


# region Check vargrad on tumor + node
area_df = dfs_smoothen_df[dfs_smoothen_df['quantile'] == 0.95][['center', 'tumor_size', 'node_size', 'ct_tumor_all_sum',
                                                                'pt_tumor_all_sum', 'ct_node_all_sum', 'pt_node_all_sum', 'ct_outside_all_sum', 'pt_outside_all_sum']]

area_df['tumor_sum'] = (area_df['ct_tumor_all_sum'] +
                        area_df['pt_tumor_all_sum']) / 2
area_df['node_sum'] = (area_df['ct_node_all_sum'] +
                       area_df['pt_node_all_sum']) / 2
area_df['outside_sum'] = (
    area_df['ct_outside_all_sum'] + area_df['pt_outside_all_sum']) / 2
area_df['outside_size'] = 191*173*265 - \
    area_df['tumor_size'] - area_df['node_size']


area_df_cal = area_df[['center', 'tumor_sum', 'tumor_size', 'node_sum',
                       'node_size', 'outside_sum', 'outside_size']].groupby('center').sum().reset_index()

area_df_cal['Tumor'] = area_df_cal['tumor_sum'] / \
    area_df_cal['tumor_size']
area_df_cal['Node'] = area_df_cal['node_sum'] / area_df_cal['node_size']
area_df_cal['Others'] = area_df_cal['outside_sum'] / \
    area_df_cal['outside_size']

plt.subplot(3, 2, 5)
ax = sns.barplot(area_df_cal[['center', 'Tumor', 'Node', 'Others']].melt(
    'center'), x='variable', y='value', hue='center', order=['Tumor', 'Node', 'Others'], hue_order=['OUS', 'MAASTRO'])
ax.set_xlabel('Area')
ax.set_ylabel('Mean VarGrad')
ax.set_ylim(0., 0.4)
sns.move_legend(ax, 'upper right', title='Center',)
# plt.show()

# endregion


os_ous_smoothen_v2 = pd.read_csv(
    'P:/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/interpret_analyze_results/os_ous_smoothen_v2.csv')

os_ous_actual_res = pd.read_csv(
    'P:/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/interpret_analyze_results/os_ous_actual_res.csv'
)

os_maastro_smoothen_v2 = pd.read_csv(
    'P:/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/interpret_analyze_results/os_maastro_smoothen_v2.csv')
os_maastro_actual_res = pd.read_csv(
    'P:/REALTEK-HeadNeck-Project/Head-and-Neck/PhDs/Ngoc/Orion/interpret_analyze_results/os_maastro_actual_res.csv'
)


os_actual_res = pd.concat([os_ous_actual_res, os_maastro_actual_res])
os_smoothen_df = pd.concat([os_ous_smoothen_v2, os_maastro_smoothen_v2])
os_smoothen_df = os_smoothen_df.merge(
    os_actual_res, 'left', on=['pid', 'center', 'val_fold', 'test_fold'])
correct = os_smoothen_df.y == (os_smoothen_df.predicted > 0.5).astype(int)
os_smoothen_df['correct'] = correct
confusion_matrix = np.empty(len(correct), dtype=object)
positive_y = os_smoothen_df.y == 1
confusion_matrix[positive_y & correct] = 'TP'
confusion_matrix[(~positive_y) & correct] = 'TN'
confusion_matrix[positive_y & (~correct)] = 'FN'
confusion_matrix[(~positive_y) & (~correct)] = 'FP'
os_smoothen_df['confusion_matrix'] = confusion_matrix

# region Check SUV histogram
suv_regions = ['zeros', '0_2', '2_4', '4_6', '6_8', '8_10', '10_over']
suv_df = os_ous_smoothen_v2[os_ous_smoothen_v2['quantile'] == 0.95][[
    f'all_suv_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

sum_suv_df = suv_df.sum()
hist_vals = []
for suv in suv_regions:
    hist_vals.append(
        sum_suv_df[f'all_suv_{suv}_sum'] / sum_suv_df[f'all_suv_{suv}_area'])

maastro_suv_df = os_maastro_smoothen_v2[os_maastro_smoothen_v2['quantile'] == 0.95][[
    f'all_suv_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

maastro_sum_suv_df = maastro_suv_df.sum()
maastro_hist_vals = []
for suv in suv_regions:
    maastro_hist_vals.append(
        maastro_sum_suv_df[f'all_suv_{suv}_sum'] / maastro_sum_suv_df[f'all_suv_{suv}_area'])

plt.subplot(3, 2, 2)
ax = sns.lineplot(x=suv_regions, y=hist_vals,
                  marker='o', lw=3, label='OUS')
ax = sns.lineplot(x=suv_regions, y=maastro_hist_vals,
                  marker='o', lw=3, label='MAASTRO')
ax.set_xticklabels([0, 2, 4, 6, 8, 10, '>10'])
ax.set_xlabel('SUV')
ax.set_ylabel('Mean VarGrad')
ax.set_ylim(0., 0.4)
ax.set_title('OS M6 (PET/CT + GTVp)', loc='left')
# plt.show()

# endregion SUV histogram

# region Check HU histogram
suv_regions = ['zero', '0_16', '16_32', '32_48', '48_64',
               '64_80', '80_96', '96_112', '112_128',
               '128_144', '144_160', '160_176', '176_over']
suv_df = os_ous_smoothen_v2[os_ous_smoothen_v2['quantile'] == 0.95][[
    f'all_hu_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

sum_suv_df = suv_df.sum()
hist_vals = []
for suv in suv_regions:
    hist_vals.append(
        sum_suv_df[f'all_hu_{suv}_sum'] / sum_suv_df[f'all_hu_{suv}_area'])

maastro_suv_df = os_maastro_smoothen_v2[os_maastro_smoothen_v2['quantile'] == 0.95][[
    f'all_hu_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

maastro_sum_suv_df = maastro_suv_df.sum()
maastro_hist_vals = []
for suv in suv_regions:
    maastro_hist_vals.append(
        maastro_sum_suv_df[f'all_hu_{suv}_sum'] / maastro_sum_suv_df[f'all_hu_{suv}_area'])

plt.subplot(3, 2, 4)
ax = sns.lineplot(x=suv_regions, y=hist_vals,
                  marker='o', lw=3, label='OUS')
ax = sns.lineplot(x=suv_regions, y=maastro_hist_vals,
                  marker='o', lw=3, label='MAASTRO')
# ax.set_xticklabels([-30, -14, 2, 18, 34, 50, 66, 82, 98, 114, 130, 146, 162])
ax.set_xticklabels([-30, None, 2, None, 34, None, 66,
                    None, 98, None, 130, None, 162])
ax.set_xlabel('HU')
ax.set_ylabel('Mean VarGrad')
# ax.set_title('DFS M5', loc='left')
ax.set_ylim(0., 0.4)
# plt.show()

# endregion SUV histogram


# region Check vargrad on tumor + node
area_df = os_smoothen_df[os_smoothen_df['quantile'] == 0.95][['center', 'tumor_size', 'node_size', 'ct_tumor_all_sum',
                                                              'pt_tumor_all_sum', 'ct_node_all_sum', 'pt_node_all_sum', 'ct_outside_all_sum', 'pt_outside_all_sum']]

area_df['tumor_sum'] = (area_df['ct_tumor_all_sum'] +
                        area_df['pt_tumor_all_sum']) / 2
area_df['node_sum'] = (area_df['ct_node_all_sum'] +
                       area_df['pt_node_all_sum']) / 2
area_df['outside_sum'] = (
    area_df['ct_outside_all_sum'] + area_df['pt_outside_all_sum']) / 2
area_df['outside_size'] = 191*173*265 - \
    area_df['tumor_size'] - area_df['node_size']


area_df_cal = area_df[['center', 'tumor_sum', 'tumor_size', 'node_sum',
                       'node_size', 'outside_sum', 'outside_size']].groupby('center').sum().reset_index()

area_df_cal['Tumor'] = area_df_cal['tumor_sum'] / \
    area_df_cal['tumor_size']
area_df_cal['Node'] = area_df_cal['node_sum'] / area_df_cal['node_size']
area_df_cal['Others'] = area_df_cal['outside_sum'] / \
    area_df_cal['outside_size']

plt.subplot(3, 2, 6)
ax = sns.barplot(area_df_cal[['center', 'Tumor', 'Node', 'Others']].melt(
    'center'), x='variable', y='value', hue='center', order=['Tumor', 'Node', 'Others'], hue_order=['OUS', 'MAASTRO'])
ax.set_xlabel('Area')
ax.set_ylabel('Mean VarGrad')
ax.set_ylim(0., 0.4)
sns.move_legend(ax, 'upper right', title='Center',)
# plt.show()

# endregion
plt.tight_layout()

plt.savefig('outcome_paper_figures/vargrad.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures/vargrad.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures/vargrad.tiff',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
