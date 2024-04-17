from matplotlib.patches import ConnectionPatch
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style('whitegrid')


DFS_df = pd.concat(
    [
        pd.read_csv('outcome_analysis_res_reviewer/DFS_solo.csv'),
        pd.read_csv('outcome_analysis_res_reviewer/DFS_solo_maastro.csv'),
        pd.read_csv('outcome_analysis_res_reviewer/DFS_combine.csv'),
        pd.read_csv('outcome_analysis_res_reviewer/DFS_combine_maastro.csv')
    ]
)
DFS_df['center'] = np.concatenate(
    [
        ['OUS']*63,
        ['MAASTRO']*63,
        ['OUS']*180,
        ['MAASTRO']*180,
    ]
)

OS_df = pd.concat(
    [
        pd.read_csv('outcome_analysis_res_reviewer/OS_solo.csv'),
        pd.read_csv('outcome_analysis_res_reviewer/OS_solo_maastro.csv'),
        pd.read_csv('outcome_analysis_res_reviewer/OS_combine.csv'),
        pd.read_csv('outcome_analysis_res_reviewer/OS_combine_maastro.csv')
    ]
)
OS_df['center'] = np.concatenate(
    [
        ['OUS']*63,
        ['MAASTRO']*63,
        ['OUS']*180,
        ['MAASTRO']*180,
    ]
)

# update naming convention


def update_base_name(name):
    if name.endswith('CT_PET'):
        return 'M5'
    elif name.endswith('CT_PET_T'):
        return 'M6'
    elif name.endswith('CT_PET_T_N'):
        return 'M7'
    elif 'Logistic' in name:
        return 'M1'
    elif 'RandomForest' in name:
        return 'M2'
    elif 'basic_nn' in name:
        return 'M3'
    elif 'interaction_nn' in name:
        return 'M4'


def update_image_type_name(name):
    if name.endswith('CT_PET'):
        return 'M5'
    elif name.endswith('CT_PET_T'):
        return 'M6'
    elif name.endswith('CT_PET_T_N'):
        return 'M7'


def update_type(name):
    if 'clinical' in name:
        return 'D1'
    elif 'radiomics' in name:
        return 'D2'
    elif 'tabular' in name:
        return 'D1+D2'
    else:
        return 'D3'


def update_image_group(name):
    if name == 'CT_PET' or name == 'CT_PET_T' or name == 'CT_PET_T_N':
        return 'Image models (D3)'
    if 'clinical' in name:
        return '+Clinical (D1 + D3)'
    elif 'radiomics' in name:
        return '+Radiomics (D2 + D3)'
    elif 'tabular' in name:
        return '+Tabular (D1 + D2 + D3)'


def update_select_group(name):
    if 'CT_PET' in name:
        return None
    if 'selected_0' in name:
        return 'At least once'
    elif 'selected_10' in name:
        return '10%'
    elif 'selected_50' in name:
        return '50%'
    elif 'selected_90' in name:
        return '90%'
    else:
        return 'All'


# DFS_df['weighted_score'] = (DFS_df['auc'] + DFS_df['mcc']/2 + 0.5 + DFS_df['accuracy'] + DFS_df['f1'] + DFS_df['f1_0']) / 5
DFS_df['weighted_score'] = (DFS_df['auc'] + DFS_df['mcc']/2 + 0.5) / 2
DFS_df['base_model'] = DFS_df['name'].apply(update_base_name)
DFS_df['image_type'] = DFS_df['name'].apply(update_image_type_name)
DFS_df['dataset_type'] = DFS_df['name'].apply(update_type)
DFS_df['image_group'] = DFS_df['name'].apply(update_image_group)
DFS_df['select_group'] = DFS_df['name'].apply(update_select_group)

OS_df['weighted_score'] = (OS_df['auc'] + OS_df['mcc']/2 + 0.5) / 2
OS_df['base_model'] = OS_df['name'].apply(update_base_name)
OS_df['image_type'] = OS_df['name'].apply(update_image_type_name)
OS_df['dataset_type'] = OS_df['name'].apply(update_type)
OS_df['image_group'] = OS_df['name'].apply(update_image_group)
OS_df['select_group'] = OS_df['name'].apply(update_select_group)

# DFS_df[DFS_df.center=='MAASTRO'].sort_values('weighted_score', ascending=False)[['name', 'auc', 'mcc', 'f1', 'f1_0', 'accuracy']].head(50)


# region Plot image model DFS decimal
# find best models for each category
selected_names = DFS_df[DFS_df.center == 'MAASTRO'].sort_values(
    'weighted_score', ascending=False).groupby(
        ['image_group', 'image_type']).first()['name'].values
image_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].round(2).reset_index()
# image_plot_data['auc'] = (image_plot_data['auc'] * 100).astype(int)
# image_plot_data['mcc'] = (image_plot_data['mcc'] * 100).astype(int)


ax = sns.catplot(data=image_plot_data,
                 x='image_type', y='auc',
                 hue='center',
                 col='image_group',
                 kind='bar',
                 #  legend_out=False,
                 height=3.5,
                 aspect=0.84/2)

ax.set_axis_labels("", "AUC")
ax.set(ylim=(.55, .85))
for mpl_ax in ax._axes.flatten():
    first = True
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars, fmt='%.2f' + ('  ' if first else ''))
        first = False
ax.set_titles("{col_name}")
ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.15, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center',)
plt.savefig('outcome_paper_figures_reviewer/DFS_image_auc_decimal.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_image_auc_decimal.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

ax = sns.catplot(data=image_plot_data,
                 x='image_type', y='mcc',
                 hue='center',
                 col='image_group',
                 kind='bar',
                 #  legend=False,
                 height=3.5,
                 aspect=0.84/2)
ax.set_axis_labels("", "MCC")
ax.set(ylim=(0, .6))
for mpl_ax in ax._axes.flatten():
    first = True
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars, fmt='%.2f' + ('  ' if first else ''))
        first = False
ax.set_titles("{col_name}")
ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.15, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center',)
plt.savefig('outcome_paper_figures_reviewer/DFS_image_mcc_decimal.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_image_mcc_decimal.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()


# endregion Plot image model

# region Plot image model DFS
# find best models for each category
selected_names = DFS_df[DFS_df.center == 'MAASTRO'].sort_values(
    'weighted_score', ascending=False).groupby(
        ['image_group', 'image_type']).first()['name'].values
image_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].round(2).reset_index()
image_plot_data['auc'] = (image_plot_data['auc'] * 100).astype(int)
image_plot_data['mcc'] = (image_plot_data['mcc'] * 100).astype(int)


ax = sns.catplot(data=image_plot_data,
                 x='image_type', y='auc',
                 hue='center',
                 col='image_group',
                 kind='bar',
                 #  legend_out=False,
                 height=3.5,
                 aspect=0.84/2)

ax.set_axis_labels("", "AUC (%)")
ax.set(ylim=(55, 85))
for mpl_ax in ax._axes.flatten():
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles("{col_name}")
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.15, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center',)
plt.savefig('outcome_paper_figures_reviewer/DFS_image_auc.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_image_auc.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

ax = sns.catplot(data=image_plot_data,
                 x='image_type', y='mcc',
                 hue='center',
                 col='image_group',
                 kind='bar',
                 #  legend=False,
                 height=3.5,
                 aspect=0.84/2)
ax.set_axis_labels("", "MCC (%)")
ax.set(ylim=(0, 60))
for mpl_ax in ax._axes.flatten():
    first = True
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles("{col_name}")
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.15, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center',)
plt.savefig('outcome_paper_figures_reviewer/DFS_image_mcc.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_image_mcc.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()


# endregion Plot image model


# region Plot image model OS
# find best models for each category
selected_names = OS_df[OS_df.center == 'MAASTRO'].sort_values(
    'weighted_score', ascending=False).groupby(
        ['image_group', 'image_type']).first()['name'].values
image_plot_data = OS_df[OS_df.name.isin(
    selected_names)].round(2).reset_index()
# image_plot_data['auc'] = (image_plot_data['auc'] * 100).astype(int)
# image_plot_data['mcc'] = (image_plot_data['mcc'] * 100).astype(int)


ax = sns.catplot(data=image_plot_data,
                 x='image_type', y='auc',
                 hue='center',
                 col='image_group',
                 kind='bar',
                 #  legend_out=False,
                 height=3.5,
                 aspect=0.84/2)

ax.set_axis_labels("", "AUC")
ax.set(ylim=(.55, .85))
for mpl_ax in ax._axes.flatten():
    first = True
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars, fmt='%.2f' + ('  ' if first else ''))
        first = False
ax.set_titles("{col_name}")
ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.15, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center',)
plt.savefig('outcome_paper_figures_reviewer/OS_image_auc_decimal.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_image_auc_decimal.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

ax = sns.catplot(data=image_plot_data,
                 x='image_type', y='mcc',
                 hue='center',
                 col='image_group',
                 kind='bar',
                 #  legend=False,
                 height=3.5,
                 aspect=0.84/2)
ax.set_axis_labels("", "MCC")
ax.set(ylim=(0, .6))
for mpl_ax in ax._axes.flatten():
    first = True
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars, fmt='%.2f' + ('  ' if first else ''))
        first = False
ax.set_titles("{col_name}")
ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.15, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center',)
plt.savefig('outcome_paper_figures_reviewer/OS_image_mcc_decimal.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_image_mcc_decimal.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()


# endregion Plot image model

# region Plot image model OS
# find best models for each category
selected_names = OS_df[OS_df.center == 'MAASTRO'].sort_values(
    'weighted_score', ascending=False).groupby(
        ['image_group', 'image_type']).first()['name'].values
image_plot_data = OS_df[OS_df.name.isin(
    selected_names)].round(2).reset_index()
image_plot_data['auc'] = (image_plot_data['auc'] * 100).astype(int)
image_plot_data['mcc'] = (image_plot_data['mcc'] * 100).astype(int)


ax = sns.catplot(data=image_plot_data,
                 x='image_type', y='auc',
                 hue='center',
                 col='image_group',
                 kind='bar',
                 #  legend_out=False,
                 height=3.5,
                 aspect=0.84/2)

ax.set_axis_labels("", "AUC (%)")
ax.set(ylim=(55, 85))
for mpl_ax in ax._axes.flatten():
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles("{col_name}")
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.15, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center',)
plt.savefig('outcome_paper_figures_reviewer/OS_image_auc.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_image_auc.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

ax = sns.catplot(data=image_plot_data,
                 x='image_type', y='mcc',
                 hue='center',
                 col='image_group',
                 kind='bar',
                 #  legend=False,
                 height=3.5,
                 aspect=0.84/2)
ax.set_axis_labels("", "MCC (%)")
ax.set(ylim=(0, 60))
for mpl_ax in ax._axes.flatten():
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles("{col_name}")
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.15, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center',)
plt.savefig('outcome_paper_figures_reviewer/OS_image_mcc.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_image_mcc.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()


# endregion Plot image model

# region Plot image model DFS all metrics

selected_names = DFS_df[DFS_df.center == 'MAASTRO'].sort_values(
    'weighted_score', ascending=False).groupby(
        ['image_group', 'image_type']).first()['name'].values
image_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
image_plot_data['mcc'] = (image_plot_data['mcc'] / 2) + 0.5
image_plot_data = image_plot_data.round(2)
image_plot_data['auc'] = (image_plot_data['auc'] * 100).astype(int)
image_plot_data['mcc'] = (image_plot_data['mcc'] * 100).astype(int)
image_plot_data['f1'] = (image_plot_data['f1'] * 100).astype(int)
image_plot_data['f1_0'] = (image_plot_data['f1_0'] * 100).astype(int)
image_plot_data['accuracy'] = (image_plot_data['accuracy'] * 100).astype(int)
image_plot_data['precision'] = (image_plot_data['precision'] * 100).astype(int)
image_plot_data['recall'] = (image_plot_data['recall'] * 100).astype(int)
image_plot_data['specificity'] = (
    image_plot_data['specificity'] * 100).astype(int)


multi_data = image_plot_data.melt(['name', 'image_type', 'center', 'image_group'],
                                  value_vars=['auc', 'mcc',
                                              'f1', 'f1_0', 'accuracy',
                                              'precision', 'recall', 'specificity'],
                                  var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data,
                 x='image_type', y='score',
                 order=['M5', 'M6', 'M7'],
                 hue='center',
                 col='image_group',
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=0.87
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(55, 90))
for j in range(5):
    for i in range(4):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC scaled (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_image_all.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_image_all.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()


multi_data.to_csv(
    'outcome_paper_figures_reviewer/DFS_image_all.csv', index=False)

# endregion

# region Plot image model OS all metrics

selected_names = OS_df[OS_df.center == 'MAASTRO'].sort_values(
    'weighted_score', ascending=False).groupby(
        ['image_group', 'image_type']).first()['name'].values
image_plot_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
image_plot_data['mcc'] = (image_plot_data['mcc'] / 2) + 0.5
image_plot_data = image_plot_data.round(2)
image_plot_data['auc'] = (image_plot_data['auc'] * 100).astype(int)
image_plot_data['mcc'] = (image_plot_data['mcc'] * 100).astype(int)
image_plot_data['f1'] = (image_plot_data['f1'] * 100).astype(int)
image_plot_data['f1_0'] = (image_plot_data['f1_0'] * 100).astype(int)
image_plot_data['accuracy'] = (image_plot_data['accuracy'] * 100).astype(int)
image_plot_data['precision'] = (image_plot_data['precision'] * 100).astype(int)
image_plot_data['recall'] = (image_plot_data['recall'] * 100).astype(int)
image_plot_data['specificity'] = (
    image_plot_data['specificity'] * 100).astype(int)

multi_data = image_plot_data.melt(['name', 'image_type', 'center', 'image_group'],
                                  value_vars=['auc', 'mcc',
                                              'f1', 'f1_0', 'accuracy',
                                              'precision', 'recall', 'specificity'],
                                  var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data,
                 x='image_type', y='score',
                 order=['M5', 'M6', 'M7'],
                 hue='center',
                 col='image_group',
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=0.87
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(55, 90))
for j in range(5):
    for i in range(4):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_image_all.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_image_all.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
multi_data.to_csv(
    'outcome_paper_figures_reviewer/OS_image_all.csv', index=False)

# endregion


# region Plot tabular DFS
# find best models for each category
selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].round(2).reset_index()
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)

ax = sns.catplot(data=clinical_plot_data,
                 x='base_model', y='auc',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='dataset_type',
                 row_order=['D1', 'D2', 'D1+D2'],
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 height=2.5,
                 aspect=0.61)

# ax.set_axis_labels(None, "AUC (%)")
ax.set(ylim=(40, 90))
for j in range(3):
    for i in range(4):
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))
for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template="{row_name}", col_template="{col_name}")
for i in range(3):
    ax._axes[i, 0].set_ylabel('AUC (%)')
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_tabular.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_tabular.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion Plot tabular DFS


# region Plot tabular OS
# find best models for each category
selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].round(2).reset_index()
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)

ax = sns.catplot(data=clinical_plot_data,
                 x='base_model', y='auc',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='dataset_type',
                 row_order=['D1', 'D2', 'D1+D2'],
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 height=2.5,
                 aspect=0.61)

# ax.set_axis_labels(None, "AUC (%)")
ax.set(ylim=(40, 90))
for j in range(3):
    for i in range(4):
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))
for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template="{row_name}", col_template="{col_name}")
for i in range(3):
    ax._axes[i, 0].set_ylabel('AUC (%)')
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_tabular.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_tabular.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion Plot tabular OS


# region Plot DFS LR
selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].round(2).reset_index()
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)

ax = sns.catplot(data=clinical_plot_data[clinical_plot_data.base_model == 'M1'],
                 x='base_model',
                 y='auc',
                 hue='center',
                 col='select_group',
                 #  col_order=['All', 'At least once', '50%', '90%'],
                 row='dataset_type',
                 row_order=['D1', 'D2', 'D1+D2'],
                 kind='bar',
                 #  legend_out=False,
                 margin_titles=True,
                 height=2.5,
                 aspect=0.35)

# ax.set_axis_labels(None, "AUC (%)")
ax.set(ylim=(40, 90))
for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template="{row_name}", col_template="{col_name}")
for i in range(3):
    ax._axes[i, 0].set_ylabel('AUC (%)')
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_tabular_logistic.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_tabular_logistic.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()


# endregion

# region Plot OS LR
selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].round(2).reset_index()
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)

ax = sns.catplot(data=clinical_plot_data[clinical_plot_data.base_model == 'M1'],
                 x='base_model',
                 y='auc',
                 hue='center',
                 col='select_group',
                 #  col_order=['All', 'At least once', '50%', '90%'],
                 row='dataset_type',
                 row_order=['D1', 'D2', 'D1+D2'],
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 height=2.5,
                 aspect=0.35)

# ax.set_axis_labels(None, "AUC (%)")
ax.set(ylim=(40, 90))
for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template="{row_name}", col_template="{col_name}")
for i in range(3):
    ax._axes[i, 0].set_ylabel('AUC (%)')
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_tabular_logistic.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_tabular_logistic.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()


# endregion


# region Plot clinical model DFS all metrics

selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)
clinical_plot_data['precision'] = (
    clinical_plot_data['precision'] * 100).astype(int)
clinical_plot_data['recall'] = (
    clinical_plot_data['recall'] * 100).astype(int)
clinical_plot_data['specificity'] = (
    clinical_plot_data['specificity'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=0.87
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
for j in range(5):
    for i in range(4):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC scaled (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_clinical_all.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_clinical_all.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

multi_data[multi_data.dataset_type == 'D1'].to_csv(
    'outcome_paper_figures_reviewer/DFS_clinical_all.csv', index=False)
# endregion

# region Plot clinical model OS all metrics

selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)
clinical_plot_data['precision'] = (
    clinical_plot_data['precision'] * 100).astype(int)
clinical_plot_data['recall'] = (
    clinical_plot_data['recall'] * 100).astype(int)
clinical_plot_data['specificity'] = (
    clinical_plot_data['specificity'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=0.87
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
for j in range(5):
    for i in range(4):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC scaled (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_clinical_all.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_clinical_all.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
multi_data[multi_data.dataset_type == 'D1'].to_csv(
    'outcome_paper_figures_reviewer/OS_clinical_all.csv', index=False)
# endregion

# region Plot radiomics model DFS all metrics

selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)
clinical_plot_data['precision'] = (
    clinical_plot_data['precision'] * 100).astype(int)
clinical_plot_data['recall'] = (
    clinical_plot_data['recall'] * 100).astype(int)
clinical_plot_data['specificity'] = (
    clinical_plot_data['specificity'] * 100).astype(int)
multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=0.87
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
for j in range(5):
    for i in range(4):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC scaled (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_radiomics_all.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_radiomics_all.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
multi_data[multi_data.dataset_type == 'D2'].to_csv(
    'outcome_paper_figures_reviewer/DFS_radiomics_all.csv', index=False)

# endregion

# region Plot radiomics model OS all metrics

selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)
clinical_plot_data['precision'] = (
    clinical_plot_data['precision'] * 100).astype(int)
clinical_plot_data['recall'] = (
    clinical_plot_data['recall'] * 100).astype(int)
clinical_plot_data['specificity'] = (
    clinical_plot_data['specificity'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=0.87
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
for j in range(5):
    for i in range(4):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC scaled (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_radiomics_all.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_radiomics_all.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
multi_data[multi_data.dataset_type == 'D2'].to_csv(
    'outcome_paper_figures_reviewer/OS_radiomics_all.csv', index=False)

# endregion

# region Plot d1 d2 model DFS all metrics

selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)
clinical_plot_data['precision'] = (
    clinical_plot_data['precision'] * 100).astype(int)
clinical_plot_data['recall'] = (
    clinical_plot_data['recall'] * 100).astype(int)
clinical_plot_data['specificity'] = (
    clinical_plot_data['specificity'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1+D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=0.87
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
for j in range(5):
    for i in range(4):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC scaled (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_tab_all.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_tab_all.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
multi_data[multi_data.dataset_type ==
           'D1+D2'].to_csv('outcome_paper_figures_reviewer/DFS_tab_all.csv', index=False)
# endregion

# region Plot d1 d2 model OS all metrics

selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)
clinical_plot_data['precision'] = (
    clinical_plot_data['precision'] * 100).astype(int)
clinical_plot_data['recall'] = (
    clinical_plot_data['recall'] * 100).astype(int)
clinical_plot_data['specificity'] = (
    clinical_plot_data['specificity'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1+D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=0.87
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
for j in range(5):
    for i in range(4):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC scaled (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_tab_all.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_tab_all.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
multi_data[multi_data.dataset_type ==
           'D1+D2'].to_csv('outcome_paper_figures_reviewer/OS_tab_all.csv', index=False)
# endregion

# remove 10%

# region Plot clinical model DFS all metrics

selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=1.1
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
for j in range(5):
    for i in range(3):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC scaled (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_clinical_all_short.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_clinical_all_short.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot clinical model OS all metrics

selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=1.1
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
for j in range(5):
    for i in range(3):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC scaled (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_clinical_all_short.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_clinical_all_short.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot radiomics model DFS all metrics

selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=1.1
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
for j in range(5):
    for i in range(3):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC scaled (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_radiomics_all_short.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_radiomics_all_short.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot radiomics model OS all metrics

selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=1.1
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
for j in range(5):
    for i in range(3):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC scaled (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_radiomics_all_short.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_radiomics_all_short.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot d1 d2 model DFS all metrics

selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1+D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=1.1
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
for j in range(5):
    for i in range(3):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC scaled (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_tab_all_short.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_tab_all_short.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot d1 d2 model OS all metrics

selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1+D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 sharey=False,
                 sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=1.55,
                 aspect=1.1
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
for j in range(5):
    for i in range(3):
        # if j == 1:
        #     print(j)
        #     ax.axes[1, i].set_ylim(0, 65)
        # else:
        #     ax.axes[j, i].set_ylim(55, 85)
        if i != 0:
            y_labels = ax.axes[j, i].get_yticks()
            ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC scaled (%)')
ax._axes[2, 0].set_ylabel('F1 score class 1 (%)')
ax._axes[3, 0].set_ylabel('F1 score class 0 (%)')
ax._axes[4, 0].set_ylabel('Accuracy (%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
# plt.savefig('outcome_paper_figures_reviewer/OS_tab_all_short.pdf',
#             edgecolor='black', dpi=300, facecolor='white', transparent=True)
# plt.savefig('outcome_paper_figures_reviewer/OS_tab_all_short.png',
#             edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# plt.rc('font', size=8)
# ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1+D2'],
#                  x='base_model', y='score',
#                  hue='center',
#                  col='select_group',
#                  col_order=['All', 'At least once', '50%'],
#                  row='metric',
#                  kind='bar',
#                  #  legend_out=False,
#                 #  sharey=False,
#                 #  sharex=False,
#                  margin_titles=True,
#                  #  height=3.5,
#                  #  aspect=0.84/2
#                  height=0.75,
#                  aspect=1.8
#                  )

# # ax.set_axis_labels("", "{row_name} (%)")
# ax.set(ylim=(35, 90))
# # for j in range(5):
# #     for i in range(3):
# #         # if j == 1:
# #         #     print(j)
# #         #     ax.axes[1, i].set_ylim(0, 65)
# #         # else:
# #         #     ax.axes[j, i].set_ylim(55, 85)
# #         if i != 0:
# #             y_labels = ax.axes[j, i].get_yticks()
# #             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

# for mpl_ax in ax._axes.flatten():
#     mpl_ax.set_xlabel('')
#     for bars in mpl_ax.containers:
#         mpl_ax.bar_label(bars)
# ax.set_titles(row_template='', col_template='{col_name}')
# ax._axes[0, 0].set_ylabel('AUC (%)')
# ax._axes[1, 0].set_ylabel('MCC\nscaled (%)')
# ax._axes[2, 0].set_ylabel('F1 score\nclass 1 (%)')
# ax._axes[3, 0].set_ylabel('F1 score\nclass 0 (%)')
# ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# # ax.set(margin_titles=True)
# # ax.despine(left=True)
# plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
# sns.move_legend(ax, "lower center", title='', ncol=2)
# plt.savefig('outcome_paper_figures_reviewer/check_size.pdf',
#             edgecolor='black', dpi=300, facecolor='white', transparent=True)
# plt.savefig('outcome_paper_figures_reviewer/check_size.png',
#             edgecolor='black', dpi=300, facecolor='white', transparent=True)
# plt.show()

plt.rc('font', size=8)
ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1+D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '10%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=0.6,
                 aspect=2.2
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
# for j in range(5):
#     for i in range(3):
#         # if j == 1:
#         #     print(j)
#         #     ax.axes[1, i].set_ylim(0, 65)
#         # else:
#         #     ax.axes[j, i].set_ylim(55, 85)
#         if i != 0:
#             y_labels = ax.axes[j, i].get_yticks()
#             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC\nscaled\n(%)')
ax._axes[2, 0].set_ylabel('F1\nclass 1\n(%)')
ax._axes[3, 0].set_ylabel('F1\nclass 0\n(%)')
ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.00001, h_pad=0.04)
sns.move_legend(ax, "lower center", title='', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/check_size_2.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/check_size_2.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# size xs
plt.rc('font', size=8)

# region Plot clinical model DFS all metrics

selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=0.6,
                 aspect=2.8
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
# for j in range(5):
#     for i in range(3):
#         # if j == 1:
#         #     print(j)
#         #     ax.axes[1, i].set_ylim(0, 65)
#         # else:
#         #     ax.axes[j, i].set_ylim(55, 85)
#         if i != 0:
#             y_labels = ax.axes[j, i].get_yticks()
#             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC\nscaled\n(%)')
ax._axes[2, 0].set_ylabel('F1\nclass 1\n(%)')
ax._axes[3, 0].set_ylabel('F1\nclass 0\n(%)')
ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001, h_pad=0.04)
sns.move_legend(ax, "lower center", title='', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_clinical_all_xs.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_clinical_all_xs.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot clinical model OS all metrics

selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=0.6,
                 aspect=2.8
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
# for j in range(5):
#     for i in range(3):
#         # if j == 1:
#         #     print(j)
#         #     ax.axes[1, i].set_ylim(0, 65)
#         # else:
#         #     ax.axes[j, i].set_ylim(55, 85)
#         if i != 0:
#             y_labels = ax.axes[j, i].get_yticks()
#             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC\nscaled\n(%)')
ax._axes[2, 0].set_ylabel('F1\nclass 1\n(%)')
ax._axes[3, 0].set_ylabel('F1\nclass 0\n(%)')
ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001, h_pad=0.04)
sns.move_legend(ax, "lower center", title='', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_clinical_all_xs.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_clinical_all_xs.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot radiomics model DFS all metrics

selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=0.6,
                 aspect=2.8
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
# for j in range(5):
#     for i in range(3):
#         # if j == 1:
#         #     print(j)
#         #     ax.axes[1, i].set_ylim(0, 65)
#         # else:
#         #     ax.axes[j, i].set_ylim(55, 85)
#         if i != 0:
#             y_labels = ax.axes[j, i].get_yticks()
#             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC\nscaled\n(%)')
ax._axes[2, 0].set_ylabel('F1\nclass 1\n(%)')
ax._axes[3, 0].set_ylabel('F1\nclass 0\n(%)')
ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001, h_pad=0.04)
sns.move_legend(ax, "lower center", title='', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_radiomics_all_xs.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_radiomics_all_xs.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot radiomics model OS all metrics

selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=0.6,
                 aspect=2.8
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
# for j in range(5):
#     for i in range(3):
#         # if j == 1:
#         #     print(j)
#         #     ax.axes[1, i].set_ylim(0, 65)
#         # else:
#         #     ax.axes[j, i].set_ylim(55, 85)
#         if i != 0:
#             y_labels = ax.axes[j, i].get_yticks()
#             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC\nscaled\n(%)')
ax._axes[2, 0].set_ylabel('F1\nclass 1\n(%)')
ax._axes[3, 0].set_ylabel('F1\nclass 0\n(%)')
ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001, h_pad=0.04)
sns.move_legend(ax, "lower center", title='', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_radiomics_all_xs.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_radiomics_all_xs.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot d1 d2 model DFS all metrics

selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1+D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=0.6,
                 aspect=2.8
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
# for j in range(5):
#     for i in range(3):
#         # if j == 1:
#         #     print(j)
#         #     ax.axes[1, i].set_ylim(0, 65)
#         # else:
#         #     ax.axes[j, i].set_ylim(55, 85)
#         if i != 0:
#             y_labels = ax.axes[j, i].get_yticks()
#             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC\nscaled\n(%)')
ax._axes[2, 0].set_ylabel('F1\nclass 1\n(%)')
ax._axes[3, 0].set_ylabel('F1\nclass 0\n(%)')
ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001, h_pad=0.04)
sns.move_legend(ax, "lower center", title='', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_tab_all_xs.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_tab_all_xs.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot d1 d2 model OS all metrics

selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1+D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=0.6,
                 aspect=2.8
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
# for j in range(5):
#     for i in range(3):
#         # if j == 1:
#         #     print(j)
#         #     ax.axes[1, i].set_ylim(0, 65)
#         # else:
#         #     ax.axes[j, i].set_ylim(55, 85)
#         if i != 0:
#             y_labels = ax.axes[j, i].get_yticks()
#             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC\nscaled\n(%)')
ax._axes[2, 0].set_ylabel('F1\nclass 1\n(%)')
ax._axes[3, 0].set_ylabel('F1\nclass 0\n(%)')
ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001, h_pad=0.04)
sns.move_legend(ax, "lower center", title='', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_tab_all_xs.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_tab_all_xs.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# size s

plt.rc('font', size=8)
# region Plot clinical model DFS all metrics

selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=0.6,
                 aspect=2.2
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
# for j in range(5):
#     for i in range(3):
#         # if j == 1:
#         #     print(j)
#         #     ax.axes[1, i].set_ylim(0, 65)
#         # else:
#         #     ax.axes[j, i].set_ylim(55, 85)
#         if i != 0:
#             y_labels = ax.axes[j, i].get_yticks()
#             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC\nscaled\n(%)')
ax._axes[2, 0].set_ylabel('F1\nclass 1\n(%)')
ax._axes[3, 0].set_ylabel('F1\nclass 0\n(%)')
ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001, h_pad=0.04)
sns.move_legend(ax, "lower center", title='', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_clinical_all_s.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_clinical_all_s.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot clinical model OS all metrics

selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=0.6,
                 aspect=2.2
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
# for j in range(5):
#     for i in range(3):
#         # if j == 1:
#         #     print(j)
#         #     ax.axes[1, i].set_ylim(0, 65)
#         # else:
#         #     ax.axes[j, i].set_ylim(55, 85)
#         if i != 0:
#             y_labels = ax.axes[j, i].get_yticks()
#             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC\nscaled\n(%)')
ax._axes[2, 0].set_ylabel('F1\nclass 1\n(%)')
ax._axes[3, 0].set_ylabel('F1\nclass 0\n(%)')
ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001, h_pad=0.04)
sns.move_legend(ax, "lower center", title='', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_clinical_all_s.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_clinical_all_s.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot radiomics model DFS all metrics

selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=0.6,
                 aspect=2.2
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
# for j in range(5):
#     for i in range(3):
#         # if j == 1:
#         #     print(j)
#         #     ax.axes[1, i].set_ylim(0, 65)
#         # else:
#         #     ax.axes[j, i].set_ylim(55, 85)
#         if i != 0:
#             y_labels = ax.axes[j, i].get_yticks()
#             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC\nscaled\n(%)')
ax._axes[2, 0].set_ylabel('F1\nclass 1\n(%)')
ax._axes[3, 0].set_ylabel('F1\nclass 0\n(%)')
ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001, h_pad=0.04)
sns.move_legend(ax, "lower center", title='', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_radiomics_all_s.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_radiomics_all_s.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot radiomics model OS all metrics

selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=0.6,
                 aspect=2.2
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
# for j in range(5):
#     for i in range(3):
#         # if j == 1:
#         #     print(j)
#         #     ax.axes[1, i].set_ylim(0, 65)
#         # else:
#         #     ax.axes[j, i].set_ylim(55, 85)
#         if i != 0:
#             y_labels = ax.axes[j, i].get_yticks()
#             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC\nscaled\n(%)')
ax._axes[2, 0].set_ylabel('F1\nclass 1\n(%)')
ax._axes[3, 0].set_ylabel('F1\nclass 0\n(%)')
ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001, h_pad=0.04)
sns.move_legend(ax, "lower center", title='', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_radiomics_all_s.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_radiomics_all_s.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot d1 d2 model DFS all metrics

selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1+D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=0.6,
                 aspect=2.2
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
# for j in range(5):
#     for i in range(3):
#         # if j == 1:
#         #     print(j)
#         #     ax.axes[1, i].set_ylim(0, 65)
#         # else:
#         #     ax.axes[j, i].set_ylim(55, 85)
#         if i != 0:
#             y_labels = ax.axes[j, i].get_yticks()
#             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC\nscaled\n(%)')
ax._axes[2, 0].set_ylabel('F1\nclass 1\n(%)')
ax._axes[3, 0].set_ylabel('F1\nclass 0\n(%)')
ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001, h_pad=0.04)
sns.move_legend(ax, "lower center", title='', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/DFS_tab_all_s.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_tab_all_s.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot d1 d2 model OS all metrics

selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)

multi_data = clinical_plot_data.melt(['name', 'base_model', 'center', 'dataset_type', 'select_group'],
                                     value_vars=['auc', 'mcc',
                                                 'f1', 'f1_0', 'accuracy',
                                                 'precision', 'recall', 'specificity'],
                                     var_name='metric', value_name='score')

ax = sns.catplot(data=multi_data[multi_data.dataset_type == 'D1+D2'],
                 x='base_model', y='score',
                 hue='center',
                 col='select_group',
                 col_order=['All', 'At least once', '50%', '90%'],
                 row='metric',
                 kind='bar',
                 #  legend_out=False,
                 #  sharey=False,
                 #  sharex=False,
                 margin_titles=True,
                 #  height=3.5,
                 #  aspect=0.84/2
                 height=0.6,
                 aspect=2.2
                 )

# ax.set_axis_labels("", "{row_name} (%)")
ax.set(ylim=(35, 90))
# for j in range(5):
#     for i in range(3):
#         # if j == 1:
#         #     print(j)
#         #     ax.axes[1, i].set_ylim(0, 65)
#         # else:
#         #     ax.axes[j, i].set_ylim(55, 85)
#         if i != 0:
#             y_labels = ax.axes[j, i].get_yticks()
#             ax.axes[j, i].set_yticklabels([''] * len(y_labels))

for mpl_ax in ax._axes.flatten():
    mpl_ax.set_xlabel('')
    for bars in mpl_ax.containers:
        mpl_ax.bar_label(bars)
ax.set_titles(row_template='', col_template='{col_name}')
ax._axes[0, 0].set_ylabel('AUC (%)')
ax._axes[1, 0].set_ylabel('MCC\nscaled\n(%)')
ax._axes[2, 0].set_ylabel('F1\nclass 1\n(%)')
ax._axes[3, 0].set_ylabel('F1\nclass 0\n(%)')
ax._axes[4, 0].set_ylabel('Accuracy\n(%)')

# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001, h_pad=0.04)
sns.move_legend(ax, "lower center", title='', ncol=2)
plt.savefig('outcome_paper_figures_reviewer/OS_tab_all_s.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_tab_all_s.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion


# region Plot clinical model DFS all metrics

selected_names = DFS_df[DFS_df.image_type.isnull()].name
clinical_plot_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)
clinical_plot_data['avg_score'] = (clinical_plot_data['mcc'] +
                                   clinical_plot_data['auc'] +
                                   clinical_plot_data['accuracy'] +
                                   clinical_plot_data['f1'] +
                                   clinical_plot_data['f1_0']) / 5
clinical_plot_data = clinical_plot_data[clinical_plot_data['select_group'] == 'All'].reset_index(
    drop=True)

plt.figure(figsize=(10, 15))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0', 'accuracy', metric]):
    plt.subplot(3, 2, i+1)
    scatter_plot_data = clinical_plot_data.pivot_table(
        values=metric,
        index=['base_model', 'dataset_type'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         hue='dataset_type', hue_order=['D1', 'D2', 'D1+D2'],
                         legend=False)

    for line in range(0, scatter_plot_data.shape[0]):
        p1.text(scatter_plot_data.OUS[line]+0.01, scatter_plot_data.MAASTRO[line],
                scatter_plot_data.base_model[line], horizontalalignment='left',
                size='medium', color='black', weight='semibold')
    ax = plt.plot([0, 100], [0, 100], ls="--", c=".3")
    # plt.set(xlim=(0, 100), ylim=(0, 100))
    plt.xlim(45, 80)
    plt.ylim(45, 80)
    if metric == 'auc':
        plt.title('AUC')
    elif metric == 'mcc':
        plt.title('MCC')
    elif metric == 'f1':
        plt.title('F1 score class 1')
    elif metric == 'f1_0':
        plt.title('F1 score class 0')
    elif metric == 'accuracy':
        plt.title('Accuracy')
    else:
        plt.title('Metric average')

    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.savefig('dfs_tab_all.png')
plt.tight_layout()
plt.show()

scatter_plot_data = clinical_plot_data.pivot_table(
    values='mcc', index=['base_model', 'dataset_type'], columns='center').reset_index()

p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                     hue='dataset_type', hue_order=['D1', 'D2', 'D1+D2'])

for line in range(0, scatter_plot_data.shape[0]):
    p1.text(scatter_plot_data.OUS[line]+0.01, scatter_plot_data.MAASTRO[line],
            scatter_plot_data.base_model[line], horizontalalignment='left',
            size='medium', color='black', weight='semibold')
ax = plt.plot([0, 100], [0, 100], ls="--", c=".3")
# plt.set(xlim=(0, 100), ylim=(0, 100))
plt.xlim(50, 75)
plt.ylim(50, 75)
plt.title('DFS MCC')
# Set x-axis label
plt.xlabel('OUS')
# Set y-axis label
plt.ylabel('MAASTRO')

plt.show()


# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
# plt.savefig('outcome_paper_figures_reviewer/DFS_clinical_all.pdf',
#             edgecolor='black', dpi=300, facecolor='white', transparent=True)
# plt.savefig('outcome_paper_figures_reviewer/DFS_clinical_all.png',
#             edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion

# region Plot clinical model OS all metrics

selected_names = OS_df[OS_df.image_type.isnull()].name
clinical_plot_data = OS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] / 2) + 0.5
clinical_plot_data = clinical_plot_data.round(2)
clinical_plot_data['auc'] = (clinical_plot_data['auc'] * 100).astype(int)
clinical_plot_data['mcc'] = (clinical_plot_data['mcc'] * 100).astype(int)
clinical_plot_data['f1'] = (clinical_plot_data['f1'] * 100).astype(int)
clinical_plot_data['f1_0'] = (clinical_plot_data['f1_0'] * 100).astype(int)
clinical_plot_data['accuracy'] = (
    clinical_plot_data['accuracy'] * 100).astype(int)
clinical_plot_data['avg_score'] = (clinical_plot_data['mcc'] +
                                   clinical_plot_data['auc'] +
                                   clinical_plot_data['accuracy'] +
                                   clinical_plot_data['f1'] +
                                   clinical_plot_data['f1_0']) / 5
clinical_plot_data = clinical_plot_data[clinical_plot_data['select_group'] == 'All'].reset_index(
    drop=True)
plt.figure(figsize=(10, 15))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0', 'accuracy', metric]):
    plt.subplot(3, 2, i+1)
    scatter_plot_data = clinical_plot_data.pivot_table(
        values=metric,
        index=['base_model', 'dataset_type'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         hue='dataset_type', hue_order=['D1', 'D2', 'D1+D2'],
                         legend=False)

    for line in range(0, scatter_plot_data.shape[0]):
        p1.text(scatter_plot_data.OUS[line]+0.01, scatter_plot_data.MAASTRO[line],
                scatter_plot_data.base_model[line], horizontalalignment='left',
                size='medium', color='black', weight='semibold')
    ax = plt.plot([0, 100], [0, 100], ls="--", c=".3")
    # plt.set(xlim=(0, 100), ylim=(0, 100))
    plt.xlim(35, 85)
    plt.ylim(35, 85)
    if metric == 'auc':
        plt.title('AUC')
    elif metric == 'mcc':
        plt.title('MCC')
    elif metric == 'f1':
        plt.title('F1 score class 1')
    elif metric == 'f1_0':
        plt.title('F1 score class 0')
    elif metric == 'accuracy':
        plt.title('Accuracy')
    else:
        plt.title('Metric average')

    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.savefig('os_tab_all.png')
plt.tight_layout()
plt.show()


scatter_plot_data = clinical_plot_data.pivot_table(values='avg_score', index=[
                                                   'base_model', 'dataset_type'], columns='center').reset_index()

p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                     hue='dataset_type', hue_order=['D1', 'D2', 'D1+D2'])

for line in range(0, scatter_plot_data.shape[0]):
    p1.text(scatter_plot_data.OUS[line]+0.01, scatter_plot_data.MAASTRO[line],
            scatter_plot_data.base_model[line], horizontalalignment='left',
            size='medium', color='black', weight='semibold')
ax = plt.plot([0, 100], [0, 100], ls="--", c=".3")
# plt.set(xlim=(0, 100), ylim=(0, 100))
plt.xlim(50, 80)
plt.ylim(50, 80)
plt.title('')
# Set x-axis label
plt.xlabel('OUS')
# Set y-axis label
plt.ylabel('MAASTRO')

plt.show()


# ax.set(margin_titles=True)
# ax.despine(left=True)
plt.tight_layout(rect=(-0.012, 0.05, 1.012, 1), w_pad=0.0001)
sns.move_legend(ax, "lower center", title='Center', ncol=2)
# plt.savefig('outcome_paper_figures_reviewer/OS_clinical_all.pdf',
#             edgecolor='black', dpi=300, facecolor='white', transparent=True)
# plt.savefig('outcome_paper_figures_reviewer/OS_clinical_all.png',
#             edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()

# endregion


# region Scatter plot DFS
select_image = DFS_df.dataset_type == 'D3'
select_tab = DFS_df.image_type.isnull()
select_group = DFS_df.select_group == 'All'
select_all_tab = select_tab & select_group
all_model_data = DFS_df[select_image | select_all_tab].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(10, 15))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0']):
    plt.subplot(3, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'dataset_type'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         hue='dataset_type', hue_order=['D1', 'D2', 'D1+D2', 'D3'],
                         legend=False)

    for line in range(0, scatter_plot_data.shape[0]):
        p1.text(scatter_plot_data.OUS[line]+0.01, scatter_plot_data.MAASTRO[line],
                scatter_plot_data.base_model[line], horizontalalignment='left',
                size='medium', color='black', weight='semibold')
    ax = plt.plot([0, 100], [0, 100], ls="--", c=".3")
    # plt.set(xlim=(0, 100), ylim=(0, 100))
    plt.xlim(35, 80)
    plt.ylim(35, 80)
    if metric == 'auc':
        plt.title('AUC')
    elif metric == 'mcc':
        plt.title('MCC')
    elif metric == 'f1':
        plt.title('F1 score class 1')
    elif metric == 'f1_0':
        plt.title('F1 score class 0')
    elif metric == 'accuracy':
        plt.title('Accuracy')
    else:
        plt.title('Metric average')

    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.savefig('dfs_tab_all.png')
plt.tight_layout()
plt.show()
# endregion


# region Scatter plot OS
select_image = OS_df.dataset_type == 'D3'
select_tab = OS_df.image_type.isnull()
select_group = OS_df.select_group == 'All'
select_all_tab = select_tab & select_group
all_model_data = OS_df[select_image | select_all_tab].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(10, 15))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0']):
    plt.subplot(3, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'dataset_type'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         hue='dataset_type', hue_order=['D1', 'D2', 'D1+D2', 'D3'],
                         legend=False)

    for line in range(0, scatter_plot_data.shape[0]):
        p1.text(scatter_plot_data.OUS[line]+0.01, scatter_plot_data.MAASTRO[line],
                scatter_plot_data.base_model[line], horizontalalignment='left',
                size='medium', color='black', weight='semibold')
    ax = plt.plot([0, 100], [0, 100], ls="--", c=".3")
    # plt.set(xlim=(0, 100), ylim=(0, 100))
    plt.xlim(35, 80)
    plt.ylim(35, 80)
    if metric == 'auc':
        plt.title('AUC')
    elif metric == 'mcc':
        plt.title('MCC')
    elif metric == 'f1':
        plt.title('F1 score class 1')
    elif metric == 'f1_0':
        plt.title('F1 score class 0')
    elif metric == 'accuracy':
        plt.title('Accuracy')
    else:
        plt.title('Metric average')

    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.savefig('os_tab_all.png')
plt.tight_layout()
plt.show()
# endregion


# region Scatter plot DFS
select_image = DFS_df.dataset_type == 'D3'
select_tab = DFS_df.image_type.isnull()
select_group = DFS_df.select_group == 'All'
select_all_tab = select_tab & select_group
all_model_data = DFS_df[select_image | select_all_tab].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(7, 7))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0']):
    plt.subplot(2, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'dataset_type'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         s=40,
                         hue='dataset_type',
                         hue_order=['D1', 'D2', 'D1+D2', 'D3'],
                         style='base_model',
                         style_order=[f'M{i}' for i in range(1, 8)],
                         legend=i == 3)

    # for line in range(0, scatter_plot_data.shape[0]):
    #     p1.text(scatter_plot_data.OUS[line]+0.01, scatter_plot_data.MAASTRO[line],
    #             scatter_plot_data.base_model[line], horizontalalignment='left',
    #             size='medium', color='black', weight='semibold')
    ax = plt.plot([0, 100], [0, 100], ls="--", c=".3")
    # plt.set(xlim=(0, 100), ylim=(0, 100))
    plt.xlim(35, 80)
    plt.ylim(35, 80)
    if metric == 'auc':
        plt.title('AUC')
    elif metric == 'mcc':
        plt.title('MCC')
    elif metric == 'f1':
        plt.title('F1 score class 1')
    elif metric == 'f1_0':
        plt.title('F1 score class 0')
    elif metric == 'accuracy':
        plt.title('Accuracy')
    else:
        plt.title('Metric average')

    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.tight_layout(rect=(0, 0.05, 1, 1))
h, labels = p1.get_legend_handles_labels()
labels[0] = 'Input type'
labels[5] = 'Models'

l1 = plt.legend(h[:5], labels[:5], loc='lower left', ncol=5,
                handletextpad=0.2,
                columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-0.8, -0.3))
l2 = plt.legend(h[5:], labels[5:], loc='lower left', ncol=8,
                handletextpad=0.2,
                columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-1.1, -0.4))
p1.add_artist(l1)
# sns.move_legend(plt.gca(), "lower left", title='Dataset',
#                 ncol=3, bbox_to_anchor=(-0.5, -0.5))
plt.savefig('outcome_paper_figures_reviewer/DFS_compare_data.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_compare_data.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion


# region Scatter plot OS
select_image = OS_df.dataset_type == 'D3'
select_tab = OS_df.image_type.isnull()
select_group = OS_df.select_group == 'All'
select_all_tab = select_tab & select_group
all_model_data = OS_df[select_image | select_all_tab].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(7, 7))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0']):
    plt.subplot(2, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'dataset_type'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         s=40,
                         hue='dataset_type',
                         hue_order=['D1', 'D2', 'D1+D2', 'D3'],
                         style='base_model',
                         style_order=[f'M{i}' for i in range(1, 8)],
                         legend=i == 3)

    # for line in range(0, scatter_plot_data.shape[0]):
    #     p1.text(scatter_plot_data.OUS[line]+0.01, scatter_plot_data.MAASTRO[line],
    #             scatter_plot_data.base_model[line], horizontalalignment='left',
    #             size='medium', color='black', weight='semibold')
    ax = plt.plot([0, 100], [0, 100], ls="--", c=".3")
    # plt.set(xlim=(0, 100), ylim=(0, 100))
    plt.xlim(35, 85)
    plt.ylim(35, 85)
    if metric == 'auc':
        plt.title('AUC')
    elif metric == 'mcc':
        plt.title('MCC')
    elif metric == 'f1':
        plt.title('F1 score class 1')
    elif metric == 'f1_0':
        plt.title('F1 score class 0')
    elif metric == 'accuracy':
        plt.title('Accuracy')
    else:
        plt.title('Metric average')

    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.tight_layout(rect=(0, 0.05, 1, 1))
h, labels = p1.get_legend_handles_labels()
labels[0] = 'Input type'
labels[5] = 'Models'

l1 = plt.legend(h[:5], labels[:5], loc='lower left', ncol=5,
                handletextpad=0.2,
                columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-0.8, -0.3))
l2 = plt.legend(h[5:], labels[5:], loc='lower left', ncol=8,
                handletextpad=0.2,
                columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-1.1, -0.4))
p1.add_artist(l1)
# sns.move_legend(plt.gca(), "lower left", title='Dataset',
#                 ncol=3, bbox_to_anchor=(-0.5, -0.5))
plt.savefig('outcome_paper_figures_reviewer/OS_compare_data.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_compare_data.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion


# region Scatter plot combine DFS
selected_names = DFS_df[DFS_df.center == 'MAASTRO'].sort_values(
    'weighted_score', ascending=False).groupby(
        ['image_group', 'image_type']).first()['name'].values
all_model_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
# all_model_data = DFS_df[select_image | select_all_tab].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(7, 7))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0']):
    plt.subplot(2, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'dataset_type', 'image_group'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         s=40,
                         hue='image_group',
                         hue_order=['+Clinical (D1 + D3)', '+Radiomics (D2 + D3)',
                                    '+Tabular (D1 + D2 + D3)', 'Image models (D3)'],
                         style='base_model',
                         markers=[(4, 0, 0), (4, 1, 0), '^'],
                         style_order=[f'M{i}' for i in range(5, 8)],
                         legend=i == 3)

    # for line in range(0, scatter_plot_data.shape[0]):
    #     p1.text(scatter_plot_data.OUS[line]+0.01, scatter_plot_data.MAASTRO[line],
    #             scatter_plot_data.base_model[line], horizontalalignment='left',
    #             size='medium', color='black', weight='semibold')
    ax = plt.plot([0, 100], [0, 100], ls="--", c=".3")
    # plt.set(xlim=(0, 100), ylim=(0, 100))
    plt.xlim(50, 85)
    plt.ylim(50, 85)
    if metric == 'auc':
        plt.title('AUC')
    elif metric == 'mcc':
        plt.title('MCC')
    elif metric == 'f1':
        plt.title('F1 score class 1')
    elif metric == 'f1_0':
        plt.title('F1 score class 0')
    elif metric == 'accuracy':
        plt.title('Accuracy')
    else:
        plt.title('Metric average')

    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.tight_layout(rect=(0, 0.05, 1, 1))
h, labels = p1.get_legend_handles_labels()
labels[0] = 'Input group'
labels[5] = 'Models'

input_group_h = [h[4]] + h[1:4]
input_group_label = [labels[4]] + labels[1:4]
l1 = plt.legend(input_group_h, input_group_label, loc='lower left', ncol=4,
                handletextpad=0.05,
                columnspacing=0.5,
                bbox_to_anchor=(-1.4, -0.3))
l2 = plt.legend(h[5:], labels[5:], loc='lower left', ncol=4,
                handletextpad=0.1,
                columnspacing=0.5,
                bbox_to_anchor=(-0.6, -0.4))
p1.add_artist(l1)
# sns.move_legend(plt.gca(), "lower left", title='Dataset',
#                 ncol=3, bbox_to_anchor=(-0.5, -0.5))
plt.savefig('outcome_paper_figures_reviewer/DFS_compare_data_combine.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_compare_data_combine.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion


# region Scatter plot combine OS
selected_names = OS_df[OS_df.center == 'MAASTRO'].sort_values(
    'weighted_score', ascending=False).groupby(
        ['image_group', 'image_type']).first()['name'].values
all_model_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(7, 7))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0']):
    plt.subplot(2, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'dataset_type', 'image_group'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         s=40,
                         hue='image_group',
                         hue_order=['+Clinical (D1 + D3)', '+Radiomics (D2 + D3)',
                                    '+Tabular (D1 + D2 + D3)', 'Image models (D3)'],
                         style='base_model',
                         markers=[(4, 0, 0), (4, 1, 0), '^'],
                         style_order=[f'M{i}' for i in range(5, 8)],
                         legend=i == 3)

    # for line in range(0, scatter_plot_data.shape[0]):
    #     p1.text(scatter_plot_data.OUS[line]+0.01, scatter_plot_data.MAASTRO[line],
    #             scatter_plot_data.base_model[line], horizontalalignment='left',
    #             size='medium', color='black', weight='semibold')
    ax = plt.plot([0, 100], [0, 100], ls="--", c=".3")
    # plt.set(xlim=(0, 100), ylim=(0, 100))
    plt.xlim(50, 85)
    plt.ylim(50, 85)
    if metric == 'auc':
        plt.title('AUC')
    elif metric == 'mcc':
        plt.title('MCC')
    elif metric == 'f1':
        plt.title('F1 score class 1')
    elif metric == 'f1_0':
        plt.title('F1 score class 0')
    elif metric == 'accuracy':
        plt.title('Accuracy')
    else:
        plt.title('Metric average')

    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.tight_layout(rect=(0, 0.05, 1, 1))
h, labels = p1.get_legend_handles_labels()
labels[0] = 'Input group'
labels[5] = 'Models'

input_group_h = [h[4]] + h[1:4]
input_group_label = [labels[4]] + labels[1:4]
l1 = plt.legend(input_group_h, input_group_label, loc='lower left', ncol=4,
                handletextpad=0.05,
                columnspacing=0.5,
                bbox_to_anchor=(-1.4, -0.3))
l2 = plt.legend(h[5:], labels[5:], loc='lower left', ncol=4,
                handletextpad=0.1,
                columnspacing=0.5,
                bbox_to_anchor=(-0.6, -0.4))
p1.add_artist(l1)
# sns.move_legend(plt.gca(), "lower left", title='Dataset',
#                 ncol=3, bbox_to_anchor=(-0.5, -0.5))
plt.savefig('outcome_paper_figures_reviewer/OS_compare_data_combine.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_compare_data_combine.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion


# region Scatter plot DFS clinical RENT
select_clinical = DFS_df.dataset_type == 'D1'
select_tab = DFS_df.image_type.isnull()
# select_group = DFS_df.select_group == 'All'
# select_all_tab = select_tab & select_group
all_model_data = DFS_df[select_clinical & select_tab].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(7, 7))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0']):
    plt.subplot(2, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'select_group'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         s=40,
                         hue='select_group',
                         hue_order=['All', 'At least once', '50%'],
                         palette=[sns.color_palette()[0], sns.color_palette()
                                  [-1], sns.color_palette()[-3]],
                         style='base_model',
                         style_order=[f'M{i}' for i in range(1, 5)],
                         legend=i == 3)

    # for line in range(0, scatter_plot_data.shape[0]):
    #     p1.text(scatter_plot_data.OUS[line]+0.01, scatter_plot_data.MAASTRO[line],
    #             scatter_plot_data.base_model[line], horizontalalignment='left',
    #             size='medium', color='black', weight='semibold')
    ax = plt.plot([0, 100], [0, 100], ls="--", c=".3")
    # plt.set(xlim=(0, 100), ylim=(0, 100))
    plt.xlim(35, 80)
    plt.ylim(35, 80)
    if metric == 'auc':
        plt.title('AUC')
    elif metric == 'mcc':
        plt.title('MCC')
    elif metric == 'f1':
        plt.title('F1 score class 1')
    elif metric == 'f1_0':
        plt.title('F1 score class 0')
    elif metric == 'accuracy':
        plt.title('Accuracy')
    else:
        plt.title('Metric average')

    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.tight_layout(rect=(0, 0.05, 1, 1))
h, labels = p1.get_legend_handles_labels()
labels[0] = 'Data'
labels[4] = 'Models'

l1 = plt.legend(h[1:4], ['All features', 'RENT 1%', 'RENT 50%'], loc='lower left', ncol=4,
                handletextpad=0.2,
                columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-0.8, -0.3))
l2 = plt.legend(h[4:], labels[4:], loc='lower left', ncol=5,
                handletextpad=0.2,
                columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-0.8, -0.4))
p1.add_artist(l1)
# sns.move_legend(plt.gca(), "lower left", title='Dataset',
#                 ncol=3, bbox_to_anchor=(-0.5, -0.5))
plt.savefig('outcome_paper_figures_reviewer/DFS_compare_data_RENT.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/DFS_compare_data_RENT.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion


# region Scatter plot OS clinical RENT
select_clinical = OS_df.dataset_type == 'D1'
select_tab = OS_df.image_type.isnull()
# select_group = DFS_df.select_group == 'All'
# select_all_tab = select_tab & select_group
all_model_data = OS_df[select_clinical & select_tab].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(7, 7))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0']):
    plt.subplot(2, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'select_group'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         s=40,
                         hue='select_group',
                         hue_order=['All', 'At least once', '50%'],
                         palette=[sns.color_palette()[0], sns.color_palette()
                                  [-1], sns.color_palette()[-3]],
                         style='base_model',
                         style_order=[f'M{i}' for i in range(1, 5)],
                         legend=i == 3)

    # for line in range(0, scatter_plot_data.shape[0]):
    #     p1.text(scatter_plot_data.OUS[line]+0.01, scatter_plot_data.MAASTRO[line],
    #             scatter_plot_data.base_model[line], horizontalalignment='left',
    #             size='medium', color='black', weight='semibold')
    ax = plt.plot([0, 100], [0, 100], ls="--", c=".3")
    # plt.set(xlim=(0, 100), ylim=(0, 100))
    plt.xlim(35, 80)
    plt.ylim(35, 80)
    if metric == 'auc':
        plt.title('AUC')
    elif metric == 'mcc':
        plt.title('MCC')
    elif metric == 'f1':
        plt.title('F1 score class 1')
    elif metric == 'f1_0':
        plt.title('F1 score class 0')
    elif metric == 'accuracy':
        plt.title('Accuracy')
    else:
        plt.title('Metric average')

    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.tight_layout(rect=(0, 0.05, 1, 1))
h, labels = p1.get_legend_handles_labels()
labels[0] = 'Data'
labels[4] = 'Models'

l1 = plt.legend(h[1:4], ['All features', 'RENT 1%', 'RENT 50%'], loc='lower left', ncol=4,
                handletextpad=0.2,
                columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-0.8, -0.3))
l2 = plt.legend(h[4:], labels[4:], loc='lower left', ncol=5,
                handletextpad=0.2,
                columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-0.8, -0.4))
p1.add_artist(l1)
# sns.move_legend(plt.gca(), "lower left", title='Dataset',
#                 ncol=3, bbox_to_anchor=(-0.5, -0.5))
plt.savefig('outcome_paper_figures_reviewer/OS_compare_data_RENT.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/OS_compare_data_RENT.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion

SMALL_SIZE = 9
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title


# region Scatter plot DFS
lowest = 35
lower = 54
upper = 79
# lowest = 35
# lower = 60
# upper = 80
select_image = DFS_df.dataset_type == 'D3'
select_tab = DFS_df.image_type.isnull()
select_group = DFS_df.select_group == 'All'
select_all_tab = select_tab & select_group
all_model_data = DFS_df[select_image | select_all_tab].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(5, 7.5))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0', None,  'f1_0']):
    if metric is None:
        prev_plot = p1
        p1.plot([lower, lower, upper, upper, lower], [
                lower, upper, upper, lower, lower], ls="-", c='r', lw=2)
        continue
    plt.subplot(3, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'dataset_type'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         s=40,
                         hue='dataset_type',
                         hue_order=['D1', 'D2', 'D1+D2', 'D3'],
                         style='base_model',
                         style_order=[f'M{i}' for i in range(1, 8)],
                         legend=i == 5)

    ax = plt.plot([0, 100], [0, 100], ls="--", c=".5", zorder=-100)
    if i < 5:
        plt.xlim(lower, upper)
        plt.ylim(lower, upper)
        plt.xticks(ticks=[tick for tick in plt.yticks()[
                   0] if tick >= lower and tick <= upper])
    else:
        plt.xlim(lowest, upper)
        plt.ylim(lowest, upper)
        plt.xticks(ticks=[tick for tick in plt.yticks()[0]
                          if tick >= lowest and tick <= upper])
    if metric == 'auc':
        plt.title('AUC (%)')
    elif metric == 'mcc':
        plt.title('Scaled MCC (%)')
    elif metric == 'f1':
        plt.title('F1 score class 1 (%)')
    elif metric == 'f1_0':
        if i < 5:
            plt.title('F1 score class 0 (%)')
        else:
            plt.title('')

    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.tight_layout(rect=(0, -0.05, 1, 1))
h, labels = p1.get_legend_handles_labels()
labels[0] = 'Input type'
labels[5] = 'Models'

l1 = plt.legend(h[:5], labels[:5], loc='upper left',  # ncol=5,
                # handletextpad=0.2,
                # columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-0.9, 1))
l2 = plt.legend(h[5:], labels[5:], loc='upper left',  # ncol=8,
                # handletextpad=0.2,
                # columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-1.4, 1))
p1.add_artist(l1)

p1.plot([lower, lower, upper - 0.2, upper - 0.2, lower],
        [lower, upper - 0.2, upper - 0.2, lower, lower],
        ls="-", c='r', lw=0.5)
con1 = ConnectionPatch(xyA=(lower, lower), xyB=(lower, upper),
                       coordsA="data", coordsB="data",
                       linestyle=':', linewidth=1,
                       axesA=prev_plot, axesB=p1, color="red")
con2 = ConnectionPatch(xyA=(upper, lower), xyB=(upper, upper),
                       coordsA="data", coordsB="data",
                       linestyle=':', linewidth=1,
                       axesA=prev_plot, axesB=p1, color="red")
p1.add_artist(con1)
p1.add_artist(con2)
# sns.move_legend(plt.gca(), "lower left", title='Dataset',
#                 ncol=3, bbox_to_anchor=(-0.5, -0.5))
plt.savefig('outcome_paper_figures_reviewer/zoom_DFS_compare_data.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/zoom_DFS_compare_data.tiff',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/zoom_DFS_compare_data.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion

# region Scatter plot OS
lowest = 35
lower = 52
upper = 81
# lowest = 35
# lower = 60
# upper = 85
select_image = OS_df.dataset_type == 'D3'
select_tab = OS_df.image_type.isnull()
select_group = OS_df.select_group == 'All'
select_all_tab = select_tab & select_group
all_model_data = OS_df[select_image | select_all_tab].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(5, 7.5))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0', None,  'f1_0']):
    if metric is None:
        prev_plot = p1
        p1.plot([lower, lower, upper, upper, lower], [
                lower, upper, upper, lower, lower], ls="-", c='r', lw=2)
        continue
    plt.subplot(3, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'dataset_type'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         s=40,
                         hue='dataset_type',
                         hue_order=['D1', 'D2', 'D1+D2', 'D3'],
                         style='base_model',
                         style_order=[f'M{i}' for i in range(1, 8)],
                         legend=i == 5)

    ax = plt.plot([0, 100], [0, 100], ls="--", c=".5", zorder=-100)
    if i < 5:
        plt.xlim(lower, upper)
        plt.ylim(lower, upper)
        plt.xticks(ticks=[tick for tick in plt.yticks()[
                   0] if tick >= lower and tick <= upper])
    else:
        plt.xlim(lowest, upper)
        plt.ylim(lowest, upper)
        plt.xticks(ticks=[tick for tick in plt.yticks()[0]
                          if tick >= lowest and tick <= upper])
    if metric == 'auc':
        plt.title('AUC (%)')
    elif metric == 'mcc':
        plt.title('Scaled MCC (%)')
    elif metric == 'f1':
        plt.title('F1 score class 1 (%)')
    elif metric == 'f1_0':
        if i < 5:
            plt.title('F1 score class 0 (%)')
        else:
            plt.title('')
    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.tight_layout(rect=(0, -0.05, 1, 1))
h, labels = p1.get_legend_handles_labels()
labels[0] = 'Input type'
labels[5] = 'Models'
l1 = plt.legend(h[:5], labels[:5], loc='upper left',  # ncol=5,
                # handletextpad=0.2,
                # columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-0.9, 1))
l2 = plt.legend(h[5:], labels[5:], loc='upper left',  # ncol=8,
                # handletextpad=0.2,
                # columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-1.4, 1))
p1.add_artist(l1)
p1.plot([lower, lower, upper - 0.2, upper - 0.2, lower],
        [lower, upper - 0.2, upper - 0.2, lower, lower],
        ls="-", c='r', lw=0.5)
con1 = ConnectionPatch(xyA=(lower, lower), xyB=(lower, upper),
                       coordsA="data", coordsB="data",
                       linestyle=':', linewidth=1,
                       axesA=prev_plot, axesB=p1, color="red")
con2 = ConnectionPatch(xyA=(upper, lower), xyB=(upper, upper),
                       coordsA="data", coordsB="data",
                       linestyle=':', linewidth=1,
                       axesA=prev_plot, axesB=p1, color="red")
p1.add_artist(con1)
p1.add_artist(con2)
plt.savefig('outcome_paper_figures_reviewer/zoom_OS_compare_data.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/zoom_OS_compare_data.tiff',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/zoom_OS_compare_data.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion


# region Scatter plot combine DFS
lowest = 35
lower = 60
upper = 80
selected_names = DFS_df[DFS_df.center == 'MAASTRO'].sort_values(
    'weighted_score', ascending=False).groupby(
        ['image_group', 'image_type']).first()['name'].values
all_model_data = DFS_df[DFS_df.name.isin(
    selected_names)].reset_index(drop=True)
# all_model_data = DFS_df[select_image | select_all_tab].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(5, 6.3))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0']):
    if metric is None:
        prev_plot = p1
        p1.plot([lower, lower, upper, upper, lower], [
                lower, upper, upper, lower, lower], ls="-", c='r', lw=2)
        continue
    plt.subplot(2, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'dataset_type', 'image_group'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         s=40,
                         hue='image_group',
                         hue_order=['+Clinical (D1 + D3)', '+Radiomics (D2 + D3)',
                                    '+Tabular (D1 + D2 + D3)', 'Image models (D3)'],
                         style='base_model',
                         markers=[(4, 0, 0), (4, 1, 0), '^'],
                         style_order=[f'M{i}' for i in range(5, 8)],
                         legend=i == 3)

    ax = plt.plot([0, 100], [0, 100], ls="--", c=".5", zorder=-100)
    if i < 5:
        plt.xlim(lower, upper)
        plt.ylim(lower, upper)
    else:
        plt.xlim(lowest, upper)
        plt.ylim(lowest, upper)
    if metric == 'auc':
        plt.title('AUC (%)')
    elif metric == 'mcc':
        plt.title('Scaled MCC (%)')
    elif metric == 'f1':
        plt.title('F1 score class 1 (%)')
    elif metric == 'f1_0':
        if i < 5:
            plt.title('F1 score class 0 (%)')
        else:
            plt.title('')
    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.tight_layout(rect=(0, 0.15, 1, 1))
h, labels = p1.get_legend_handles_labels()
labels[0] = 'Input group'
labels[5] = 'Models'

input_group_h = [h[0], h[4]] + h[1:4]
input_group_label = ['Input group', 'D3', 'D1+D3', 'D2+D3',
                     'D1+D2+D3']  # [labels[0], labels[4]] + labels[1:4]
l1 = plt.legend(input_group_h, input_group_label, loc='lower left',
                bbox_to_anchor=(-0.5, -0.72))
l2 = plt.legend(h[5:], labels[5:], loc='lower left',
                bbox_to_anchor=(-1.2, -0.72))
p1.add_artist(l1)
# sns.move_legend(plt.gca(), "lower left", title='Dataset',
#                 ncol=3, bbox_to_anchor=(-0.5, -0.5))
plt.savefig('outcome_paper_figures_reviewer/zoom_DFS_compare_data_combine.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/zoom_DFS_compare_data_combine.tiff',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/zoom_DFS_compare_data_combine.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion


# region Scatter plot combine OS
lowest = 35
lower = 60
upper = 85
selected_names = OS_df[OS_df.center == 'MAASTRO'].sort_values(
    'weighted_score', ascending=False).groupby(
        ['image_group', 'image_type']).first()['name'].values
all_model_data = OS_df[OS_df.name.isin(
    selected_names)].reset_index(drop=True)
# all_model_data = DFS_df[select_image | select_all_tab].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(5, 6.3))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0']):
    if metric is None:
        prev_plot = p1
        p1.plot([lower, lower, upper, upper, lower], [
                lower, upper, upper, lower, lower], ls="-", c='r', lw=2)
        continue
    plt.subplot(2, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'dataset_type', 'image_group'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         s=40,
                         hue='image_group',
                         hue_order=['+Clinical (D1 + D3)', '+Radiomics (D2 + D3)',
                                    '+Tabular (D1 + D2 + D3)', 'Image models (D3)'],
                         style='base_model',
                         markers=[(4, 0, 0), (4, 1, 0), '^'],
                         style_order=[f'M{i}' for i in range(5, 8)],
                         legend=i == 3)

    ax = plt.plot([0, 100], [0, 100], ls="--", c=".5", zorder=-100)
    if i < 5:
        plt.xlim(lower, upper)
        plt.ylim(lower, upper)
    else:
        plt.xlim(lowest, upper)
        plt.ylim(lowest, upper)
    if metric == 'auc':
        plt.title('AUC (%)')
    elif metric == 'mcc':
        plt.title('Scaled MCC (%)')
    elif metric == 'f1':
        plt.title('F1 score class 1 (%)')
    elif metric == 'f1_0':
        if i < 5:
            plt.title('F1 score class 0 (%)')
        else:
            plt.title('')
    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.tight_layout(rect=(0, 0.15, 1, 1))
h, labels = p1.get_legend_handles_labels()
labels[0] = 'Input group'
labels[5] = 'Models'

input_group_h = [h[0], h[4]] + h[1:4]
input_group_label = ['Input group', 'D3', 'D1+D3', 'D2+D3',
                     'D1+D2+D3']  # [labels[0], labels[4]] + labels[1:4]
l1 = plt.legend(input_group_h, input_group_label, loc='lower left',
                bbox_to_anchor=(-0.5, -0.72))
l2 = plt.legend(h[5:], labels[5:], loc='lower left',
                bbox_to_anchor=(-1.2, -0.72))
p1.add_artist(l1)
# sns.move_legend(plt.gca(), "lower left", title='Dataset',
#                 ncol=3, bbox_to_anchor=(-0.5, -0.5))
plt.savefig('outcome_paper_figures_reviewer/zoom_OS_compare_data_combine.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/zoom_OS_compare_data_combine.tiff',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/zoom_OS_compare_data_combine.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion


# region Scatter plot DFS clinical RENT
lowest = 40
lower = 58
upper = 80
select_clinical = DFS_df.dataset_type == 'D1'
select_tab = DFS_df.image_type.isnull()
# select_group = DFS_df.select_group == 'All'
# select_all_tab = select_tab & select_group
all_model_data = DFS_df[select_clinical & select_tab].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(5, 6))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0']):
    plt.subplot(2, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'select_group'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         s=40,
                         hue='select_group',
                         hue_order=['All', 'At least once', '50%'],
                         palette=[sns.color_palette()[0], sns.color_palette()
                                  [-1], sns.color_palette()[-3]],
                         style='base_model',
                         style_order=[f'M{i}' for i in range(1, 5)],
                         legend=i == 3)

    ax = plt.plot([0, 100], [0, 100], ls="--", c=".5", zorder=-100)
    if i < 3:
        plt.xlim(lower, upper)
        plt.ylim(lower, upper)
        plt.xticks(ticks=[tick for tick in plt.yticks()[
                   0] if tick >= lower and tick <= upper])
    else:
        plt.xlim(50, 80)
        plt.ylim(40, 70)
        plt.xticks(ticks=np.arange(50., 85., 5.))

    if metric == 'auc':
        plt.title('AUC (%)')
    elif metric == 'mcc':
        plt.title('Scaled MCC (%)')
    elif metric == 'f1':
        plt.title('F1 score class 1 (%)')
    elif metric == 'f1_0':
        plt.title('F1 score class 0 (%)')
    elif metric == 'accuracy':
        plt.title('Accuracy')
    else:
        plt.title('Metric average')

    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.tight_layout(rect=(0, 0.15, 1, 1))
h, labels = p1.get_legend_handles_labels()
labels[0] = 'Data'
labels[4] = 'Models'

l1 = plt.legend(h[1:4], ['All features', 'RENT 1%', 'RENT 50%'], loc='lower left', ncol=4,
                handletextpad=0.2,
                columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-1.1, -0.41))
l2 = plt.legend(h[4:], labels[4:], loc='lower left', ncol=5,
                handletextpad=0.2,
                columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-1.1, -0.55))
p1.add_artist(l1)
# sns.move_legend(plt.gca(), "lower left", title='Dataset',
#                 ncol=3, bbox_to_anchor=(-0.5, -0.5))
plt.savefig('outcome_paper_figures_reviewer/zoom_DFS_compare_data_RENT.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/zoom_DFS_compare_data_RENT.tiff',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/zoom_DFS_compare_data_RENT.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion

# region Scatter plot OS clinical RENT
lowest = 40
lower = 58
upper = 80
select_clinical = OS_df.dataset_type == 'D1'
select_tab = OS_df.image_type.isnull()
# select_group = DFS_df.select_group == 'All'
# select_all_tab = select_tab & select_group
all_model_data = OS_df[select_clinical & select_tab].reset_index(drop=True)

all_model_data['mcc'] = (all_model_data['mcc'] / 2) + 0.5
all_model_data = all_model_data.round(2)
all_model_data['auc'] = (all_model_data['auc'] * 100).astype(int)
all_model_data['mcc'] = (all_model_data['mcc'] * 100).astype(int)
all_model_data['f1'] = (all_model_data['f1'] * 100).astype(int)
all_model_data['f1_0'] = (all_model_data['f1_0'] * 100).astype(int)
all_model_data['accuracy'] = (
    all_model_data['accuracy'] * 100).astype(int)
all_model_data['avg_score'] = (all_model_data['mcc'] +
                               all_model_data['auc'] +
                               all_model_data['accuracy'] +
                               all_model_data['f1'] +
                               all_model_data['f1_0']) / 5

plt.figure(figsize=(5, 6))
for i, metric in enumerate(['auc', 'mcc', 'f1', 'f1_0']):
    plt.subplot(2, 2, i+1)
    scatter_plot_data = all_model_data.pivot_table(
        values=metric,
        index=['base_model', 'select_group'],
        columns='center').reset_index()
    p1 = sns.scatterplot(data=scatter_plot_data, x='OUS', y='MAASTRO',
                         s=40,
                         hue='select_group',
                         hue_order=['All', 'At least once', '50%'],
                         palette=[sns.color_palette()[0], sns.color_palette()
                                  [-1], sns.color_palette()[-3]],
                         style='base_model',
                         style_order=[f'M{i}' for i in range(1, 5)],
                         legend=i == 3)

    ax = plt.plot([0, 100], [0, 100], ls="--", c=".5", zorder=-100)
    if i < 3:
        plt.xlim(lower, upper)
        plt.ylim(lower, upper)
        plt.xticks(ticks=[tick for tick in plt.yticks()[
                   0] if tick >= lower and tick <= upper])
    else:
        plt.xlim(50, 80)
        plt.ylim(40, 70)
        plt.xticks(ticks=np.arange(50., 85., 5.))
    if metric == 'auc':
        plt.title('AUC (%)')
    elif metric == 'mcc':
        plt.title('Scaled MCC (%)')
    elif metric == 'f1':
        plt.title('F1 score class 1 (%)')
    elif metric == 'f1_0':
        plt.title('F1 score class 0 (%)')
    elif metric == 'accuracy':
        plt.title('Accuracy')
    else:
        plt.title('Metric average')

    # Set x-axis label
    plt.xlabel('OUS')
    # Set y-axis label
    plt.ylabel('MAASTRO')
plt.tight_layout(rect=(0, 0.15, 1, 1))
h, labels = p1.get_legend_handles_labels()
labels[0] = 'Data'
labels[4] = 'Models'

l1 = plt.legend(h[1:4], ['All features', 'RENT 1%', 'RENT 50%'], loc='lower left', ncol=4,
                handletextpad=0.2,
                columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-1.1, -0.41))
l2 = plt.legend(h[4:], labels[4:], loc='lower left', ncol=5,
                handletextpad=0.2,
                columnspacing=0.7,
                # frameon=False,
                bbox_to_anchor=(-1.1, -0.55))
p1.add_artist(l1)
# sns.move_legend(plt.gca(), "lower left", title='Dataset',
#                 ncol=3, bbox_to_anchor=(-0.5, -0.5))
plt.savefig('outcome_paper_figures_reviewer/zoom_OS_compare_data_RENT.pdf',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/zoom_OS_compare_data_RENT.tiff',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.savefig('outcome_paper_figures_reviewer/zoom_OS_compare_data_RENT.png',
            edgecolor='black', dpi=300, facecolor='white', transparent=True)
plt.show()
# endregion
