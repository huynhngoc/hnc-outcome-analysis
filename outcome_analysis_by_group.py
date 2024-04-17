import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, matthews_corrcoef, roc_auc_score, roc_curve, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.metrics import AUC

auc_class = AUC()


def specificity(y_true, y_pred):
    TN = ((1 - y_true) * (1 - y_pred)).sum()
    # negative_pred = (1 - y_pred).sum()
    # return TN / negative_pred
    return TN / (1 - y_true).sum()


def auc_score(y_true, y_pred):
    auc_class.reset_state()
    return auc_class(tf.convert_to_tensor(y_true),
                     tf.convert_to_tensor(y_pred)).numpy()


df_test_image = pd.read_csv(
    'outcome_ensemble_results/outcome_test_result_image_per_fold.csv')
df_test_concat_image = pd.read_csv(
    'outcome_ensemble_results/outcome_test_result_image_concat.csv')
df_test_concat_tab_full = pd.read_csv(
    'outcome_ensemble_results/outcome_test_result_tab_full_concat.csv')
df_test_concat_tab_selected = pd.read_csv(
    'outcome_ensemble_results/outcome_test_result_tab_selected_concat.csv')
df_test_comb_tab_full_image = pd.read_csv(
    'outcome_ensemble_results/outcome_test_result_comb_full_tab_image.csv')
df_test_comb_tab_selected_image = pd.read_csv(
    'outcome_ensemble_results/outcome_test_result_comb_selected_tab_image.csv')


df_maastro_image = pd.read_csv(
    'outcome_ensemble_results/outcome_maastro_result_image_per_fold.csv')
df_maastro_concat_image = pd.read_csv(
    'outcome_ensemble_results/outcome_maastro_result_image_concat.csv')
df_maastro_concat_tab_full = pd.read_csv(
    'outcome_ensemble_results/outcome_maastro_result_tab_full_concat.csv')
df_maastro_concat_tab_selected = pd.read_csv(
    'outcome_ensemble_results/outcome_maastro_result_tab_selected_concat.csv')
df_maastro_comb_tab_full_image = pd.read_csv(
    'outcome_ensemble_results/outcome_maastro_result_comb_full_tab_image.csv')
df_maastro_comb_tab_selected_image = pd.read_csv(
    'outcome_ensemble_results/outcome_maastro_result_comb_selected_tab_image.csv')

df_test_image[['endpoint', 'name', 'test_fold']].value_counts()

df_test_concat_image[['endpoint', 'name']].value_counts()

df_maastro_image[['endpoint', 'name', 'test_fold']].value_counts()

df_maastro_concat_tab_full[['endpoint', 'name']].value_counts()

names = ['clinical_basic_nn', 'clinical_interaction_nn', 'clinical_LogisticRegression',
         'clinical_RandomForestClassifier',  'CT_PET', 'CT_PET_T', 'CT_PET_T_N']
endpoints = ['DFS', 'OS']

df_test_image.pid.value_counts()

image_name_group = ['CT_PET', 'CT_PET_T', 'CT_PET_T_N']

tab_input_group = ['clinical',
                   'radiomics',
                   'tabular']
model_group = ['basic_nn', 'interaction_nn',
               'LogisticRegression', 'RandomForestClassifier']

tab_model_names = ['_'.join([n1, n2])
                   for n1 in tab_input_group for n2 in model_group]

tab_image_combine_name = ['_'.join([n1, n2])
                          for n1 in tab_model_names for n2 in image_name_group]

selected_tab_input_group = [
    'clinical_selected_90',
    'clinical_selected_50',
    'clinical_selected_10',
    'clinical_selected_0',
    'radiomics_selected_90',
    'radiomics_selected_50',
    'radiomics_selected_10',
    'radiomics_selected_0',
    'tabular_selected_90',
    'tabular_selected_50',
    'tabular_selected_10',
    'tabular_selected_0'
]
selected_tab_model_names = ['_'.join([n1, n2])
                            for n1 in selected_tab_input_group for n2 in model_group]
selected_tab_image_combine_name = ['_'.join([n1, n2])
                                   for n1 in selected_tab_model_names for n2 in image_name_group]


def print_metric(y_true, y_predicted):
    print(auc_score(y_true, y_predicted))
    print(0.5 + matthews_corrcoef(y_true, (y_predicted > 0.5).astype(int)) / 2)
    print(f1_score(y_true, (y_predicted > 0.5).astype(int)))
    print(f1_score(1 - y_true, 1 - (y_predicted > 0.5).astype(int)))
    print(precision_score(y_true, (y_predicted > 0.5).astype(int)))
    print(recall_score(y_true, (y_predicted > 0.5).astype(int)))
    print(specificity(y_true, (y_predicted > 0.5).astype(int)))


for name in image_name_group:
    print(name)
    data = df_test_concat_image[df_test_concat_image.endpoint ==
                                'DFS'][df_test_concat_image.name == name].reset_index(drop=True)
    ous_info = pd.read_csv('outcome_data_tmp/OUS_D1.csv')

    merge_data = pd.merge(data, ous_info, left_on='pid', right_on='patient_id')
    sns.stripplot(merge_data, hue='y', y='predicted', x='oropharynx')
    plt.show()

    oropharyx_selected = merge_data['oropharynx'] == 1
    hpv_related = merge_data['hpv_related'] == 1

    y_true = merge_data[oropharyx_selected]['y']
    y_predicted = merge_data[oropharyx_selected]['predicted']

    print('oropharynx')
    print_metric(y_true, y_predicted)

    y_true = merge_data[~oropharyx_selected]['y']
    y_predicted = merge_data[~oropharyx_selected]['predicted']

    print('others')
    print_metric(y_true, y_predicted)

    print('all')
    print_metric(merge_data['y'], merge_data['predicted'])

    data = df_maastro_concat_image[df_maastro_concat_image.endpoint ==
                                   'DFS'][df_maastro_concat_image.name == name].reset_index(drop=True)
    ous_info = pd.read_csv('outcome_data_tmp/MAASTRO_D1.csv')

    merge_data = pd.merge(data, ous_info, left_on='pid', right_on='patient_id')
    sns.stripplot(merge_data, hue='y', y='predicted', x='oropharynx')
    plt.show()

    oropharyx_selected = merge_data['oropharynx'] == 1
    hpv_related = merge_data['hpv_related'] == 1

    y_true = merge_data[oropharyx_selected]['y']
    y_predicted = merge_data[oropharyx_selected]['predicted']

    print('oropharynx')
    print_metric(y_true, y_predicted)

    y_true = merge_data[~oropharyx_selected]['y']
    y_predicted = merge_data[~oropharyx_selected]['predicted']

    print('others')
    print_metric(y_true, y_predicted)

    print('all')
    print_metric(merge_data['y'], merge_data['predicted'])


for name in image_name_group:
    print(name)
    data = df_test_concat_image[df_test_concat_image.endpoint ==
                                'DFS'][df_test_concat_image.name == name].reset_index(drop=True)
    ous_info = pd.read_csv('outcome_data_tmp/OUS_D1.csv')

    merge_data = pd.merge(data, ous_info, left_on='pid', right_on='patient_id')
    sns.stripplot(merge_data, hue='y', y='predicted', x='hpv_related')
    plt.show()

    oropharyx_selected = merge_data['oropharynx'] == 1
    hpv_related = merge_data['hpv_related'] == 1

    y_true = merge_data[hpv_related]['y']
    y_predicted = merge_data[hpv_related]['predicted']

    print('hpv_related')
    print_metric(y_true, y_predicted)

    y_true = merge_data[~hpv_related]['y']
    y_predicted = merge_data[~hpv_related]['predicted']

    print('others')
    print_metric(y_true, y_predicted)

    data = df_maastro_concat_image[df_maastro_concat_image.endpoint ==
                                   'DFS'][df_maastro_concat_image.name == name].reset_index(drop=True)
    ous_info = pd.read_csv('outcome_data_tmp/MAASTRO_D1.csv')

    merge_data = pd.merge(data, ous_info, left_on='pid', right_on='patient_id')
    sns.stripplot(merge_data, hue='y', y='predicted', x='hpv_related')
    plt.show()

    oropharyx_selected = merge_data['oropharynx'] == 1
    hpv_related = merge_data['hpv_related'] == 1

    y_true = merge_data[hpv_related]['y']
    y_predicted = merge_data[hpv_related]['predicted']

    print('hpv_related')
    print_metric(y_true, y_predicted)

    y_true = merge_data[~hpv_related]['y']
    y_predicted = merge_data[~hpv_related]['predicted']

    print('others')
    print_metric(y_true, y_predicted)

for name in tab_model_names:
    print(name)
    data = df_test_concat_tab_full[df_test_concat_tab_full.endpoint ==
                                   'DFS'][df_test_concat_tab_full.name == name].reset_index(drop=True)
    ous_info = pd.read_csv('outcome_data_tmp/OUS_D1.csv')

    merge_data = pd.merge(data, ous_info, left_on='pid', right_on='patient_id')
    sns.stripplot(merge_data, hue='y', y='predicted', x='oropharynx')
    plt.show()

    oropharyx_selected = merge_data['oropharynx'] == 1
    hpv_related = merge_data['hpv_related'] == 1

    y_true = merge_data[oropharyx_selected]['y']
    y_predicted = merge_data[oropharyx_selected]['predicted']

    print('oropharynx')
    print_metric(y_true, y_predicted)

    y_true = merge_data[~oropharyx_selected]['y']
    y_predicted = merge_data[~oropharyx_selected]['predicted']

    print('others')
    print_metric(y_true, y_predicted)

    data = df_maastro_concat_tab_full[df_maastro_concat_tab_full.endpoint ==
                                      'DFS'][df_maastro_concat_tab_full.name == name].reset_index(drop=True)
    ous_info = pd.read_csv('outcome_data_tmp/MAASTRO_D1.csv')

    merge_data = pd.merge(data, ous_info, left_on='pid', right_on='patient_id')
    sns.stripplot(merge_data, hue='y', y='predicted', x='oropharynx')
    plt.show()

    oropharyx_selected = merge_data['oropharynx'] == 1
    hpv_related = merge_data['hpv_related'] == 1

    y_true = merge_data[oropharyx_selected]['y']
    y_predicted = merge_data[oropharyx_selected]['predicted']

    print('oropharynx')
    print_metric(y_true, y_predicted)

    y_true = merge_data[~oropharyx_selected]['y']
    y_predicted = merge_data[~oropharyx_selected]['predicted']

    print('others')
    print_metric(y_true, y_predicted)


for name in tab_model_names:
    print(name)
    data = df_test_concat_tab_full[df_test_concat_tab_full.endpoint ==
                                   'DFS'][df_test_concat_tab_full.name == name].reset_index(drop=True)
    ous_info = pd.read_csv('outcome_data_tmp/OUS_D1.csv')

    merge_data = pd.merge(data, ous_info, left_on='pid', right_on='patient_id')
    sns.stripplot(merge_data, hue='y', y='predicted', x='hpv_related')
    plt.show()

    oropharyx_selected = merge_data['oropharynx'] == 1
    hpv_related = merge_data['hpv_related'] == 1

    y_true = merge_data[hpv_related]['y']
    y_predicted = merge_data[hpv_related]['predicted']

    print('hpv_related')
    print_metric(y_true, y_predicted)

    y_true = merge_data[~hpv_related]['y']
    y_predicted = merge_data[~hpv_related]['predicted']

    print('others')
    print_metric(y_true, y_predicted)

    data = df_maastro_concat_tab_full[df_maastro_concat_tab_full.endpoint ==
                                      'DFS'][df_maastro_concat_tab_full.name == name].reset_index(drop=True)
    ous_info = pd.read_csv('outcome_data_tmp/MAASTRO_D1.csv')

    merge_data = pd.merge(data, ous_info, left_on='pid', right_on='patient_id')
    sns.stripplot(merge_data, hue='y', y='predicted', x='hpv_related')
    plt.show()

    oropharyx_selected = merge_data['oropharynx'] == 1
    hpv_related = merge_data['hpv_related'] == 1

    y_true = merge_data[hpv_related]['y']
    y_predicted = merge_data[hpv_related]['predicted']

    print('hpv_related')
    print_metric(y_true, y_predicted)

    y_true = merge_data[~hpv_related]['y']
    y_predicted = merge_data[~hpv_related]['predicted']

    print('others')
    print_metric(y_true, y_predicted)


data = df_test_concat_tab_full[df_test_concat_tab_full.endpoint ==
                               'OS'][df_test_concat_tab_full.name == name].reset_index(drop=True)
ous_info = pd.read_csv('outcome_data_tmp/OUS_D1.csv')

merge_data = pd.merge(data, ous_info, left_on='pid', right_on='patient_id')
sns.stripplot(merge_data, hue='y', y='predicted', x='oropharynx')
plt.show()


oropharyx_selected = merge_data['oropharynx'] == 1
hpv_related = merge_data['hpv_related'] == 1

y_true = merge_data[oropharyx_selected & hpv_related]['y']
y_predicted = merge_data[oropharyx_selected]['predicted']

f1_score(y_true, (y_predicted > 0.5).astype(int))
f1_score(1 - y_true, 1 - (y_predicted > 0.5).astype(int))
auc_score(y_true, y_predicted)

y_true = merge_data[~hpv_related]['y']
y_predicted = merge_data[~oropharyx_selected]['predicted']

f1_score(y_true, (y_predicted > 0.5).astype(int))
f1_score(1 - y_true, 1 - (y_predicted > 0.5).astype(int))
auc_score(y_true, y_predicted)


data = df_maastro_concat_tab_full[df_maastro_concat_tab_full.endpoint ==
                                  'OS'][df_maastro_concat_tab_full.name == 'radiomics_RandomForestClassifier'].reset_index(drop=True)
ous_info = pd.read_csv('outcome_data_tmp/MAASTRO_D1.csv')

merge_data = pd.merge(data, ous_info, left_on='pid', right_on='patient_id')
sns.stripplot(merge_data, hue='y', y='predicted', x='oropharynx')
plt.show()

oropharyx_selected = merge_data['oropharynx'] == 1
hpv_related = merge_data['hpv_related'] == 1

y_true = merge_data[hpv_related]['y']
y_predicted = merge_data[oropharyx_selected]['predicted']

f1_score(y_true, (y_predicted > 0.5).astype(int))
f1_score(1 - y_true, 1 - (y_predicted > 0.5).astype(int))
auc_score(y_true, y_predicted)

y_true = merge_data[~hpv_related]['y']
y_predicted = merge_data[~oropharyx_selected]['predicted']

f1_score(y_true, (y_predicted > 0.5).astype(int))
f1_score(1 - y_true, 1 - (y_predicted > 0.5).astype(int))
auc_score(y_true, y_predicted)
