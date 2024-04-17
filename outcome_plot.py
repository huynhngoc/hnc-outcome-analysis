import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, matthews_corrcoef, roc_auc_score, roc_curve, f1_score
import tensorflow as tf
from tensorflow.keras.metrics import AUC

auc_class = AUC()


def auc_score(y_true, y_pred):
    auc_class.reset_state()
    return auc_class(tf.convert_to_tensor(y_true),
                     tf.convert_to_tensor(y_pred)).numpy()


df_test_res_all = pd.read_csv('outcom_test_result_per_fold.csv')
df_test_concat_res_all = pd.read_csv(
    'old_ml_results/outcom_test_result_concat.csv')
df_test_concat_combine = pd.read_csv('outcom_test_results_concat_comb.csv')
df_test_concat_additional = pd.read_csv(
    'outcom_test_result_additional_concat.csv')
df_test_concat_combine_additional = pd.read_csv(
    'outcom_test_results_concat_comb_additional.csv')
df_maastro_res_all = pd.read_csv(
    'old_ml_results/outcom_maastro_result_per_fold.csv')
df_maastro_concat_res_all = pd.read_csv(
    'old_ml_results/outcom_maastro_result_concat.csv')
df_maastro_concat_combine = pd.read_csv(
    'outcom_maastro_results_concat_comb.csv')
df_maastro_concat_additional = pd.read_csv(
    'outcom_maastro_result_additional_concat.csv')
df_maastro_concat_combine_additional = pd.read_csv(
    'outcom_maastro_results_concat_comb_additional.csv')

df_test_res_all[['endpoint', 'name', 'test_fold']].value_counts()

df_test_concat_res_all[['endpoint', 'name']].value_counts()

df_maastro_res_all[['endpoint', 'name', 'test_fold']].value_counts()

df_maastro_concat_res_all[['endpoint', 'name']].value_counts()
df_maastro_concat_additional[['endpoint', 'name']].value_counts()

names = ['clinical', 'clinical interaction', 'clinical_LogisticRegression',
         'clinical_RandomForestClassifier',  'CT_PET', 'CT_PET_T', 'CT_PET_T_N']
endpoints = ['DFS', 'OS']

# region Print out metrics scores

for endpoint in endpoints:
    # print(endpoint)
    for name in names:
        # print(name)
        # print('AUC')
        for fold in range(5):
            selected_endpoint = df_test_res_all.endpoint == endpoint
            selected_name = df_test_res_all.name == name
            selected_fold = df_test_res_all.test_fold == fold
            selected = selected_endpoint & selected_name & selected_fold
            y_true = df_test_res_all[selected]['y']
            y_pred = df_test_res_all[selected]['predicted']
            y_pred_binarized = (y_pred > 0.5).astype(int)
            print(auc_score(y_true, y_pred))


for endpoint in endpoints:
    # print(endpoint)
    for name in names:
        # print(name)
        # print('AUC')
        for fold in range(5):
            selected_endpoint = df_maastro_res_all.endpoint == endpoint
            selected_name = df_maastro_res_all.name == name
            selected_fold = df_maastro_res_all.test_fold == fold
            selected = selected_endpoint & selected_name & selected_fold
            y_true = df_maastro_res_all[selected]['y']
            y_pred = df_maastro_res_all[selected]['predicted']
            y_pred_binarized = (y_pred > 0.5).astype(int)
            print(auc_score(y_true, y_pred))


for endpoint in endpoints:
    # print(endpoint)
    for name in names:
        # print(name)
        # print('AUC')
        selected_endpoint = df_test_concat_res_all.endpoint == endpoint
        selected_name = df_test_concat_res_all.name == name
        selected = selected_endpoint & selected_name
        y_true = df_test_concat_res_all[selected]['y']
        y_pred = df_test_concat_res_all[selected]['predicted']
        y_pred_binarized = (y_pred > 0.5).astype(int)
        print(f1_score(1-y_true, 1-y_pred_binarized))


for endpoint in endpoints:
    # print(endpoint)
    for name in names:
        # print(name)
        # print('AUC')
        selected_endpoint = df_maastro_concat_res_all.endpoint == endpoint
        selected_name = df_maastro_concat_res_all.name == name
        selected = selected_endpoint & selected_name
        y_true = df_maastro_concat_res_all[selected]['y']
        y_pred = df_maastro_concat_res_all[selected]['predicted']
        y_pred_binarized = (y_pred > 0.5).astype(int)
        print(accuracy_score(y_true, y_pred_binarized))

# endregion

name_group_1 = ['clinical', 'clinical interaction', 'clinical_LogisticRegression',
                'clinical_RandomForestClassifier']
name_group_2 = ['CT_PET', 'CT_PET_T', 'CT_PET_T_N']

combine_names = ['_'.join([n1, n2])
                 for n1 in name_group_1 for n2 in name_group_2]


additional_input_group = ['clinical_selected_90',
                          'radiomics',
                          'radiomics_selected_90',
                          'tabular',
                          'tabular_selected_90',
                          'tabular_selected_50']
additional_model_group = ['basic_nn', 'interaction_nn',
                          'LogisticRegression', 'RandomForestClassifier']
additional_names = ['_'.join([n1, n2])
                    for n1 in additional_input_group for n2 in additional_model_group]


additional_combine_names = ['_'.join([n1, n2])
                            for n1 in additional_names for n2 in name_group_2]


# region Ensembling clinical and Image model results

# # test concat results
# test_concat_dfs = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name_1 in name_group_1:
#         for name_2 in name_group_2:
#             name = '_'.join([name_1, name_2])
#             print(name)
#             selected_endpoint = df_test_concat_res_all.endpoint == endpoint
#             selected_name_1 = df_test_concat_res_all.name == name_1
#             selected_1 = selected_endpoint & selected_name_1

#             selected_name_2 = df_test_concat_res_all.name == name_2
#             selected_2 = selected_endpoint & selected_name_2
#             y_true = df_test_concat_res_all[selected_1]['y'].values
#             pids = df_test_concat_res_all[selected_1]['pid'].values
#             y_pred_1 = df_test_concat_res_all[selected_1]['predicted'].values
#             y_pred_2 = df_test_concat_res_all[selected_2]['predicted'].values
#             y_pred = (y_pred_1 + y_pred_2) / 2

#             test_concat_dfs.append(pd.DataFrame({
#                 'endpoint': [endpoint] * len(pids),
#                 'pid': pids,
#                 'name': [name] * len(pids),
#                 'predicted': y_pred,
#                 'y': y_true
#             }))

# pd.concat(test_concat_dfs).to_csv('outcom_test_results_concat_comb.csv', index=False)


# # test concat results additional
# test_concat_additional_dfs = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name_1 in additional_names:
#         for name_2 in name_group_2:
#             name = '_'.join([name_1, name_2])
#             print(name)
#             selected_endpoint_1 = df_test_concat_additional.endpoint == endpoint
#             selected_name_1 = df_test_concat_additional.name == name_1
#             selected_1 = selected_endpoint_1 & selected_name_1

#             selected_endpoint_2 = df_test_concat_res_all.endpoint == endpoint
#             selected_name_2 = df_test_concat_res_all.name == name_2
#             selected_2 = selected_endpoint_2 & selected_name_2
#             y_true = df_test_concat_res_all[selected_2]['y'].values
#             pids = df_test_concat_res_all[selected_2]['pid'].values
#             assert np.all(pids == df_test_concat_additional[selected_1]['pid'].values)
#             assert np.all(y_true == df_test_concat_additional[selected_1]['y'].values)
#             y_pred_1 = df_test_concat_additional[selected_1]['predicted'].values
#             y_pred_2 = df_test_concat_res_all[selected_2]['predicted'].values
#             y_pred = (y_pred_1 + y_pred_2) / 2
#             test_concat_additional_dfs.append(pd.DataFrame({
#                 'endpoint': [endpoint] * len(pids),
#                 'pid': pids,
#                 'name': [name] * len(pids),
#                 'predicted': y_pred,
#                 'y': y_true
#             }))

# pd.concat(test_concat_additional_dfs).to_csv(
#     'outcom_test_results_concat_comb_additional.csv', index=False)


# # maastro concat results
# maastro_concat_dfs = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name_1 in name_group_1:
#         for name_2 in name_group_2:
#             name = '_'.join([name_1, name_2])
#             print(name)
#             selected_endpoint = df_maastro_concat_res_all.endpoint == endpoint
#             selected_name_1 = df_maastro_concat_res_all.name == name_1
#             selected_1 = selected_endpoint & selected_name_1

#             selected_name_2 = df_maastro_concat_res_all.name == name_2
#             selected_2 = selected_endpoint & selected_name_2
#             y_true = df_maastro_concat_res_all[selected_1]['y'].values
#             pids = df_maastro_concat_res_all[selected_1]['pid'].values
#             y_pred_1 = df_maastro_concat_res_all[selected_1]['predicted'].values
#             y_pred_2 = df_maastro_concat_res_all[selected_2]['predicted'].values
#             y_pred = (y_pred_1 + y_pred_2) / 2

#             maastro_concat_dfs.append(pd.DataFrame({
#                 'endpoint': [endpoint] * len(pids),
#                 'pid': pids,
#                 'name': [name] * len(pids),
#                 'predicted': y_pred,
#                 'y': y_true
#             }))

# pd.concat(maastro_concat_dfs).to_csv('outcom_maastro_results_concat_comb.csv', index=False)


# # maastro concat results additional
# maastro_concat_additional_dfs = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name_1 in additional_names:
#         for name_2 in name_group_2:
#             name = '_'.join([name_1, name_2])
#             print(name)
#             selected_endpoint_1 = df_maastro_concat_additional.endpoint == endpoint
#             selected_name_1 = df_maastro_concat_additional.name == name_1
#             selected_1 = selected_endpoint_1 & selected_name_1

#             selected_endpoint_2 = df_maastro_concat_res_all.endpoint == endpoint
#             selected_name_2 = df_maastro_concat_res_all.name == name_2
#             selected_2 = selected_endpoint_2 & selected_name_2

#             y_true = df_maastro_concat_res_all[selected_2]['y'].values
#             pids = df_maastro_concat_res_all[selected_2]['pid'].values
#             assert np.all(pids == df_maastro_concat_additional[selected_1]['pid'].values)
#             assert np.all(y_true == df_maastro_concat_additional[selected_1]['y'].values)
#             y_pred_1 = df_maastro_concat_additional[selected_1]['predicted'].values
#             y_pred_2 = df_maastro_concat_res_all[selected_2]['predicted'].values
#             y_pred = (y_pred_1 + y_pred_2) / 2

#             maastro_concat_additional_dfs.append(pd.DataFrame({
#                 'endpoint': [endpoint] * len(pids),
#                 'pid': pids,
#                 'name': [name] * len(pids),
#                 'predicted': y_pred,
#                 'y': y_true
#             }))

# pd.concat(maastro_concat_additional_dfs).to_csv(
#     'outcom_maastro_results_concat_comb_additional.csv', index=False)


# endregion Ensembling clinical and Image model results


# adjusting the score with same ratio for comparison
# resampling on different ratio of class: 25, 40, 50, 60, 75

# region Generating bootstrap resampling data
# resampled_test_data = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name in names:
#         print(name)
#         # print('AUC')
#         for fold in range(5):
#             print('fold', fold)
#             selected_endpoint = df_test_res_all.endpoint == endpoint
#             selected_name = df_test_res_all.name == name
#             selected_fold = df_test_res_all.test_fold == fold
#             selected = selected_endpoint & selected_name & selected_fold
#             y_true = df_test_res_all[selected]['y'].values
#             y_pred = df_test_res_all[selected]['predicted'].values
#             y_pred_binarized = (y_pred > 0.5).astype(int)
#             class_1_index = np.argwhere(y_true > 0).flatten()
#             class_0_index = np.argwhere(y_true == 0).flatten()
#             for class_1_size in [7, 11, 14, 17, 21]:
#                 for _ in range(1000):
#                     indice = [*np.random.choice(class_1_index, class_1_size),
#                               *np.random.choice(class_0_index, 28 - class_1_size)]
#                     resampled_test_data.append({
#                         'endpoint': endpoint,
#                         'name': name,
#                         'test_fold': fold,
#                         'class_1_ratio': np.round(class_1_size / 28, 2),
#                         'auc': auc_score(y_true[indice], y_pred[indice]),
#                         'f1': f1_score(y_true[indice], y_pred_binarized[indice]),
#                         'f1_0': f1_score(1 - y_true[indice], 1-y_pred_binarized[indice]),
#                         'mcc': matthews_corrcoef(y_true[indice], y_pred_binarized[indice]),
#                         'accuracy': accuracy_score(y_true[indice], y_pred_binarized[indice]),
#                     })
# resample_test_df = pd.DataFrame(resampled_test_data)
# resample_test_df.to_csv(
#     '../../outcome_test_res_per_fold_ratio.csv', index=False)


# resampled_test_concat_data = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name in names:
#         print(name)
#         selected_endpoint = df_test_concat_res_all.endpoint == endpoint
#         selected_name = df_test_concat_res_all.name == name
#         selected = selected_endpoint & selected_name
#         y_true = df_test_concat_res_all[selected]['y'].values
#         y_pred = df_test_concat_res_all[selected]['predicted'].values
#         y_pred_binarized = (y_pred > 0.5).astype(int)
#         class_1_index = np.argwhere(y_true > 0).flatten()
#         class_0_index = np.argwhere(y_true == 0).flatten()
#         for class_1_size in [35, 56, 70, 84, 105]:
#             for _ in range(1000):
#                 indice = [*np.random.choice(class_1_index, class_1_size),
#                           *np.random.choice(class_0_index, 140 - class_1_size)]
#                 resampled_test_concat_data.append({
#                     'endpoint': endpoint,
#                     'name': name,
#                     'class_1_ratio': np.round(class_1_size / 140, 2),
#                     'auc': auc_score(y_true[indice], y_pred[indice]),
#                     'f1': f1_score(y_true[indice], y_pred_binarized[indice]),
#                     'f1_0': f1_score(1 - y_true[indice], 1-y_pred_binarized[indice]),
#                     'mcc': matthews_corrcoef(y_true[indice], y_pred_binarized[indice]),
#                     'accuracy': accuracy_score(y_true[indice], y_pred_binarized[indice]),
#                 })

# resample_test_concat_df = pd.DataFrame(resampled_test_concat_data)
# resample_test_concat_df.to_csv(
#     '../../outcome_test_res_concat_per_ratio.csv', index=False)

# resampled_test_concat_data_combine = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name in ['_'.join([n1, n2]) for n1 in name_group_1 for n2 in name_group_2]:
#         print(name)
#         selected_endpoint = df_test_concat_combine.endpoint == endpoint
#         selected_name = df_test_concat_combine.name == name
#         selected = selected_endpoint & selected_name
#         y_true = df_test_concat_combine[selected]['y'].values
#         y_pred = df_test_concat_combine[selected]['predicted'].values
#         y_pred_binarized = (y_pred > 0.5).astype(int)
#         class_1_index = np.argwhere(y_true > 0).flatten()
#         class_0_index = np.argwhere(y_true == 0).flatten()
#         for class_1_size in [35, 56, 70, 84, 105]:
#             for _ in range(1000):
#                 indice = [*np.random.choice(class_1_index, class_1_size),
#                           *np.random.choice(class_0_index, 140 - class_1_size)]
#                 resampled_test_concat_data_combine.append({
#                     'endpoint': endpoint,
#                     'name': name,
#                     'class_1_ratio': np.round(class_1_size / 140, 2),
#                     'auc': auc_score(y_true[indice], y_pred[indice]),
#                     'f1': f1_score(y_true[indice], y_pred_binarized[indice]),
#                     'f1_0': f1_score(1 - y_true[indice], 1-y_pred_binarized[indice]),
#                     'mcc': matthews_corrcoef(y_true[indice], y_pred_binarized[indice]),
#                     'accuracy': accuracy_score(y_true[indice], y_pred_binarized[indice]),
#                 })

# resample_test_concat_combine_df = pd.DataFrame(resampled_test_concat_data_combine)
# resample_test_concat_combine_df.to_csv(
#     '../../outcome_test_res_concat_per_ratio_comb.csv', index=False)


# resampled_test_concat_data_additional = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name in additional_names:
#         print(name)
#         selected_endpoint = df_test_concat_additional.endpoint == endpoint
#         selected_name = df_test_concat_additional.name == name
#         selected = selected_endpoint & selected_name
#         y_true = df_test_concat_additional[selected]['y'].values
#         y_pred = df_test_concat_additional[selected]['predicted'].values
#         y_pred_binarized = (y_pred > 0.5).astype(int)
#         class_1_index = np.argwhere(y_true > 0).flatten()
#         class_0_index = np.argwhere(y_true == 0).flatten()
#         for class_1_size in [35, 56, 70, 84, 105]:
#             for _ in range(1000):
#                 indice = [*np.random.choice(class_1_index, class_1_size),
#                           *np.random.choice(class_0_index, 140 - class_1_size)]
#                 resampled_test_concat_data_additional.append({
#                     'endpoint': endpoint,
#                     'name': name,
#                     'class_1_ratio': np.round(class_1_size / 140, 2),
#                     'auc': auc_score(y_true[indice], y_pred[indice]),
#                     'f1': f1_score(y_true[indice], y_pred_binarized[indice]),
#                     'f1_0': f1_score(1 - y_true[indice], 1-y_pred_binarized[indice]),
#                     'mcc': matthews_corrcoef(y_true[indice], y_pred_binarized[indice]),
#                     'accuracy': accuracy_score(y_true[indice], y_pred_binarized[indice]),
#                 })

# resample_test_concat_additional_df = pd.DataFrame(
#     resampled_test_concat_data_additional)
# resample_test_concat_additional_df.to_csv(
#     '../../outcome_test_res_concat_per_ratio_additional.csv', index=False)


# resampled_test_concat_data_combine_additional = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name in additional_combine_names:
#         print(name)
#         selected_endpoint = df_test_concat_combine_additional.endpoint == endpoint
#         selected_name = df_test_concat_combine_additional.name == name
#         selected = selected_endpoint & selected_name
#         y_true = df_test_concat_combine_additional[selected]['y'].values
#         y_pred = df_test_concat_combine_additional[selected]['predicted'].values
#         y_pred_binarized = (y_pred > 0.5).astype(int)
#         class_1_index = np.argwhere(y_true > 0).flatten()
#         class_0_index = np.argwhere(y_true == 0).flatten()
#         for class_1_size in [35, 56, 70, 84, 105]:
#             for _ in range(1000):
#                 indice = [*np.random.choice(class_1_index, class_1_size),
#                           *np.random.choice(class_0_index, 140 - class_1_size)]
#                 resampled_test_concat_data_combine_additional.append({
#                     'endpoint': endpoint,
#                     'name': name,
#                     'class_1_ratio': np.round(class_1_size / 140, 2),
#                     'auc': auc_score(y_true[indice], y_pred[indice]),
#                     'f1': f1_score(y_true[indice], y_pred_binarized[indice]),
#                     'f1_0': f1_score(1 - y_true[indice], 1-y_pred_binarized[indice]),
#                     'mcc': matthews_corrcoef(y_true[indice], y_pred_binarized[indice]),
#                     'accuracy': accuracy_score(y_true[indice], y_pred_binarized[indice]),
#                 })

# resampled_test_concat_data_combine_additional_df = pd.DataFrame(
#     resampled_test_concat_data_combine_additional)
# resampled_test_concat_data_combine_additional_df.to_csv(
#     '../../outcome_test_res_concat_per_ratio_combine_additional.csv', index=False)


# resampled_maastro_data = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name in names:
#         print(name)
#         # print('AUC')
#         for fold in range(5):
#             print('fold', fold)
#             selected_endpoint = df_maastro_res_all.endpoint == endpoint
#             selected_name = df_maastro_res_all.name == name
#             selected_fold = df_maastro_res_all.test_fold == fold
#             selected = selected_endpoint & selected_name & selected_fold
#             y_true = df_maastro_res_all[selected]['y'].values
#             y_pred = df_maastro_res_all[selected]['predicted'].values
#             y_pred_binarized = (y_pred > 0.5).astype(int)
#             class_1_index = np.argwhere(y_true > 0).flatten()
#             class_0_index = np.argwhere(y_true == 0).flatten()
#             for class_1_size in [25, 40, 50, 60, 75]:
#                 for _ in range(1000):
#                     indice = [*np.random.choice(class_1_index, class_1_size),
#                               *np.random.choice(class_0_index, 100 - class_1_size)]
#                     resampled_maastro_data.append({
#                         'endpoint': endpoint,
#                         'name': name,
#                         'test_fold': fold,
#                         'class_1_ratio': np.round(class_1_size / 100, 2),
#                         'auc': auc_score(y_true[indice], y_pred[indice]),
#                         'f1': f1_score(y_true[indice], y_pred_binarized[indice]),
#                         'f1_0': f1_score(1 - y_true[indice], 1-y_pred_binarized[indice]),
#                         'mcc': matthews_corrcoef(y_true[indice], y_pred_binarized[indice]),
#                         'accuracy': accuracy_score(y_true[indice], y_pred_binarized[indice]),
#                     })

# resample_maastro_df = pd.DataFrame(resampled_maastro_data)
# resample_maastro_df.to_csv('../../outcome_maastro_res_per_fold_ratio.csv')

# resampled_maastro_concat_data = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name in names:
#         print(name)
#         # print('AUC')
#         selected_endpoint = df_maastro_concat_res_all.endpoint == endpoint
#         selected_name = df_maastro_concat_res_all.name == name
#         selected = selected_endpoint & selected_name
#         y_true = df_maastro_concat_res_all[selected]['y'].values
#         y_pred = df_maastro_concat_res_all[selected]['predicted'].values
#         y_pred_binarized = (y_pred > 0.5).astype(int)
#         class_1_index = np.argwhere(y_true > 0).flatten()
#         class_0_index = np.argwhere(y_true == 0).flatten()
#         for class_1_size in [25, 40, 50, 60, 75]:
#             for _ in range(1000):
#                 indice = [*np.random.choice(class_1_index, class_1_size),
#                           *np.random.choice(class_0_index, 100 - class_1_size)]
#                 resampled_maastro_concat_data.append({
#                     'endpoint': endpoint,
#                     'name': name,
#                     'class_1_ratio': np.round(class_1_size / 100, 2),
#                     'auc': auc_score(y_true[indice], y_pred[indice]),
#                     'f1': f1_score(y_true[indice], y_pred_binarized[indice]),
#                     'f1_0': f1_score(1 - y_true[indice], 1-y_pred_binarized[indice]),
#                     'mcc': matthews_corrcoef(y_true[indice], y_pred_binarized[indice]),
#                     'accuracy': accuracy_score(y_true[indice], y_pred_binarized[indice]),
#                 })

# resample_maastro_concat_df = pd.DataFrame(resampled_maastro_concat_data)
# resample_maastro_concat_df.to_csv(
#     '../../outcome_maastro_res_concat_per_ratio.csv', index=False)

# resampled_maastro_concat_comb_data = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name in ['_'.join([n1, n2]) for n1 in name_group_1 for n2 in name_group_2]:
#         print(name)
#         # print('AUC')
#         selected_endpoint = df_maastro_concat_combine.endpoint == endpoint
#         selected_name = df_maastro_concat_combine.name == name
#         selected = selected_endpoint & selected_name
#         y_true = df_maastro_concat_combine[selected]['y'].values
#         y_pred = df_maastro_concat_combine[selected]['predicted'].values
#         y_pred_binarized = (y_pred > 0.5).astype(int)
#         class_1_index = np.argwhere(y_true > 0).flatten()
#         class_0_index = np.argwhere(y_true == 0).flatten()
#         for class_1_size in [25, 40, 50, 60, 75]:
#             for _ in range(1000):
#                 indice = [*np.random.choice(class_1_index, class_1_size),
#                           *np.random.choice(class_0_index, 100 - class_1_size)]
#                 resampled_maastro_concat_comb_data.append({
#                     'endpoint': endpoint,
#                     'name': name,
#                     'class_1_ratio': np.round(class_1_size / 100, 2),
#                     'auc': auc_score(y_true[indice], y_pred[indice]),
#                     'f1': f1_score(y_true[indice], y_pred_binarized[indice]),
#                     'f1_0': f1_score(1 - y_true[indice], 1-y_pred_binarized[indice]),
#                     'mcc': matthews_corrcoef(y_true[indice], y_pred_binarized[indice]),
#                     'accuracy': accuracy_score(y_true[indice], y_pred_binarized[indice]),
#                 })

# resample_maastro_concat_comb_df = pd.DataFrame(resampled_maastro_concat_comb_data)
# resample_maastro_concat_comb_df.to_csv(
#     '../../outcome_maastro_res_concat_per_ratio_comb.csv', index=False)


# resampled_maastro_concat_additional_data = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name in additional_names:
#         print(name)
#         # print('AUC')
#         selected_endpoint = df_maastro_concat_additional.endpoint == endpoint
#         selected_name = df_maastro_concat_additional.name == name
#         selected = selected_endpoint & selected_name
#         y_true = df_maastro_concat_additional[selected]['y'].values
#         y_pred = df_maastro_concat_additional[selected]['predicted'].values
#         y_pred_binarized = (y_pred > 0.5).astype(int)
#         class_1_index = np.argwhere(y_true > 0).flatten()
#         class_0_index = np.argwhere(y_true == 0).flatten()
#         for class_1_size in [25, 40, 50, 60, 75]:
#             for _ in range(1000):
#                 indice = [*np.random.choice(class_1_index, class_1_size),
#                           *np.random.choice(class_0_index, 100 - class_1_size)]
#                 resampled_maastro_concat_additional_data.append({
#                     'endpoint': endpoint,
#                     'name': name,
#                     'class_1_ratio': np.round(class_1_size / 100, 2),
#                     'auc': auc_score(y_true[indice], y_pred[indice]),
#                     'f1': f1_score(y_true[indice], y_pred_binarized[indice]),
#                     'f1_0': f1_score(1 - y_true[indice], 1-y_pred_binarized[indice]),
#                     'mcc': matthews_corrcoef(y_true[indice], y_pred_binarized[indice]),
#                     'accuracy': accuracy_score(y_true[indice], y_pred_binarized[indice]),
#                 })

# resample_maastro_concat_additional_df = pd.DataFrame(
#     resampled_maastro_concat_additional_data)
# resample_maastro_concat_additional_df.to_csv(
#     '../../outcome_maastro_res_concat_per_ratio_additional.csv', index=False)


# resampled_maastro_concat_combine_additional_data = []
# for endpoint in endpoints:
#     print(endpoint)
#     for name in additional_combine_names:
#         print(name)
#         # print('AUC')
#         selected_endpoint = df_maastro_concat_combine_additional.endpoint == endpoint
#         selected_name = df_maastro_concat_combine_additional.name == name
#         selected = selected_endpoint & selected_name
#         y_true = df_maastro_concat_combine_additional[selected]['y'].values
#         y_pred = df_maastro_concat_combine_additional[selected]['predicted'].values
#         y_pred_binarized = (y_pred > 0.5).astype(int)
#         class_1_index = np.argwhere(y_true > 0).flatten()
#         class_0_index = np.argwhere(y_true == 0).flatten()
#         for class_1_size in [25, 40, 50, 60, 75]:
#             for _ in range(1000):
#                 indice = [*np.random.choice(class_1_index, class_1_size),
#                           *np.random.choice(class_0_index, 100 - class_1_size)]
#                 resampled_maastro_concat_combine_additional_data.append({
#                     'endpoint': endpoint,
#                     'name': name,
#                     'class_1_ratio': np.round(class_1_size / 100, 2),
#                     'auc': auc_score(y_true[indice], y_pred[indice]),
#                     'f1': f1_score(y_true[indice], y_pred_binarized[indice]),
#                     'f1_0': f1_score(1 - y_true[indice], 1-y_pred_binarized[indice]),
#                     'mcc': matthews_corrcoef(y_true[indice], y_pred_binarized[indice]),
#                     'accuracy': accuracy_score(y_true[indice], y_pred_binarized[indice]),
#                 })

# resampled_maastro_concat_combine_additional_data_df = pd.DataFrame(
#     resampled_maastro_concat_combine_additional_data)
# resampled_maastro_concat_combine_additional_data_df.to_csv(
#     '../../outcome_maastro_res_concat_per_ratio_combine_additional.csv', index=False)


# endregion
# =========================================
resample_test_df = pd.read_csv(
    '../../old_outcome_ratio/outcome_test_res_per_fold_ratio.csv')
# resample_test_df[resample_test_df.name.isin(['CT_PET', 'CT_PET_T', 'CT_PET_T_N'])].to_csv('../../outcome_test_image_per_fold_ratio.csv', index=False)
resample_test_concat_df = pd.read_csv(
    '../../old_outcome_ratio/outcome_test_res_concat_per_ratio.csv')
# resample_test_concat_df[resample_test_concat_df.name.isin(['CT_PET', 'CT_PET_T', 'CT_PET_T_N'])].to_csv('../../outcome_test_image_concat_ratio.csv', index=False)
resample_test_concat_combine_df = pd.read_csv(
    '../../outcome_test_res_concat_per_ratio_comb.csv')
resample_test_concat_additional_df = pd.read_csv(
    '../../outcome_test_res_concat_per_ratio_additional.csv')
resample_test_concat_combine_additional_df = pd.read_csv(
    '../../outcome_test_res_concat_per_ratio_combine_additional.csv')
resample_maastro_df = pd.read_csv(
    '../../old_outcome_ratio/outcome_maastro_res_per_fold_ratio.csv')
# resample_maastro_df[resample_maastro_df.name.isin(['CT_PET', 'CT_PET_T', 'CT_PET_T_N'])].to_csv('../../outcome_maastro_image_per_fold_ratio.csv', index=False)
resample_maastro_concat_df = pd.read_csv(
    '../../old_outcome_ratio/outcome_maastro_res_concat_per_ratio.csv')
# resample_maastro_concat_df[resample_maastro_concat_df.name.isin(['CT_PET', 'CT_PET_T', 'CT_PET_T_N'])].to_csv('../../outcome_maastro_image_concat_ratio.csv', index=False)
resample_maastro_concat_combine_df = pd.read_csv(
    '../../outcome_maastro_res_concat_per_ratio_comb.csv')
resample_maastro_concat_additional_df = pd.read_csv(
    '../../outcome_maastro_res_concat_per_ratio_additional.csv')
resample_maastro_concat_combine_additional_df = pd.read_csv(
    '../../outcome_maastro_res_concat_per_ratio_combine_additional.csv')

# region Print out bootstrap sampling results
ratio = 0.5
col_name = 'f1_0'
cur_data = resample_test_df.groupby(
    ['endpoint', 'name', 'test_fold', 'class_1_ratio']).mean().reset_index()
for endpoint in endpoints:
    for name in names:
        for fold in range(5):
            selected_endpoint = cur_data.endpoint == endpoint
            selected_name = cur_data.name == name
            selected_fold = cur_data.test_fold == fold
            selected_ratio = cur_data.class_1_ratio == ratio
            selected_indice = selected_endpoint & selected_name & selected_ratio & selected_fold
            for item in cur_data[selected_indice][col_name].values:
                print(item)
ratio = 0.5
cur_data = resample_maastro_df.groupby(
    ['endpoint', 'name', 'test_fold', 'class_1_ratio']).mean().reset_index()
for endpoint in endpoints:
    for name in names:
        for fold in range(5):
            selected_endpoint = cur_data.endpoint == endpoint
            selected_name = cur_data.name == name
            selected_fold = cur_data.test_fold == fold
            selected_ratio = cur_data.class_1_ratio == ratio
            selected_indice = selected_endpoint & selected_name & selected_ratio & selected_fold
            for item in cur_data[selected_indice][col_name].values:
                print(item)
# endregion

###################################################
# Plot
###################################################
# region Plot boostrap sampling results
resampled_per_fold_df = pd.concat([resample_test_df, resample_maastro_df])
resampled_per_fold_df['dataset'] = [
    *np.repeat('OUS', resample_test_df.shape[0]), *np.repeat('MAASTRO', resample_test_df.shape[0])]

sns.set_style('whitegrid')

# adjusting the percentage
class_1_ratio = resampled_per_fold_df.class_1_ratio.values
class_1_ratio[class_1_ratio == 0.39] = 0.40
class_1_ratio[class_1_ratio == 0.61] = 0.60
resampled_per_fold_df.class_1_ratio = class_1_ratio

# adjusting the model name
names = resampled_per_fold_df.name.values
names[names == 'clinical'] = 'M3 Clinical data (DL)'
names[names == 'clinical interaction'] = 'M4 Clinical data (DL w/ interaction)'
names[names ==
      'clinical_LogisticRegression'] = 'M1 Clinical data (Logistic Regression)'
names[names ==
      'clinical_RandomForestClassifier'] = 'M2 Clinical data (Random Forest)'
names[names == 'CT_PET'] = 'M5 CT & PET (DL-CNN)'
names[names == 'CT_PET_T'] = 'M6 CT, PET & GTVp (DL-CNN)'
names[names == 'CT_PET_T_N'] = 'M7 CT, PET, GTVp & GTVn (DL-CNN)'
resampled_per_fold_df.name = names


resampled_all_df = pd.concat(
    [resample_test_concat_df, resample_maastro_concat_df])
resampled_all_df['dataset'] = [
    *np.repeat('OUS', resample_test_concat_df.shape[0]), *np.repeat('MAASTRO', resample_maastro_concat_df.shape[0])]

# adjusting the model name
names = resampled_all_df.name.values
names[names == 'clinical'] = 'M3 Clinical data (DL)'
names[names == 'clinical interaction'] = 'M4 Clinical data (DL w/ interaction)'
names[names ==
      'clinical_LogisticRegression'] = 'M1 Clinical data (Logistic Regression)'
names[names ==
      'clinical_RandomForestClassifier'] = 'M2 Clinical data (Random Forest)'
names[names == 'CT_PET'] = 'M5 CT & PET (DL-CNN)'
names[names == 'CT_PET_T'] = 'M6 CT, PET & GTVp (DL-CNN)'
names[names == 'CT_PET_T_N'] = 'M7 CT, PET, GTVp & GTVn (DL-CNN)'
resampled_all_df.name = names

# order for hue name
name_order = ['M1 Clinical data (Logistic Regression)', 'M2 Clinical data (Random Forest)',
              'M3 Clinical data (DL)', 'M4 Clinical data (DL w/ interaction)',
              'M5 CT & PET (DL-CNN)', 'M6 CT, PET & GTVp (DL-CNN)',
              'M7 CT, PET, GTVp & GTVn (DL-CNN)']

# create plot
# region Plot each metric in each fold
endpoint = 'DFS'
for fold in range(5):
    plt.figure(figsize=(15, 8))
    i = 0
    for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
        i += 1
        ax = plt.subplot(2, 3, i)
        ax.set_title(f'Class 1 ratio {ratio:4.2f}')
        selected_fold = resampled_per_fold_df.test_fold == fold
        selected_ratio = resampled_per_fold_df.class_1_ratio == ratio
        selected_endpoint = resampled_per_fold_df.endpoint == endpoint
        selected_indice = selected_ratio & selected_fold & selected_endpoint
        curr_data = resampled_per_fold_df[selected_indice]
        ax = sns.boxplot(data=curr_data, x='dataset',
                         y='auc', hue='name', hue_order=name_order,
                         #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                         palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                                  '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                         #  showmeans=True,
                         #  meanprops={"marker": ".", "markerfacecolor": "red"},
                         ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('AUC')
        if i < 6:
            ax.get_legend().remove()

    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
    plt.suptitle(f'{endpoint} - Fold {fold}')
    plt.show()

endpoint = 'DFS'
for fold in range(5):
    plt.figure(figsize=(15, 8))
    i = 0
    for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
        i += 1
        ax = plt.subplot(2, 3, i)
        ax.set_title(f'Class 1 ratio {ratio:4.2f}')
        selected_fold = resampled_per_fold_df.test_fold == fold
        selected_ratio = resampled_per_fold_df.class_1_ratio == ratio
        selected_endpoint = resampled_per_fold_df.endpoint == endpoint
        selected_indice = selected_ratio & selected_fold & selected_endpoint
        curr_data = resampled_per_fold_df[selected_indice]
        ax = sns.boxplot(data=curr_data, x='dataset',
                         y='mcc', hue='name', hue_order=name_order,
                         #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                         palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                                  '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                         #  showmeans=True,
                         #  meanprops={"marker": ".", "markerfacecolor": "red"},
                         ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('MCC')
        if i < 6:
            ax.get_legend().remove()

    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
    plt.suptitle(f'{endpoint} - Fold {fold}')
    plt.show()
# endregion

###################################################################################
# average on test fold
###################################################################################
# ----------------------------------------------------------------------------------
# AUC
# ----------------------------------------------------------------------------------
# region Average test fold AUC
endpoint = 'DFS'
plt.figure(figsize=(15, 8))
i = 0
for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_per_fold_df.class_1_ratio == ratio
    selected_endpoint = resampled_per_fold_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_per_fold_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y='auc', hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('AUC')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()

endpoint = 'OS'
plt.figure(figsize=(15, 8))
i = 0
for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_per_fold_df.class_1_ratio == ratio
    selected_endpoint = resampled_per_fold_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_per_fold_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y='auc', hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('AUC')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()
# endregion

# ----------------------------------------------------------------------------------
# MCC
# ----------------------------------------------------------------------------------
# region Average test fold MCC
endpoint = 'DFS'
plt.figure(figsize=(15, 8))
i = 0
for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_per_fold_df.class_1_ratio == ratio
    selected_endpoint = resampled_per_fold_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_per_fold_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y='mcc', hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('MCC')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()

endpoint = 'OS'
plt.figure(figsize=(15, 8))
i = 0
for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_per_fold_df.class_1_ratio == ratio
    selected_endpoint = resampled_per_fold_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_per_fold_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y='mcc', hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('MCC')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()
# endregion


# region Plot concat/ensemble results
# ----------------------------------------------------------------------------------
# AUC
# ----------------------------------------------------------------------------------
# region ensemble AUC res
endpoint = 'DFS'
plt.figure(figsize=(15, 8))
i = 0
for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y='auc', hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('AUC')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()

endpoint = 'OS'
plt.figure(figsize=(15, 8))
i = 0
for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y='auc', hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('AUC')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()
# endregion ensemble MCC res

# ----------------------------------------------------------------------------------
# MCC
# ----------------------------------------------------------------------------------
# region ensemble MCC res
endpoint = 'DFS'
plt.figure(figsize=(15, 8))
i = 0
for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y='mcc', hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('MCC')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()

endpoint = 'OS'
plt.figure(figsize=(15, 8))
i = 0
for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y='mcc', hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('MCC')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()
# endregion ensemble MCC res

# ----------------------------------------------------------------------------------
# F1
# ----------------------------------------------------------------------------------
# region ensemble F1 res
endpoint = 'DFS'
plt.figure(figsize=(15, 8))
i = 0
for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y='f1', hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('F1 score')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()

endpoint = 'OS'
plt.figure(figsize=(15, 8))
i = 0
for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y='f1', hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('F1 score')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()
# endregion

# ----------------------------------------------------------------------------------
# F1 class 0
# ----------------------------------------------------------------------------------
# region ensemble F1 class 0 res
endpoint = 'DFS'
plt.figure(figsize=(15, 8))
i = 0
for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y='f1_0', hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('F1 score class 0')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()

endpoint = 'OS'
plt.figure(figsize=(15, 8))
i = 0
for ratio in [0.25, 0.40, 0.50, 0.60, 0.75]:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y='f1_0', hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('F1 score class 0')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()
# endregion

# ----------------------------------------------------------------------------------
# Plot all metrics 50% ratio
# ----------------------------------------------------------------------------------
# region Plot all metrics 50% ratio
metric_names = [('auc', 'AUC'), ('mcc', 'MCC'), ('f1', 'F1 score'),
                ('f1_0', 'F1 score class 0'), ('accuracy', 'Accuracy')]
ratio = 0.50

endpoint = 'DFS'
plt.figure(figsize=(15, 8))
i = 0
for metric_name, metric_label in metric_names:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y=metric_name, hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(metric_label)
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()

endpoint = 'OS'
plt.figure(figsize=(15, 8))
i = 0
for metric_name, metric_label in metric_names:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y=metric_name, hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(metric_label)
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()
# endregion

# ----------------------------------------------------------------------------------
# Plot all metrics 25% ratio
# ----------------------------------------------------------------------------------
# region Plot all metrics 25% ratio
metric_names = [('auc', 'AUC'), ('mcc', 'MCC'), ('f1', 'F1 score'),
                ('f1_0', 'F1 score class 0'), ('accuracy', 'Accuracy')]
ratio = 0.25

endpoint = 'DFS'
plt.figure(figsize=(15, 8))
i = 0
for metric_name, metric_label in metric_names:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y=metric_name, hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(metric_label)
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()

endpoint = 'OS'
plt.figure(figsize=(15, 8))
i = 0
for metric_name, metric_label in metric_names:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y=metric_name, hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(metric_label)
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()
# endregion


# ----------------------------------------------------------------------------------
# Plot all metrics 75% ratio
# ----------------------------------------------------------------------------------
# region Plot all metrics 75% ratio
metric_names = [('auc', 'AUC'), ('mcc', 'MCC'), ('f1', 'F1 score'),
                ('f1_0', 'F1 score class 0'), ('accuracy', 'Accuracy')]
ratio = 0.75

endpoint = 'DFS'
plt.figure(figsize=(15, 8))
i = 0
for metric_name, metric_label in metric_names:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y=metric_name, hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(metric_label)
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()

endpoint = 'OS'
plt.figure(figsize=(15, 8))
i = 0
for metric_name, metric_label in metric_names:
    i += 1
    ax = plt.subplot(2, 3, i)
    ax.set_title(f'Class 1 ratio {ratio:4.2f}')
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y=metric_name, hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(metric_label)
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()
# endregion


# region Plot all metrics 50% ratio ESTRO
metric_names = [('auc', 'AUC'), ('mcc', 'MCC'), ('f1', 'F1 score class 1'),
                ('f1_0', 'F1 score class 0'), ('accuracy', 'Accuracy')]
ratio = 0.50

endpoint = 'DFS'
plt.figure(figsize=(8, 13))
i = 0
for metric_name, metric_label in metric_names:
    i += 1
    ax = plt.subplot(3, 2, i)
    ax.set_title(metric_label)
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y=metric_name, hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()

endpoint = 'OS'
plt.figure(figsize=(8, 13))
i = 0
for metric_name, metric_label in metric_names:
    i += 1
    ax = plt.subplot(3, 2, i)
    ax.set_title(metric_label)
    selected_ratio = resampled_all_df.class_1_ratio == ratio
    selected_endpoint = resampled_all_df.endpoint == endpoint
    selected_indice = selected_ratio & selected_endpoint
    curr_data = resampled_all_df[selected_indice]
    ax = sns.boxplot(data=curr_data, x='dataset',
                     y=metric_name, hue='name', hue_order=name_order,
                     #  palette=['#FDB366', '#F67E4B', '#DD3D2D', '#A50026', '#CCDDAA', '#CCEEFF','#BBCCEE'],
                     palette=['#D1E5F0', '#92C5DE',  '#4393C3',
                              '#2166AC', '#F4A582', '#D6604D', '#B2182B'],
                     #  showmeans=True,
                     #  meanprops={"marker": ".", "markerfacecolor": "red"},
                     ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if i < 6:
        ax.get_legend().remove()

plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0)
plt.suptitle(f'{endpoint}')
plt.show()
# endregion


# endregion Plot concat/ensemble results

# endregion Plot boostrap sampling results


# region Plot original results OUS

# -------------------------------------------------
# AUC
# -------------------------------------------------
names = ['clinical_LogisticRegression',
         'clinical_RandomForestClassifier',
         'clinical', 'clinical interaction',
         'CT_PET', 'CT_PET_T', 'CT_PET_T_N']
endpoints = ['DFS', 'OS']

for endpoint in endpoints:
    plt.figure(figsize=(15, 7))
    i = 0
    for name in names:
        i += 1
        ax = plt.subplot(2, 4, i)
        # plot each fold
        for fold in range(5):
            selected_fold = df_test_res_all.test_fold == fold
            selected_endpoint = df_test_res_all.endpoint == endpoint
            selected_name = df_test_res_all.name == name
            selected_indice = selected_endpoint & selected_name & selected_fold
            cur_y = df_test_res_all[selected_indice]['y']
            cur_pred = df_test_res_all[selected_indice]['predicted']
            fpr, tpr, _ = roc_curve(cur_y, cur_pred)
            score = auc_score(cur_y, cur_pred)
            ax.plot(fpr, tpr, label=f'fold {fold} - {score:4.2f}')
        # plot the concat data
        selected_endpoint = df_test_concat_res_all.endpoint == endpoint
        selected_name = df_test_concat_res_all.name == name
        selected_indice = selected_endpoint & selected_name
        cur_y = df_test_concat_res_all[selected_indice]['y']
        cur_pred = df_test_concat_res_all[selected_indice]['predicted']
        fpr, tpr, _ = roc_curve(cur_y, cur_pred)
        score = auc_score(cur_y, cur_pred)
        ax.plot(fpr, tpr, label=f'concat - {score:4.2f}', lw=3)
        # plot the 0.5 diagon
        ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
        ax.set_title(name)
        ax.legend()
    # comparing all
    i += 1
    ax = plt.subplot(2, 4, i)
    for name in names:
        selected_endpoint = df_test_concat_res_all.endpoint == endpoint
        selected_name = df_test_concat_res_all.name == name
        selected_indice = selected_endpoint & selected_name
        cur_y = df_test_concat_res_all[selected_indice]['y']
        cur_pred = df_test_concat_res_all[selected_indice]['predicted']
        fpr, tpr, _ = roc_curve(cur_y, cur_pred)
        score = auc_score(cur_y, cur_pred)
        ax.plot(fpr, tpr, label=f'{name} - {score:4.2f}')
    ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
    ax.set_title('All')
    ax.legend()
    plt.suptitle(f'OUS data - {endpoint}')
    plt.show()
# endregion Plot original results OUS


# region Plot original results MAASTRO

# -------------------------------------------------
# AUC
# -------------------------------------------------
names = ['clinical_LogisticRegression',
         'clinical_RandomForestClassifier',
         'clinical', 'clinical interaction',
         'CT_PET', 'CT_PET_T', 'CT_PET_T_N']
endpoints = ['DFS', 'OS']

for endpoint in endpoints:
    plt.figure(figsize=(15, 7))
    i = 0
    for name in names:
        i += 1
        ax = plt.subplot(2, 4, i)
        # plot each fold
        for fold in range(5):
            selected_fold = df_maastro_res_all.test_fold == fold
            selected_endpoint = df_maastro_res_all.endpoint == endpoint
            selected_name = df_maastro_res_all.name == name
            selected_indice = selected_endpoint & selected_name & selected_fold
            cur_y = df_maastro_res_all[selected_indice]['y']
            cur_pred = df_maastro_res_all[selected_indice]['predicted']
            fpr, tpr, _ = roc_curve(cur_y, cur_pred)
            score = auc_score(cur_y, cur_pred)
            ax.plot(fpr, tpr, label=f'fold {fold} - {score:4.2f}')
        # plot the concat data
        selected_endpoint = df_maastro_concat_res_all.endpoint == endpoint
        selected_name = df_maastro_concat_res_all.name == name
        selected_indice = selected_endpoint & selected_name
        cur_y = df_maastro_concat_res_all[selected_indice]['y']
        cur_pred = df_maastro_concat_res_all[selected_indice]['predicted']
        fpr, tpr, _ = roc_curve(cur_y, cur_pred)
        score = auc_score(cur_y, cur_pred)
        ax.plot(fpr, tpr, label=f'concat - {score:4.2f}', lw=3)
        # plot the 0.5 diagon
        ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
        ax.set_title(name)
        ax.legend()
    # comparing all
    i += 1
    ax = plt.subplot(2, 4, i)
    for name in names:
        selected_endpoint = df_maastro_concat_res_all.endpoint == endpoint
        selected_name = df_maastro_concat_res_all.name == name
        selected_indice = selected_endpoint & selected_name
        cur_y = df_maastro_concat_res_all[selected_indice]['y']
        cur_pred = df_maastro_concat_res_all[selected_indice]['predicted']
        fpr, tpr, _ = roc_curve(cur_y, cur_pred)
        score = auc_score(cur_y, cur_pred)
        ax.plot(fpr, tpr, label=f'{name} - {score:4.2f}')
    ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
    ax.set_title('All')
    ax.legend()
    plt.suptitle(f'MAASTRO data - {endpoint}')
    plt.show()
# endregion Plot original results MAASTRO


resampled_all_df[resampled_all_df.class_1_ratio == 0.5].groupby(
    ['endpoint', 'dataset', 'name']).agg(['median'])

# region Results with combined models
resample_all_comb_df = pd.concat(
    [resample_test_concat_df, resample_maastro_concat_df,
     resample_test_concat_combine_df, resample_maastro_concat_combine_df])
resample_all_comb_df['dataset'] = [
    *np.repeat('OUS', resample_test_concat_df.shape[0]),
    *np.repeat('MAASTRO', resample_maastro_concat_df.shape[0]),
    *np.repeat('OUS', resample_test_concat_combine_df.shape[0]),
    *np.repeat('MAASTRO', resample_maastro_concat_combine_df.shape[0])]

# adjusting the model name
names = resample_all_comb_df.name.values
names[names == 'clinical'] = 'M3 Clinical data (DL)'
names[names == 'clinical interaction'] = 'M4 Clinical data (DL w/ interaction)'
names[names ==
      'clinical_LogisticRegression'] = 'M1 Clinical data (Logistic Regression)'
names[names ==
      'clinical_RandomForestClassifier'] = 'M2 Clinical data (Random Forest)'
names[names == 'CT_PET'] = 'M5 CT & PET (DL-CNN)'
names[names == 'CT_PET_T'] = 'M6 CT, PET & GTVp (DL-CNN)'
names[names == 'CT_PET_T_N'] = 'M7 CT, PET, GTVp & GTVn (DL-CNN)'
resample_all_comb_df.name = names

# order for hue name
name_order = ['M1 Clinical data (Logistic Regression)', 'M2 Clinical data (Random Forest)',
              'M3 Clinical data (DL)', 'M4 Clinical data (DL w/ interaction)',
              'M5 CT & PET (DL-CNN)', 'M6 CT, PET & GTVp (DL-CNN)',
              'M7 CT, PET, GTVp & GTVn (DL-CNN)']

# endregion

resample_all_comb_df[resample_all_comb_df.class_1_ratio == 0.5].groupby(
    ['endpoint', 'dataset', 'name']).agg(['median']).tail(40)


resample_all_solo_df = pd.concat([
    resample_test_concat_df, resample_test_concat_additional_df,
    resample_maastro_concat_df, resample_maastro_concat_additional_df
])

resample_all_solo_df['dataset'] = [
    *np.repeat('OUS', resample_test_concat_df.shape[0]),
    *np.repeat('OUS', resample_test_concat_additional_df.shape[0]),
    *np.repeat('MAASTRO', resample_maastro_concat_df.shape[0]),
    *np.repeat('MAASTRO', resample_maastro_concat_additional_df.shape[0])]

resample_all_solo_df[resample_all_solo_df.class_1_ratio == 0.5].groupby(
    ['endpoint', 'dataset', 'name']).agg(['mean'])


all_names = resample_all_solo_df.name.values
data_group = []
for name in all_names:
    if 'clinical' in name:
        data_group.append('D1')
    elif 'radiomics' in name:
        data_group.append('D2')
    elif 'CT_PET' in name:
        data_group.append('D3')
    elif 'tabular' in name:
        data_group.append('D1 + D2')
    else:
        raise ValueError()
resample_all_solo_df['data_group'] = data_group

input_data_name = []
for name in all_names:
    if 'clinical' in name:
        if 'selected' in name:
            input_data_name.append('clinical_selected_90')
        else:
            input_data_name.append('clinical')
    elif 'radiomics' in name:
        if 'selected' in name:
            input_data_name.append('radiomics_selected_90')
        else:
            input_data_name.append('radiomics')
    elif 'tabular' in name:
        if 'selected' in name:
            input_data_name.append(name[:len('tabular_selected_xx')])
        else:
            input_data_name.append('tabular')
    elif name in ['CT_PET', 'CT_PET_T', 'CT_PET_T_N']:
        input_data_name.append(name)
    else:
        raise ValueError()
resample_all_solo_df['input_data_name'] = input_data_name

model_name = []
for name in all_names:
    if name == 'clinical':
        model_name.append('NN_basic')
    elif name in ['CT_PET', 'CT_PET_T', 'CT_PET_T_N']:
        model_name.append('CNN')
    elif 'basic' in name:
        model_name.append('NN_basic')
    elif 'interaction' in name:
        model_name.append('NN_interaction')
    elif 'LogisticRegression' in name:
        model_name.append('LogisticRegression')
    elif 'RandomForestClassifier' in name:
        model_name.append('RandomForestClassifier')
    else:
        print(name)
        raise ValueError()
resample_all_solo_df['model_name'] = model_name


single_model_res = resample_all_solo_df[resample_all_solo_df.class_1_ratio == 0.5].groupby(
    ['endpoint', 'dataset', 'input_data_name', 'model_name']).median().reset_index()

model_name_sorted = ['LogisticRegression',
                     'RandomForestClassifier', 'NN_basic', 'NN_interaction', 'CNN']
input_data_name_sorted = ['clinical_selected_90', 'radiomics_selected_90', 'tabular_selected_90', 'tabular_selected_50',
                          'clinical', 'radiomics', 'tabular', 'CT_PET', 'CT_PET_T', 'CT_PET_T_N']

single_model_res['model_name'] = single_model_res['model_name'].astype(
    "category").cat.set_categories(model_name_sorted)
single_model_res['input_data_name'] = single_model_res['input_data_name'].astype(
    "category").cat.set_categories(input_data_name_sorted)

endpoint = 'OS'
selected_endpoint = single_model_res.endpoint == endpoint
selected_dataset = single_model_res.dataset == 'MAASTRO'
single_model_res[selected_endpoint & selected_dataset][['input_data_name', 'model_name', 'auc',
                                                        'mcc', 'f1', 'f1_0', 'accuracy']].sort_values(['input_data_name', 'model_name']).head(33)


resample_all_combine_df = pd.concat([
    resample_test_concat_combine_df, resample_test_concat_combine_additional_df,
    resample_maastro_concat_combine_df, resample_maastro_concat_combine_additional_df
])


resample_all_combine_df['dataset'] = [
    *np.repeat('OUS', resample_test_concat_combine_df.shape[0]),
    *np.repeat('OUS', resample_test_concat_combine_additional_df.shape[0]),
    *np.repeat('MAASTRO', resample_maastro_concat_combine_df.shape[0]),
    *np.repeat('MAASTRO', resample_maastro_concat_combine_additional_df.shape[0])]

dfs_models = [
    'clinical_selected_90_LogisticRegression_CT_PET',
    'radiomics_selected_90_basic_nn_CT_PET',
    'tabular_selected_90_LogisticRegression_CT_PET',
    'tabular_selected_50_LogisticRegression_CT_PET',
    'clinical_LogisticRegression_CT_PET',
    'radiomics_basic_nn_CT_PET',
    'tabular_interaction_nn_CT_PET',
]

os_models = [
    'clinical_selected_90_LogisticRegression_CT_PET_T',
    'radiomics_selected_90_interaction_nn_CT_PET_T',
    'tabular_selected_90_interaction_nn_CT_PET_T',
    'tabular_selected_50_LogisticRegression_CT_PET_T',
    'clinical_LogisticRegression_CT_PET_T',
    'radiomics_basic_nn_CT_PET_T',
    'tabular_interaction_nn_CT_PET_T',
]


combine_model_res = resample_all_combine_df[resample_all_combine_df.class_1_ratio == 0.5].groupby(
    ['endpoint', 'dataset', 'name']).median().reset_index()

selected_endpoint = combine_model_res['endpoint'] == 'DFS'
selected_model = combine_model_res['name'].isin(dfs_models)
dfs_combine_model = combine_model_res[selected_endpoint &
                                      selected_model].reset_index(drop=True)
dfs_combine_model['name'] = dfs_combine_model['name'].astype(
    "category").cat.set_categories(dfs_models)

dfs_combine_model[['dataset', 'name', 'auc', 'mcc', 'f1',
                   'f1_0', 'accuracy']].sort_values(['dataset', 'name'])

selected_endpoint = combine_model_res['endpoint'] == 'OS'
selected_model = combine_model_res['name'].isin(os_models)
os_combine_model = combine_model_res[selected_endpoint &
                                     selected_model].reset_index(drop=True)
os_combine_model['name'] = os_combine_model['name'].astype(
    "category").cat.set_categories(os_models)

os_combine_model[['dataset', 'name', 'auc', 'mcc', 'f1',
                  'f1_0', 'accuracy']].sort_values(['dataset', 'name'])


combine_model_res[combine_model_res.endpoint == 'DFS'][[
    'dataset', 'name', 'auc', 'mcc', 'f1', 'f1_0', 'accuracy']].head(50)

combine_model_res[['endpoint', 'dataset', 'name', 'auc', 'mcc', 'f1',
                   'f1_0', 'accuracy']].to_csv('combine_model_median.csv', index=False)
