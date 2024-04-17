import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, f1_score
from itertools import product
import pandas as pd
import os


classifiers = [{
    'name': LogisticRegression,
    'preprocess': [StandardScaler],
    'params': {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-4, 4, 20).round(2)[5:],
        'solver': ['liblinear'],
    }
},
    {
    'name': RandomForestClassifier,
    'preprocess': [None],
    'params': {
        'n_estimators': list(range(10, 101, 10)),
        'max_features': list(range(1, 15, 1))
    }
}]

input_type = 'tabular_selected_0'
input_dataset = 'all_tabular'
importance_features = True

selected_features = {
    'DFS': {
        'clinical_selected_90': [7],
        'clinical_selected_50': [7, 10],
        'clinical_selected_10': [7, 9, 10, 3, 2],
        'clinical_selected_0': [7, 9, 10, 3, 5, 2, 1, 8, 0, 11],
        'radiomics_selected_90': [10, 368],
        'radiomics_selected_50': [0, 10, 368, 131],
        'radiomics_selected_10': [0, 10, 368, 371, 131, 217, 114],
        'radiomics_selected_0': [0, 10, 368, 371, 1, 131, 217, 241, 277, 114, 356, 107, 116, 165, 219, 240, 366, 111, 236, 232, 275, 363, 253, 205, 216],
        'tabular_selected_90': [14, 24, 382],
        'tabular_selected_50': [14, 24, 145, 382, 385, 7],
        'tabular_selected_10': [10, 14, 24, 145, 291, 382, 385, 15, 7, 231, 250, 128, 254, 2],
        'tabular_selected_0': [10, 14, 24, 145, 291, 382, 385, 0, 5, 15, 7, 231, 55, 380, 4, 250, 255, 1, 110, 128, 370, 214, 179, 377, 254, 2, 332, 125, 246, 289, 244, 114, 303, 42, 252, 230, 123, 267, 203, 337, 6, 116],
    },
    'OS': {
        'clinical_selected_90': [7, 10],
        'clinical_selected_50': [7, 10],
        'clinical_selected_10': [7, 10, 3, 9],
        'clinical_selected_0': [2, 7, 10, 3, 9, 8, 0],
        'radiomics_selected_90': [10],
        'radiomics_selected_50': [3, 10, 186, 191],
        'radiomics_selected_10': [3, 7, 10, 20, 186, 191, 320, 165, 253, 164],
        'radiomics_selected_0': [3, 7, 10, 20, 186, 191, 115, 320, 165, 253, 261, 349, 359, 164, 275, 317, 6, 72, 368, 28, 33, 205, 25, 279, 64, 45, 162, 170, 278, 209, 0, 301],
        'tabular_selected_90': [24],
        'tabular_selected_50': [7, 10, 24],
        'tabular_selected_10': [7, 10, 24, 3],
        'tabular_selected_0': [7, 10, 24, 9, 2, 3, 0],
    }
}


def clean_input(X, endpoint, input_type):
    if importance_features:
        selected_channel = selected_features[endpoint][input_type]
        return X[..., selected_channel]
    else:
        return X


def get_data(folds):
    data = []
    OS = []
    DFS = []
    for fold in folds:
        with h5py.File('../../outcome_all.h5', 'r') as f:
            data.append(f[f'fold_{fold}'][input_dataset][:])
            OS.append(f[f'fold_{fold}']['OS'][:])
            DFS.append(f[f'fold_{fold}']['DFS'][:])
    data = np.concatenate(data)
    OS = np.concatenate(OS).flatten()
    DFS = np.concatenate(DFS).flatten()
    return data, OS, DFS


def get_maastro():
    data = []
    OS = []
    DFS = []
    with h5py.File('../../outcome_all_maastro.h5', 'r') as f:
        for fold in f.keys():
            data.append(f[fold][input_dataset][:])
            OS.append(f[fold]['OS'][:])
            DFS.append(f[fold]['DFS'][:])
    data = np.concatenate(data)
    OS = np.concatenate(OS).flatten()
    DFS = np.concatenate(DFS).flatten()

    return data, OS, DFS


def ensemble(results):
    return np.mean(results, axis=-1)


def small_swap(original):
    res = [original]
    for i in range(2, -1, -1):
        new_str = res[-1].copy()
        new_str[3] = res[-1][i]
        new_str[i] = res[-1][3]
        res.append(new_str)
    return res


def large_swap(original):
    res = [original]
    for i in range(3, -1, -1):
        new_str = res[-1].copy()
        new_str[4] = res[-1][i]
        new_str[i] = res[-1][4]
        res.append(new_str)
    return res


fold_orders = []
for fold_order in large_swap(np.arange(5)):
    fold_orders.extend(small_swap(fold_order))


results = []
for classifier in classifiers:
    classifiers_cls = classifier['name']
    name = classifiers_cls.__name__
    keys, values = list(classifier['params'].keys()
                        ), classifier['params'].values()
    params = [{keys[i]: values[i]
               for i in range(len(keys))} for values in product(*values)]
    print(name)
    for fold_order in fold_orders:
        train_X, train_OS, train_DFS = get_data(fold_order[:3])
        val_X, val_OS, val_DFS = get_data(fold_order[3:4])
        test_X, test_OS, test_DFS = get_data(fold_order[4:5])
        external_X, external_OS, external_DFS = get_maastro()

        for preprocess in classifier['preprocess']:
            if preprocess:
                pp = preprocess()
                train_X = pp.fit_transform(train_X)
                val_X = pp.transform(val_X)
                test_X = pp.transform(test_X)
                external_X = pp.transform(external_X)

            print('preprocess', preprocess)

            for param in params:
                print(param)
                param_str = '__'.join(
                    [f'{key}_{str(val)}' for key, val in param.items()])
                try:
                    OS_model = classifiers_cls(**param)
                    OS_model.fit(clean_input(
                        train_X, 'OS', input_type), train_OS)

                    train_pred_OS = OS_model.predict(
                        clean_input(train_X, 'OS', input_type))
                    train_prob_OS = OS_model.predict_proba(
                        clean_input(train_X, 'OS', input_type))[:, 1]

                    val_pred_OS = OS_model.predict(
                        clean_input(val_X, 'OS', input_type))
                    val_prob_OS = OS_model.predict_proba(
                        clean_input(val_X, 'OS', input_type))[:, 1]

                    test_pred_OS = OS_model.predict(
                        clean_input(test_X, 'OS', input_type))
                    test_prob_OS = OS_model.predict_proba(
                        clean_input(test_X, 'OS', input_type))[:, 1]

                    external_pred_OS = OS_model.predict(
                        clean_input(external_X, 'OS', input_type))
                    external_prob_OS = OS_model.predict_proba(
                        clean_input(external_X, 'OS', input_type))[:, 1]

                    filename = f'../../outcome_ml_res/test/OS_{input_type}_{name}__test_{fold_order[4]}__{param_str}.csv'
                    if os.path.exists(filename):
                        temp_df = pd.read_csv(filename)
                        temp_df[f'val_{fold_order[3]}'] = test_prob_OS
                    else:
                        temp_df = pd.DataFrame(test_prob_OS, columns=[
                            f'val_{fold_order[3]}'])
                    temp_df.to_csv(filename, index=False)

                    filename = f'../../outcome_ml_res/external/OS_{input_type}_{name}__test_{fold_order[4]}__{param_str}.csv'
                    if os.path.exists(filename):
                        temp_df = pd.read_csv(filename)
                        temp_df[f'val_{fold_order[3]}'] = external_prob_OS
                    else:
                        temp_df = pd.DataFrame(external_prob_OS, columns=[
                            f'val_{fold_order[3]}'])
                    temp_df.to_csv(filename, index=False)

                    results.append({
                        'name': name,
                        'params': param_str,
                        'target': 'OS',
                        'val_fold': fold_order[3],
                        'test_fold': fold_order[4],

                        'train_acc': accuracy_score(train_OS, train_pred_OS),
                        'train_mcc': matthews_corrcoef(train_OS, train_pred_OS),
                        'train_f1': f1_score(train_OS, train_pred_OS),
                        'train_f1_0': f1_score(1 - train_OS, 1 - train_pred_OS),
                        'train_auc': roc_auc_score(1 - train_OS, 1 - train_prob_OS),

                        'val_acc': accuracy_score(val_OS, val_pred_OS),
                        'val_mcc': matthews_corrcoef(val_OS, val_pred_OS),
                        'val_f1': f1_score(val_OS, val_pred_OS),
                        'val_f1_0': f1_score(1 - val_OS, 1 - val_pred_OS),
                        'val_auc': roc_auc_score(1 - val_OS, 1 - val_prob_OS),

                        'test_acc': accuracy_score(test_OS, test_pred_OS),
                        'test_mcc': matthews_corrcoef(test_OS, test_pred_OS),
                        'test_f1': f1_score(test_OS, test_pred_OS),
                        'test_f1_0': f1_score(1 - test_OS, 1 - test_pred_OS),
                        'test_auc': roc_auc_score(1 - test_OS, 1 - test_prob_OS),

                        'external_acc': accuracy_score(external_OS, external_pred_OS),
                        'external_mcc': matthews_corrcoef(external_OS, external_pred_OS),
                        'external_f1': f1_score(external_OS, external_pred_OS),
                        'external_f1_0': f1_score(1 - external_OS, 1 - external_pred_OS),
                        'external_auc': roc_auc_score(1 - external_OS, 1 - external_prob_OS),
                    })
                    print(results[-1]['val_auc'], results[-1]['val_mcc'])
                except ValueError as e:
                    print(e)
                    print('Invalid tree params due to less number of columns in OS')

                try:
                    DFS_model = classifiers_cls(**param)
                    DFS_model.fit(clean_input(
                        train_X, 'DFS', input_type), train_DFS)

                    train_pred_DFS = DFS_model.predict(
                        clean_input(train_X, 'DFS', input_type))
                    train_prob_DFS = DFS_model.predict_proba(
                        clean_input(train_X, 'DFS', input_type))[:, 1]

                    val_pred_DFS = DFS_model.predict(
                        clean_input(val_X, 'DFS', input_type))
                    val_prob_DFS = DFS_model.predict_proba(
                        clean_input(val_X, 'DFS', input_type))[:, 1]

                    test_pred_DFS = DFS_model.predict(
                        clean_input(test_X, 'DFS', input_type))
                    test_prob_DFS = DFS_model.predict_proba(
                        clean_input(test_X, 'DFS', input_type))[:, 1]

                    external_pred_DFS = DFS_model.predict(
                        clean_input(external_X, 'DFS', input_type))
                    external_prob_DFS = DFS_model.predict_proba(
                        clean_input(external_X, 'DFS', input_type))[:, 1]

                    filename = f'../../outcome_ml_res/test/DFS_{input_type}_{name}__test_{fold_order[4]}__{param_str}.csv'
                    if os.path.exists(filename):
                        temp_df = pd.read_csv(filename)
                        temp_df[f'val_{fold_order[3]}'] = test_prob_DFS
                    else:
                        temp_df = pd.DataFrame(test_prob_DFS, columns=[
                            f'val_{fold_order[3]}'])
                    temp_df.to_csv(filename, index=False)

                    filename = f'../../outcome_ml_res/external/DFS_{input_type}_{name}__test_{fold_order[4]}__{param_str}.csv'
                    if os.path.exists(filename):
                        temp_df = pd.read_csv(filename)
                        temp_df[f'val_{fold_order[3]}'] = external_prob_DFS
                    else:
                        temp_df = pd.DataFrame(external_prob_DFS, columns=[
                            f'val_{fold_order[3]}'])
                    temp_df.to_csv(filename, index=False)

                    results.append({
                        'name': name,
                        'params': param_str,
                        'target': 'DFS',
                        'val_fold': fold_order[3],
                        'test_fold': fold_order[4],

                        'train_acc': accuracy_score(train_DFS, train_pred_DFS),
                        'train_mcc': matthews_corrcoef(train_DFS, train_pred_DFS),
                        'train_f1': f1_score(train_DFS, train_pred_DFS),
                        'train_f1_0': f1_score(1 - train_DFS, 1 - train_pred_DFS),
                        'train_auc': roc_auc_score(1 - train_DFS, 1 - train_prob_DFS),

                        'val_acc': accuracy_score(val_DFS, val_pred_DFS),
                        'val_mcc': matthews_corrcoef(val_DFS, val_pred_DFS),
                        'val_f1': f1_score(val_DFS, val_pred_DFS),
                        'val_f1_0': f1_score(1 - val_DFS, 1 - val_pred_DFS),
                        'val_auc': roc_auc_score(1 - val_DFS, 1 - val_prob_DFS),

                        'test_acc': accuracy_score(test_DFS, test_pred_DFS),
                        'test_mcc': matthews_corrcoef(test_DFS, test_pred_DFS),
                        'test_f1': f1_score(test_DFS, test_pred_DFS),
                        'test_f1_0': f1_score(1 - test_DFS, 1 - test_pred_DFS),
                        'test_auc': roc_auc_score(1 - test_DFS, 1 - test_prob_DFS),

                        'external_acc': accuracy_score(external_DFS, external_pred_DFS),
                        'external_mcc': matthews_corrcoef(external_DFS, external_pred_DFS),
                        'external_f1': f1_score(external_DFS, external_pred_DFS),
                        'external_f1_0': f1_score(1 - external_DFS, 1 - external_pred_DFS),
                        'external_auc': roc_auc_score(1 - external_DFS, 1 - external_prob_DFS),
                    })
                    print(results[-1]['val_auc'], results[-1]['val_mcc'])
                except ValueError as e:
                    print(e)
                    print('Invalid tree params due to less number of columns in DFS')


res_df = pd.DataFrame(results)
# res_df['avg_score'] = (res_df['val_auc'] + res_df['val_mcc'] +
#                        res_df['val_f1'] + 0.75*res_df['val_f1_0'] + 0.5*res_df['train_f1']) / 4.25
train_f1 = 2 * np.sqrt(res_df['train_f1']) / 3
res_df['avg_score'] = (res_df['val_auc'] + res_df['val_mcc'] +
                       res_df['val_f1'] + 0.8*res_df['val_f1_0'] + 0.8*train_f1) / 4.6
res_df.to_csv(
    f'ml_results/outcome_{input_type}_traditional_ml.csv', index=False)


metrics = ['train_f1', 'train_f1_0', 'train_auc', 'train_mcc', 'val_f1',
           'val_f1_0', 'val_auc', 'val_mcc', 'test_f1', 'test_f1_0', 'test_auc', 'test_mcc']
res_df[res_df.target == 'OS'][res_df.name == 'LogisticRegression'][res_df.test_fold == 4].groupby(['name', 'params']).mean().sort_values(
    'avg_score', ascending=False)[metrics].round(2).head(3)
res_df[res_df.target == 'OS'][res_df.name == 'LogisticRegression'][res_df.test_fold == 3].groupby(['name', 'params']).mean().sort_values(
    'avg_score', ascending=False)[metrics].round(2).head(3)
res_df[res_df.target == 'OS'][res_df.name == 'LogisticRegression'][res_df.test_fold == 2].groupby(['name', 'params']).mean().sort_values(
    'avg_score', ascending=False)[metrics].round(2).head(3)
res_df[res_df.target == 'OS'][res_df.name == 'LogisticRegression'][res_df.test_fold == 1].groupby(['name', 'params']).mean().sort_values(
    'avg_score', ascending=False)[metrics].round(2).head(3)
res_df[res_df.target == 'OS'][res_df.name == 'LogisticRegression'][res_df.test_fold == 0].groupby(['name', 'params']).mean().sort_values(
    'avg_score', ascending=False)[metrics].round(2).head(3)


res_df[res_df.target == 'OS'][res_df.name == 'RandomForestClassifier'][res_df.test_fold == 4].groupby(['name', 'params']).mean().sort_values(
    'avg_score', ascending=False)[metrics].round(2)
res_df[res_df.target == 'OS'][res_df.name == 'RandomForestClassifier'][res_df.test_fold == 3].groupby(['name', 'params']).mean().sort_values(
    'avg_score', ascending=False)[metrics].round(2)
res_df[res_df.target == 'OS'][res_df.name == 'RandomForestClassifier'][res_df.test_fold == 2].groupby(['name', 'params']).mean().sort_values(
    'avg_score', ascending=False)[metrics].round(2)
res_df[res_df.target == 'OS'][res_df.name == 'RandomForestClassifier'][res_df.test_fold == 1].groupby(['name', 'params']).mean().sort_values(
    'avg_score', ascending=False)[metrics].round(2)
res_df[res_df.target == 'OS'][res_df.name == 'RandomForestClassifier'][res_df.test_fold == 0].groupby(['name', 'params']).mean().sort_values(
    'avg_score', ascending=False)[metrics].round(2)


OS_name, OS_param_str = res_df[res_df.target == 'OS'].groupby(['name', 'params']).mean().sort_values(
    'avg_score', ascending=False).index[0]
DFS_name, DFS_param_str = res_df[res_df.target == 'DFS'].groupby(['name', 'params']).mean().sort_values(
    'avg_score', ascending=False).index[0]


res_df[res_df.target == 'OS'][res_df.name == 'LogisticRegression'].groupby(['params']).mean().sort_values(
    'avg_score', ascending=False).index.values[0]
# penalty_l2__C_0.03__solver_liblinear

res_df[res_df.target == 'OS'][res_df.name == 'RandomForestClassifier'].groupby(['params']).mean().sort_values(
    'avg_score', ascending=False).index.values[0]
# n_estimators_30__max_features_1


filename = '../../outcome_ml_res/{type}/{endpoint}_{input_type}_{name}__test_{fold}__{param_str}.csv'
_, external_OS, external_DFS = get_maastro()
test_ensemble_res = []
for fold in range(5):
    for name, param_str in res_df.groupby(['name', 'params']).mean().index.values:
        try:
            _, test_OS, test_DFS = get_data([fold])
            OS_en_test = pd.read_csv(filename.format(
                type='test', endpoint='OS', name=name, fold=fold, param_str=param_str, input_type=input_type)).mean(axis=1)
            OS_en_test_binarized = (OS_en_test > 0.5).astype(int)

            OS_en_external = pd.read_csv(filename.format(
                type='external', endpoint='OS', name=name, fold=fold, param_str=param_str, input_type=input_type)).mean(axis=1)
            OS_en_external_binarized = (OS_en_external > 0.5).astype(int)

            test_ensemble_res.append({
                'name': name,
                'params': param_str,
                'target': 'OS',
                'test_fold': fold,
                'test_acc': accuracy_score(test_OS, OS_en_test_binarized),
                'test_mcc': matthews_corrcoef(test_OS, OS_en_test_binarized),
                'test_f1': f1_score(test_OS, OS_en_test_binarized),
                'test_f1_0': f1_score(1 - test_OS, 1 - OS_en_test_binarized),
                'test_auc': roc_auc_score(test_OS, OS_en_test),

                'external_acc': accuracy_score(external_OS, OS_en_external_binarized),
                'external_mcc': matthews_corrcoef(external_OS, OS_en_external_binarized),
                'external_f1': f1_score(external_OS, OS_en_external_binarized),
                'external_f1_0': f1_score(1 - external_OS, 1 - OS_en_external_binarized),
                'external_auc': roc_auc_score(external_OS, OS_en_external)
            })

        except FileNotFoundError as e:
            print('CSV not found')

        try:
            DFS_en_test = pd.read_csv(filename.format(
                type='test', endpoint='DFS', name=name, fold=fold, param_str=param_str, input_type=input_type)).mean(axis=1)
            DFS_en_test_binarized = (DFS_en_test > 0.5).astype(int)

            DFS_en_external = pd.read_csv(filename.format(
                type='external', endpoint='DFS', name=name, fold=fold, param_str=param_str, input_type=input_type)).mean(axis=1)
            DFS_en_external_binarized = (DFS_en_external > 0.5).astype(int)

            test_ensemble_res.append({
                'name': name,
                'params': param_str,
                'target': 'DFS',
                'test_fold': fold,
                'test_acc': accuracy_score(test_DFS, DFS_en_test_binarized),
                'test_mcc': matthews_corrcoef(test_DFS, DFS_en_test_binarized),
                'test_f1': f1_score(test_DFS, DFS_en_test_binarized),
                'test_f1_0': f1_score(1 - test_DFS, 1 - DFS_en_test_binarized),
                'test_auc': roc_auc_score(test_DFS, DFS_en_test),

                'external_acc': accuracy_score(external_DFS, DFS_en_external_binarized),
                'external_mcc': matthews_corrcoef(external_DFS, DFS_en_external_binarized),
                'external_f1': f1_score(external_DFS, DFS_en_external_binarized),
                'external_f1_0': f1_score(1 - external_DFS, 1 - DFS_en_external_binarized),
                'external_auc': roc_auc_score(external_DFS, DFS_en_external)
            })
        except FileNotFoundError as e:
            print('CSV not found')

test_ensemble_df = pd.DataFrame(test_ensemble_res)
test_ensemble_df.to_csv(f'ml_results/test_ensemble_{input_type}_ml.csv')


test_ensemble_res_selected = []
for name in ['LogisticRegression', 'RandomForestClassifier']:
    param_str_OS = res_df[res_df.target == 'OS'][res_df.name == name].groupby(
        ['params']).mean().sort_values('avg_score', ascending=False).index.values[0]
    param_str_DFS = res_df[res_df.target == 'DFS'][res_df.name == name].groupby(
        ['params']).mean().sort_values('avg_score', ascending=False).index.values[0]
    for fold in range(5):
        _, test_OS, test_DFS = get_data([fold])
        OS_en_test = pd.read_csv(filename.format(
            type='test', endpoint='OS', name=name, fold=fold, param_str=param_str_OS, input_type=input_type)).mean(axis=1)
        OS_en_test_binarized = (OS_en_test > 0.5).astype(int)
        DFS_en_test = pd.read_csv(filename.format(
            type='test', endpoint='DFS', name=name, fold=fold, param_str=param_str_DFS, input_type=input_type)).mean(axis=1)
        DFS_en_test_binarized = (DFS_en_test > 0.5).astype(int)
        OS_en_external = pd.read_csv(filename.format(
            type='external', endpoint='OS', name=name, fold=fold, param_str=param_str_OS, input_type=input_type)).mean(axis=1)
        OS_en_external_binarized = (OS_en_external > 0.5).astype(int)
        DFS_en_external = pd.read_csv(filename.format(
            type='external', endpoint='DFS', name=name, fold=fold, param_str=param_str_DFS, input_type=input_type)).mean(axis=1)
        DFS_en_external_binarized = (DFS_en_external > 0.5).astype(int)

        test_ensemble_res_selected.append({
            'name': name,
            'params': param_str_OS,
            'target': 'OS',
            'test_fold': fold,
            'test_acc': accuracy_score(test_OS, OS_en_test_binarized),
            'test_mcc': matthews_corrcoef(test_OS, OS_en_test_binarized),
            'test_f1': f1_score(test_OS, OS_en_test_binarized),
            'test_f1_0': f1_score(1 - test_OS, 1 - OS_en_test_binarized),
            'test_auc': roc_auc_score(test_OS, OS_en_test),

            'external_acc': accuracy_score(external_OS, OS_en_external_binarized),
            'external_mcc': matthews_corrcoef(external_OS, OS_en_external_binarized),
            'external_f1': f1_score(external_OS, OS_en_external_binarized),
            'external_f1_0': f1_score(1 - external_OS, 1 - OS_en_external_binarized),
            'external_auc': roc_auc_score(external_OS, OS_en_external)
        })

        test_ensemble_res_selected.append({
            'name': name,
            'params': param_str_DFS,
            'target': 'DFS',
            'test_fold': fold,
            'test_acc': accuracy_score(test_DFS, DFS_en_test_binarized),
            'test_mcc': matthews_corrcoef(test_DFS, DFS_en_test_binarized),
            'test_f1': f1_score(test_DFS, DFS_en_test_binarized),
            'test_f1_0': f1_score(1 - test_DFS, 1 - DFS_en_test_binarized),
            'test_auc': roc_auc_score(test_DFS, DFS_en_test),

            'external_acc': accuracy_score(external_DFS, DFS_en_external_binarized),
            'external_mcc': matthews_corrcoef(external_DFS, DFS_en_external_binarized),
            'external_f1': f1_score(external_DFS, DFS_en_external_binarized),
            'external_f1_0': f1_score(1 - external_DFS, 1 - DFS_en_external_binarized),
            'external_auc': roc_auc_score(external_DFS, DFS_en_external)
        })
test_ensemble_selected_df = pd.DataFrame(test_ensemble_res_selected)
test_ensemble_selected_df.to_csv(
    f'ml_results/test_ensemble_{input_type}_selected_ml.csv', index=False)


print(test_ensemble_selected_df.groupby(
    ['name', 'target', 'test_fold']
).mean().reset_index().sort_values(['target', 'name', 'test_fold'], ascending=[True, True, False]).test_auc.to_string(index=False))


test_ensemble_selected_df.groupby(
    ['name', 'target', 'test_fold']
).mean().reset_index().sort_values(['target', 'name', 'test_fold'], ascending=[True, True, False])


test_ensemble_selected_df.groupby(['target', 'name']).mean()
