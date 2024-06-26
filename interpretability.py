"""
Example of running a single experiment of unet in the head and neck data.
The json config of the main model is 'examples/json/unet-sample-config.json'
All experiment outputs are stored in '../../hn_perf/logs'.
After running 3 epochs, the performance of the training process can be accessed
as log file and perforamance plot.
In addition, we can peek the result of 42 first images from prediction set.
"""

import customize_obj
# import h5py
# from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
# from deoxys.model.callbacks import PredictionCheckpoint
# from deoxys.utils import read_file
import argparse
# import os
from deoxys.utils import read_csv
import numpy as np
# from pathlib import Path
# from comet_ml import Experiment as CometEx
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
import h5py
import gc
import pandas as pd


class Matthews_corrcoef_scorer:
    def __call__(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)

    def _score_func(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)


metrics.SCORERS['mcc'] = Matthews_corrcoef_scorer()

try:
    metrics._scorer._SCORERS['mcc'] = Matthews_corrcoef_scorer()
except:
    pass


def metric_avg_score(res_df, postprocessor):
    auc = res_df['AUC']
    mcc = res_df['mcc'] / 2 + 0.5
    f1 = res_df['f1']
    f0 = res_df['f1_0']

    # get f1 score in train data
    epochs = res_df['epochs']
    train_df = read_csv(
        postprocessor.log_base_path + '/logs.csv')
    train_df['real_epoch'] = train_df['epoch'] + 1
    train_f1 = train_df[train_df.real_epoch.isin(epochs)]['BinaryFbeta'].values
    train_f1 = 2 * np.sqrt(train_f1) / 3

    res_df['avg_score'] = (auc + mcc + f1 + 0.75*f0 + 0.75*train_f1) / 4.5

    return res_df


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("log_folder")
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--model_checkpoint_period", default=1, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=1, type=int)
    parser.add_argument("--meta", default='patient_idx', type=str)
    parser.add_argument(
        "--monitor", default='avg_score', type=str)
    parser.add_argument(
        "--monitor_mode", default='max', type=str)
    parser.add_argument("--memory_limit", default=0, type=int)

    args, unknown = parser.parse_known_args()

    if args.memory_limit:
        # Restrict TensorFlow to only allocate X-GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=1024 * args.memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    if '2d' in args.log_folder:
        meta = args.meta
    else:
        meta = args.meta.split(',')[0]

    print('external config from', args.dataset_file,
          'and explaining on models in', args.log_folder)
    print('Unprocesssed prediction are saved to', args.temp_folder)

    def binarize(targets, predictions):
        return targets, (predictions > 0.5).astype(targets.dtype)

    def flip(targets, predictions):
        return 1 - targets, 1 - (predictions > 0.5).astype(targets.dtype)

    class_weight = None
    if 'LRC' in args.log_folder:
        class_weight = {0: 0.7, 1: 1.9}

    exp = DefaultExperimentPipeline(
        log_base_path=args.log_folder,
        temp_base_path=args.temp_folder
    ).load_best_model(
        monitor=args.monitor,
        use_raw_log=False,
        mode=args.monitor_mode,
        custom_modifier_fn=metric_avg_score
    )

    model = exp.model.model
    dr = exp.model.data_reader

    # test_gen = dr.test_generator
    # steps_per_epoch = test_gen.total_batch
    # batch_size = test_gen.batch_size
    # rolling indice
    roll_indice = [None, 0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    selected_indice = [i for i in range(40) if i == 0 or i % 8 != 0]
    # # pids
    pids = []
    with h5py.File(exp.post_processors.dataset_filename) as f:
        for fold in test_gen.folds:
            pids.append(f[fold][meta][:])
    pids = np.concatenate(pids)
    print('Checking patients:', ', '.join([str(id) for id in pids]))

    with h5py.File(args.log_folder + '/ous_test.h5', 'w') as f:
        print('created file', args.log_folder + '/ous_test.h5')

    tf_dtype = model.inputs[0].dtype
    print('TF dtype', tf_dtype)
    data_gen = test_gen.generate()
    i = 0
    sub_idx = 0
    for x, _ in data_gen:
        print(f'Batch {i+1}/{steps_per_epoch}')
        np_random_gen = np.random.default_rng(1123)
        new_shape = list(x.shape) + [40]
        var_grad = np.zeros(new_shape)
        for trial in range(40):
            print(f'Trial {trial+1}/40')
            roll_idx = roll_indice[trial % 8]
            if roll_idx:
                x_noised = np.roll(x, 1 + trial//8, roll_idx) + \
                    np_random_gen.normal(loc=0.0, scale=.05, size=x.shape)
            else:
                x_noised = x + \
                    np_random_gen.normal(loc=0.0, scale=.05, size=x.shape)
            x_noised = tf.Variable(x_noised, dtype=tf_dtype)
            with tf.GradientTape() as tape:
                tape.watch(x_noised)
                pred = model(x_noised)
            grads = tape.gradient(pred, x_noised).numpy()
            if roll_idx:
                var_grad[..., trial] = np.roll(grads, -1 - trial//8, roll_idx)
            else:
                var_grad[..., trial] = grads
        final_var_grad = var_grad[..., selected_indice].std(axis=-1)**2
        with h5py.File(args.log_folder + '/ous_test.h5', 'a') as f:
            for b_idx, pid in enumerate(pids[sub_idx: sub_idx + x.shape[0]]):
                f.create_dataset(str(pid), data=final_var_grad[b_idx])
        sub_idx += x.shape[0]
        i += 1
        gc.collect()
        if i == steps_per_epoch:
            break

    # print('Finish running data on OUS dataset')
    print('Loading MAASTRO dataset')
    seed = 1

    exp.load_new_dataset(args.dataset_file)
    model = exp.model.model
    dr = exp.model.data_reader

    test_gen = dr.test_generator
    steps_per_epoch = test_gen.total_batch
    batch_size = test_gen.batch_size
    # pids
    pids = []
    with h5py.File(exp.post_processors.dataset_filename) as f:
        for fold in test_gen.folds:
            pids.append(f[fold][meta][:])
    pids = np.concatenate(pids)

    with h5py.File(args.log_folder + f'/maastro_mc_{seed}.h5', 'w') as f:
        print('created file', args.log_folder + f'/maastro_mc_{seed}.h5')
    data_gen = test_gen.generate()
    i = 0
    sub_idx = 0
    mc_preds = []
    tta_preds = []
    for x, _ in data_gen:
        print('MC results ....')
        tf.random.set_seed(seed)
        mc_pred = model(x, training=True).numpy().flatten()
        mc_preds.append(mc_pred)
        print(f'Batch {i}/{steps_per_epoch}')
        np_random_gen = np.random.default_rng(1123)
        new_shape = list(x.shape) + [40]
        var_grad = np.zeros(new_shape)
        tta_pred = np.zeros((x.shape[0], 40))
        for trial in range(40):
            print(f'Trial {trial+1}/40')
            roll_idx = roll_indice[trial % 8]
            if roll_idx:
                x_noised = np.roll(x, 1 + trial//8, roll_idx) + \
                    np_random_gen.normal(loc=0.0, scale=.05, size=x.shape)
            else:
                x_noised = x + \
                    np_random_gen.normal(loc=0.0, scale=.05, size=x.shape)
            x_noised = tf.Variable(x_noised)
            tf.random.set_seed(seed)
            with tf.GradientTape() as tape:
                tape.watch(x_noised)
                pred = model(x_noised, training=True)

            grads = tape.gradient(pred, x_noised).numpy()
            if roll_idx:
                var_grad[..., trial] = np.roll(grads, -1 - trial//8, roll_idx)
            else:
                var_grad[..., trial] = grads

            tta_pred[..., trial] = pred.numpy().flatten()

        tta_preds.append(tta_pred)
        final_var_grad = var_grad[..., selected_indice].std(axis=-1)**2
        with h5py.File(args.log_folder + f'/maastro_mc_{seed}.h5', 'a') as f:
            for b_idx, pid in enumerate(pids[sub_idx: sub_idx + x.shape[0]]):
                f.create_dataset(str(pid), data=final_var_grad[b_idx])
        sub_idx += x.shape[0]
        i += 1
        gc.collect()
        if i == steps_per_epoch:
            break

    df = pd.DataFrame({'pid': pids, 'predicted': np.concatenate(mc_preds)})
    tta_preds = np.concatenate(tta_preds)
    for trial in range(40):
        df[f'tta_pred_{trial}'] = tta_preds[..., trial]

    df.to_csv(
        args.log_folder + f'/mc_predicted_{seed}.csv', index=False)
