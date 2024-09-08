import pickle
import torch
import random
import sys
import warnings
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor

warnings.filterwarnings('ignore')


LM_RESIDUE_PATH = "../data/classification/LM_residue_partitioned"
LM_RESIDUE_PATH = "/home/mkhokhar21/Documents/COSBI/Allostery_Paper/data/classification/LM_residue_partitioned"
THRESHOLD = 0.5
SEED = 42

def seed_everything(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

seed_everything()

def load_data(partition: str):
    pdb_data = None
    with open(f"{LM_RESIDUE_PATH}/LM_{partition}.pkl", "rb") as f:
        pdb_data = pickle.load(f)

    return pdb_data

def get_res_data(pdb_data: list):
    X = []
    Y = []

    for pdb in pdb_data:
        for i in range(min(len(pdb['poc_res_emb']), len(pdb['pocket_coordinates']))): # iterate over pockets
            seq_emb = []
            for res_idx in range(min(len(pdb['poc_res_emb'][i]), len(pdb['pocket_coordinates'][i]))):
                seq_emb.append(pdb['poc_res_emb'][i][res_idx])
            seq_emb = np.array(seq_emb).mean(axis=0)
            poc = pdb["poc_emb"][i]
            X.append(np.concatenate((seq_emb, poc)))
            # X.append(seq_emb)
            Y.append(pdb["poc_labels"][i])

    return X, Y

x_train, y_train = get_res_data(load_data("train"))
# x_val, y_val = get_res_data(load_data("val"))
x_test, y_test = get_res_data(load_data("test"))

# x_train.extend(x_val)
# y_train.extend(y_val)

x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)

train_data = np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)
test_data = np.concatenate((x_test, y_test.reshape(-1, 1)), axis=1)

train_data = TabularDataset(train_data)
test_data = TabularDataset(test_data)

train_data.columns = [str(i) for i in range(1, x_train.shape[1] + 2)]
test_data.columns = [str(i) for i in range(1, x_train.shape[1] + 2)]


label = str(x_train.shape[1] + 1)
predictor = TabularPredictor(label=label, eval_metric='f1').fit(
        train_data, presets='best_quality'
        , ag_args_fit={'num_gpus': 1}
    )

# individual models performance
#leaderboard = predictor.leaderboard(test_data)

predictor.fit_summary()

# validation results
# y_val_label = valid_data['20']
# y_val_nolab = valid_data.drop(columns=['20'])

# y_pred_val = predictor.predict(y_val_nolab)
# perf_val = predictor.evaluate_predictions(
#     y_true=y_val_label, y_pred=y_pred_val, auxiliary_metrics=True)


# testing results
y_test_label = test_data[label]
y_test_nolab = test_data.drop(columns=[label])

predictor.leaderboard(test_data, extra_metrics=['precision', 'recall'])

y_pred_test = predictor.predict(y_test_nolab)
perf_test = predictor.evaluate_predictions(
    y_true=y_test_label, y_pred=y_pred_test, auxiliary_metrics=True, detailed_report=True)

print(perf_test)

