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
from tqdm import tqdm
from utils.top_accuracies import calculate_top_n_accuracies, topk, topN

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
            # poc_X.append(poc)
            Y.append(pdb["poc_labels"][i])

    return X, Y

x_train, y_train = get_res_data(load_data("train"))
x_test, y_test = get_res_data(load_data("test"))


x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)

train_data = np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)
test_data = np.concatenate((x_test, y_test.reshape(-1, 1)), axis=1)

train_data = TabularDataset(train_data)
test_data = TabularDataset(test_data)

train_data.columns = [str(i) for i in range(1, x_train.shape[1] + 2)]
test_data.columns = [str(i) for i in range(1, x_train.shape[1] + 2)]


label = str(x_train.shape[1] + 1)

predictor = TabularPredictor.load("/home/mkhokhar21/Documents/COSBI/Allostery_Paper/src/AutogluonModels/LM_All")

# predictor.fit_summary()

y_test_label = test_data[label]
y_test_nolab = test_data.drop(columns=[label])

# predictor.leaderboard(test_data, extra_metrics=['precision', 'recall'])

y_pred_test = predictor.predict_proba(y_test_nolab)
# perf_test = predictor.evaluate_predictions(
#     y_true=y_test_label, y_pred=y_pred_test, auxiliary_metrics=True, detailed_report=True)

# print(perf_test)

y_test_label = y_test_label.to_numpy()
y_pred_test = y_pred_test.to_numpy()[:, 1]

topPCT = topN(y_pred_test, y_test_label)
print(f"Top1%: {topPCT[0]:.4f} | Top3%: {topPCT[1]:.4f} | Top5%: {topPCT[2]:.4f} | Top10%: {topPCT[3]:.4f} | Top20%: {topPCT[4]:.4f}")

Y_True = []
Y_Pred = []

test_data = load_data("test")
for i in tqdm(range(len(test_data))):
    x_test, y_test = get_res_data([test_data[i]])
    x_test, y_test = np.array(x_test), np.array(y_test)
    test_row = np.concatenate((x_test, y_test.reshape(-1, 1)), axis=1)
    test_row = TabularDataset(test_row)
    test_row.columns = [str(i) for i in range(1, x_train.shape[1] + 2)]
    label = str(x_train.shape[1] + 1)

    y_test_label = test_row[label]
    y_test_nolab = test_row.drop(columns=[label])

    y_pred_test = predictor.predict(y_test_nolab)

    y_test_label = y_test_label.to_numpy()
    y_pred_test = y_pred_test.to_numpy()


    Y_True.append(y_test_label)
    Y_Pred.append(y_pred_test)

top3 = calculate_top_n_accuracies(Y_True, Y_Pred)

print(f"Top1: {top3['top_1']:.4f} | Top2: {top3['top_2']:.4f} | Top3: {top3['top_3']:.4f}")
