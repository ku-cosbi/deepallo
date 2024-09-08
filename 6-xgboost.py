import copy
import pickle
import numpy as np

from scipy.stats import uniform, randint

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import xgboost as xgb

from utils import calculate_top_n_accuracies, topk, topN

LM_RESIDUE_PATH = "../data/classification/LM_residue_partitioned"
LM_RESIDUE_PATH = "/home/mkhokhar21/Documents/COSBI/Allostery_Paper/data/classification/LM_residue_partitioned"
THRESHOLD = 0.5

def load_data(partition: str):
    pdb_data = None
    with open(f"{LM_RESIDUE_PATH}/MTL_LM_{partition}.pkl", "rb") as f:
        pdb_data = pickle.load(f)

    return pdb_data

def get_data(pdb_data: list, is_test=False):
    X = []
    Y = []

    if is_test:
        for pdb in pdb_data:
            for i in range(len(pdb["poc_labels"])):
                poc = pdb["poc_emb"][i]
                seq = pdb["seq_emb"]
                X.append(np.concatenate((seq, poc)))
                # X.append(poc)
                Y.append(pdb["poc_labels"][i])
    else:
        for pdb in pdb_data:
            pos_idx = np.where(np.array(pdb["poc_labels"]) == 1)[0]
            for idx in pos_idx:
                poc = pdb["poc_emb"][idx]
                seq = pdb["seq_emb"]
                X.append(np.concatenate((seq, poc)))
                # X.append(poc)
                Y.append(pdb["poc_labels"][idx])
            neg_idx = np.where(np.array(pdb["poc_labels"]) == 0)[0]
            for i in range(min(len(pos_idx) * 1, len(neg_idx))):
                poc = pdb["poc_emb"][neg_idx[i]]
                seq = pdb["seq_emb"]
                X.append(np.concatenate((seq, poc)))
                # X.append(poc)
                Y.append(pdb["poc_labels"][neg_idx[i]])

    return X, Y

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

    # pca_X = np.array(X)
    # pca = PCA(n_components=256)
    # X = pca.fit_transform(pca_X).tolist()

    # for i in range(len(reduced_data)):
    #     X.append(np.concatenate((reduced_data[i], poc_X[i])))

    return X, Y

def get_res_data_new(pdb_data: list):
    X = []
    Y = []

    for pdb in pdb_data:
        for i in range(min(len(pdb['poc_res_emb']), len(pdb['poc_labels']))): # iterate over pockets
            seq_emb = []
            for res_idx in range(len(pdb['poc_res_emb'][i])):
                seq_emb.append(pdb['poc_res_emb'][i][res_idx])
            seq_emb = np.array(seq_emb).mean(axis=0)
            poc = pdb["poc_emb"][i]
            X.append(np.concatenate((seq_emb, poc)))
            # X.append(seq_emb)
            Y.append(pdb["poc_labels"][i])

    return X, Y

def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    # Make binary prediction
    y_pred_binary = np.where(y_pred > THRESHOLD, 1, 0)
    return 'f1', -f1_score(y_true, y_pred_binary)

x_train, y_train = get_res_data(load_data("train"))
# x_val, y_val = get_res_data(load_data("val"))
# x_test, y_test = get_data(load_data("test"))

# x_train.extend(x_val)
# y_train.extend(y_val)
# x_train.extend(x_test)
# y_train.extend(y_test)

# x_train, X_Test, y_train, Y_Test = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

x_train_fold = np.array(x_train)
y_train_fold = np.array(y_train)


# kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# # pdb_data = load_data("train")
# # pdb_data.extend(load_data("test"))

# best_model = None
# best_f1 = 0

# f1_all, prec_all, rec_all = [], [], []

# for train_index, test_index in kfold.split(x_train_fold):
#     # pdb_train, pdb_test = [pdb_data[i] for i in train_index], [pdb_data[i] for i in test_index]
#     # x_train_fold, y_train_fold = get_res_data_new(pdb_train)
#     # x_test_fold, y_test_fold = get_res_data_new(pdb_test)
#     # x_train, x_test = np.array(x_train_fold), np.array(x_test_fold)
#     # y_train, y_test = np.array(y_train_fold), np.array(y_test_fold)

#     x_train, x_test = x_train_fold[train_index], x_train_fold[test_index]
#     y_train, y_test = y_train_fold[train_index], y_train_fold[test_index]

#     # with open(f"{LM_RESIDUE_PATH}/MTL_LM_train.pkl", "wb") as f:
#     #     pickle.dump(pdb_train, f)

#     # with open(f"{LM_RESIDUE_PATH}/MTL_LM_test.pkl", "wb") as f:
#     #     pickle.dump(pdb_test, f)

#     dtrain = xgb.DMatrix(x_train, label=y_train)
#     dtest = xgb.DMatrix(x_test, label=y_test)
#     params = {
#         'objective': 'binary:logistic',
#         'max_depth': 7,
#         'lambda': 0.15,
#         'scale_pos_weight': (sum(y_train==0) / sum(y_train))
#     }
#     num_rounds = 100
#     evallist = [(dtrain, 'train'), (dtest, 'eval')]
#     bst = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=evallist, custom_metric=f1_eval, maximize=True, verbose_eval=False)
#     y_pred = (bst.predict(dtest) > THRESHOLD).astype(int)
#     f1 = f1_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred)
#     rec = recall_score(y_test, y_pred)
#     f1_all.append(f1)
#     prec_all.append(prec)
#     rec_all.append(rec)
#     print(confusion_matrix(y_test, y_pred))
#     print(f"F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

#     # if best_f1 < f1:
#     #     best_f1 = f1
#     #     best_model = copy.deepcopy(bst)
#     #     best_model.save_model('xgboost.model')
#     #     break

# print('--------------------------------------------------')
# print(f"Mean F1: {np.mean(f1_all):.4f} | Mean Prec: {np.mean(prec_all):.4f} | Mean Rec: {np.mean(rec_all):.4f}")
# print(f"Std F1: {np.std(f1_all):.4f} | Std Prec: {np.std(prec_all):.4f} | Std Rec: {np.std(rec_all):.4f}")

dtrain = xgb.DMatrix(x_train_fold, label=y_train_fold)
params = {
    'objective': 'binary:logistic',
    'max_depth': 7,
    'lambda': 0.15,
    'scale_pos_weight': (sum(y_train_fold==0) / sum(y_train_fold))
}
num_rounds = 100
best_model = xgb.train(params, dtrain, num_boost_round=num_rounds, custom_metric=f1_eval, maximize=True, verbose_eval=False)

print('--------------------------------------------------')
print('-----------------------TEST-----------------------')
test_data = load_data("test")
X_Test, Y_Test = get_res_data(test_data)
dtest = xgb.DMatrix(X_Test, label=Y_Test)
# y_pred = (best_model.predict(dtest) > THRESHOLD).astype(int)
# f1 = f1_score(Y_Test, y_pred)
# print(confusion_matrix(Y_Test, y_pred))
# print(f"Acc: {accuracy_score(Y_Test, y_pred):.4f} | F1: {f1:.4f} | Prec: {precision_score(Y_Test, y_pred):.4f} | Rec: {recall_score(Y_Test, y_pred):.4f} | MCC: {matthews_corrcoef(Y_Test, y_pred):.4f} | ROC-AUC: {roc_auc_score(Y_Test, y_pred):.4f}")

topPCT = topN(best_model.predict(dtest), Y_Test)
Y_True = []
Y_Pred = []

for i in range(len(test_data)):
  X_Test, Y_Test = get_res_data([test_data[i]])
  dtest = xgb.DMatrix(X_Test, label=Y_Test)
  y_pred = best_model.predict(dtest)

  Y_True.append(Y_Test)
  Y_Pred.append(y_pred)

top3 = calculate_top_n_accuracies(Y_True, Y_Pred)

print(f"Top1: {top3['top_1']:.4f} | Top2: {top3['top_2']:.4f} | Top3: {top3['top_3']:.4f}")
print(f"Top1%: {topPCT[0]:.4f} | Top3%: {topPCT[1]:.4f} | Top5%: {topPCT[2]:.4f} | Top10%: {topPCT[3]:.4f} | Top20%: {topPCT[4]:.4f}")
