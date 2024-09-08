import pickle
import torch
import random
import numpy as np

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, f1_score, precision_score, recall_score, matthews_corrcoef


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

def load_data(partition: str):
    pdb_data = None
    with open(f"{LM_RESIDUE_PATH}/MTL_LM_{partition}.pkl", "rb") as f:
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
            # X.append(np.concatenate((seq_emb, poc)))
            X.append(seq_emb)
            Y.append(pdb["poc_labels"][i])

    return X, Y

seed_everything()

# pdb_data = load_data("train")
# f1_all, prec_all, rec_all = [], [], []

# kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
# for train_index, test_index in kfold.split(pdb_data):
#     pdb_train, pdb_test = [pdb_data[i] for i in train_index], [pdb_data[i] for i in test_index]
#     x_train_fold, y_train_fold = get_res_data(pdb_train)
#     x_test_fold, y_test_fold = get_res_data(pdb_test)
#     x_train, x_test = np.array(x_train_fold), np.array(x_test_fold)
#     y_train, Y_Test = np.array(y_train_fold), np.array(y_test_fold)

#     scaler = StandardScaler()
#     x_train = scaler.fit_transform(x_train)
#     x_test = scaler.fit_transform(x_test)

#     # x_train, x_test = X_train[train_index], X_train[test_index]
#     # y_train, Y_Test = Y_train[train_index], Y_train[test_index]

#     model = svm.SVC(kernel='rbf', C=80)
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)

#     f1 = f1_score(Y_Test, y_pred)
#     prec = precision_score(Y_Test, y_pred)
#     rec = recall_score(Y_Test, y_pred)
#     f1_all.append(f1)
#     prec_all.append(prec)
#     rec_all.append(rec)

#     print(confusion_matrix(Y_Test, y_pred))
#     print(f"Acc: {accuracy_score(Y_Test, y_pred):.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | MCC: {matthews_corrcoef(Y_Test, y_pred):.4f}")

# print('--------------------------------------------------')
# print(f"Mean F1: {np.mean(f1_all):.4f} | Mean Prec: {np.mean(prec_all):.4f} | Mean Rec: {np.mean(rec_all):.4f}")
# print(f"Std F1: {np.std(f1_all):.4f} | Std Prec: {np.std(prec_all):.4f} | Std Rec: {np.std(rec_all):.4f}")

print("--------------------------------------------")
print("-------------------Test---------------------")
X_train, Y_train = get_res_data(load_data("train"))
x_test, y_test = get_res_data(load_data("test"))
X_train, Y_train = np.array(X_train), np.array(Y_train)
x_test, Y_Test = np.array(x_test), np.array(y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
x_test = scaler.fit_transform(x_test)

model = svm.SVC(kernel='rbf', C=7)
model.fit(X_train, Y_train)
y_pred = model.predict(x_test)
print(confusion_matrix(Y_Test, y_pred))
print(f"Acc: {accuracy_score(Y_Test, y_pred):.4f} | F1: {f1_score(Y_Test, y_pred):.4f} | Prec: {precision_score(Y_Test, y_pred):.4f} | Rec: {recall_score(Y_Test, y_pred):.4f} | MCC: {matthews_corrcoef(Y_Test, y_pred):.4f}")
