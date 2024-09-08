import os
import re
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModel


BASE_PATH = "/home/mkhokhar21/Documents/COSBI/Allostery_Paper/"
SEQ_ANNOTATED_PATH = os.path.join(BASE_PATH, "data/classification/processed_seq_annotated_partitioned")
pdb_train = {}
pdb_test = {}

s = os.getcwd()

with open(f"{SEQ_ANNOTATED_PATH}/seq_train.pkl", "rb") as f:
    pdb_train = pickle.load(f)

with open(f"{SEQ_ANNOTATED_PATH}/seq_test.pkl", "rb") as f:
    pdb_test = pickle.load(f)

LM_RESIDUE_PATH_BASE = os.path.join(BASE_PATH, "data/classification/LM_residue_partitioned_base")
train_path = f"{LM_RESIDUE_PATH_BASE}/LM_residue_train.pkl"
test_path = f"{LM_RESIDUE_PATH_BASE}/LM_residue_test.pkl"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = os.path.join(BASE_PATH, "prot_bert_base")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False )
model = BertModel.from_pretrained(MODEL_PATH)
model = model.to(device)
model = model.eval()

with torch.no_grad():
    for dataset in [[pdb_train, train_path], [pdb_test, test_path]]:
        seq_data_LM = []
        for pdb in tqdm(dataset[0]):
            seq = re.sub(r"[UZOB]", "X", " ".join(pdb["seq"]))
            encoding = tokenizer.batch_encode_plus(
                [seq],
                add_special_tokens=True,
                padding='max_length'
            )
            input_ids = torch.tensor(encoding['input_ids']).to(device)
            attention_mask = torch.tensor(encoding['attention_mask']).to(device)
            model_output = model(input_ids=input_ids, attention_mask=attention_mask)

            last_hidden_state = model_output[0]
            embedding = last_hidden_state.cpu().numpy()
            seq_len = (attention_mask[0] == 1).sum()
            token_emb = embedding[0][1:seq_len-1]
            seq_emb = token_emb.mean(axis=0)

            poc_labels = []
            poc_res_emb = []
            for i in range(len(pdb["poc_indices"])):
                add_pocket = True
                poc_labels.append(pdb["poc_labels"][i])
                cur_poc_emb = []
                for idx in pdb["poc_indices"][i]:
                    try:
                        token = token_emb[pdb["seq_indices"][idx]]
                        cur_poc_emb.append(token)
                    except Exception as e:
                        add_pocket = False
                        poc_labels.pop()
                        break

                if add_pocket:
                    poc_res_emb.append(cur_poc_emb)

            if sum(poc_labels) > 0:
                seq_data_LM.append({ **pdb, "seq_emb": seq_emb, "poc_res_emb": poc_res_emb })

        # with open(dataset[1], "wb") as f:
        #     pickle.dump(seq_data_LM, f)

#####################################################
#####################################################

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_labels_task1, num_labels_task2):
        super(MultiTaskModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.head1 = nn.Linear(self.encoder.config.hidden_size, num_labels_task1)
        self.head2 = nn.Linear(self.encoder.config.hidden_size, num_labels_task2)

    def forward(self, input1=None, input2=None):
        output1, output2 = None, None
        encoder_output1, encoder_output2 = None, None

        if input1 is not None:
            encoder_output1 = self.encoder(**input1).last_hidden_state
            output1 = self.head1(encoder_output1)

        if input2 is not None:
            encoder_output2 = self.encoder(**input2).last_hidden_state
            output2 = self.head2(encoder_output2)

        return (output1, output2), (encoder_output1, encoder_output2)

train_path = f"{LM_RESIDUE_PATH_BASE}/MTL_LM_residue_train.pkl"
test_path = f"{LM_RESIDUE_PATH_BASE}/MTL_LM_residue_test.pkl"

model = MultiTaskModel("Rostlab/prot_bert_bfd", 2, 3)
state_dict = torch.load(os.path.join(BASE_PATH, "prot_bert_mtl/prot_bert_mtl.bin"))
model.load_state_dict(state_dict)

model = model.to(device)
model = model.eval()

with torch.no_grad():
    for dataset in [[pdb_train, train_path], [pdb_test, test_path]]:
        seq_data_LM = []
        for pdb in tqdm(dataset[0]):
            seq = re.sub(r"[UZOB]", "X", " ".join(pdb["seq"]))
            encoding = tokenizer.batch_encode_plus(
                [seq],
                add_special_tokens=True,
                padding='max_length'
            )
            input_ids = torch.tensor(encoding['input_ids']).to(device)
            attention_mask = torch.tensor(encoding['attention_mask']).to(device)
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            _, (last_hidden_state, _) = model(input1=inputs)

            embedding = last_hidden_state.cpu().numpy()
            seq_len = (attention_mask[0] == 1).sum()
            token_emb = embedding[0][1:seq_len-1]
            seq_emb = token_emb.mean(axis=0)

            poc_labels = []
            poc_res_emb = []
            for i in range(len(pdb["poc_indices"])):
                add_pocket = True
                poc_labels.append(pdb["poc_labels"][i])
                cur_poc_emb = []
                for idx in pdb["poc_indices"][i]:
                    try:
                        token = token_emb[pdb["seq_indices"][idx]]
                        cur_poc_emb.append(token)
                    except Exception as e:
                        add_pocket = False
                        poc_labels.pop()
                        break

                if add_pocket:
                    poc_res_emb.append(cur_poc_emb)

            if sum(poc_labels) > 0:
                seq_data_LM.append({ **pdb, "seq_emb": seq_emb, "poc_res_emb": poc_res_emb })

        with open(dataset[1], "wb") as f:
            pickle.dump(seq_data_LM, f)
