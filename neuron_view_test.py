import gc
import re
import torch
import torch.nn as nn
import requests
import glob, os, math
import numpy as np
from utils.bertviz.bertviz.transformers_neuron_view import BertModel, BertTokenizer
# from transformers import BertTokenizer, AutoModel,BertTokenizerFast
from utils.bertviz.bertviz.neuron_view import show
from bertviz import head_view, model_view
# import xgboost as xgb

from utils.extract_sequence import extract_sequence
from utils.pocket_feature import pocket_feature
from utils.sequence_indices import sequence_indices
from utils.pocket_coordinates import pocket_coordinates


N_ATOMS = 9
BASE_PATH = "/home/mkhokhar21/Documents/COSBI/Allostery_Paper"
MODEL_PATH = f"{BASE_PATH}/prot_bert_mtl"
TOKENIZER_PATH = "/home/mkhokhar21/Documents/COSBI/Thesis_Work/PASSerRank-main/prot_bert_allosteric"
base_url = "https://files.rcsb.org/download"
pdb_dir = f"{BASE_PATH}/data/pdbs/"
pocket_dir = f"{BASE_PATH}/data/pockets/"

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_labels_task1, num_labels_task2):
        super(MultiTaskModel, self).__init__()
        self.encoder = BertModel.from_pretrained(model_name, output_attentions=True)
        self.head1 = nn.Linear(self.encoder.config.hidden_size, num_labels_task1)
        self.head2 = nn.Linear(self.encoder.config.hidden_size, num_labels_task2)

    def forward(self, input1=None, input2=None):
        output1, output2 = None, None
        attentions1, attentions2 = None, None

        if input1 is not None:
            # Pass output_attentions=True to get attention outputs
            encoder_output1 = self.encoder(**input1, output_attentions=True)
            output1 = self.head1(encoder_output1.last_hidden_state)
            attentions1 = encoder_output1.attentions  # Extract attention outputs

        if input2 is not None:
            encoder_output2 = self.encoder(**input2, output_attentions=True)
            output2 = self.head2(encoder_output2.last_hidden_state)
            attentions2 = encoder_output2.attentions  # Extract attention outputs

        return (output1, output2), (attentions1, attentions2)

pdb_id = "5DKK"
chain_id = "A"

pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")

if not os.path.exists(pdb_path):
    response = requests.get(f"{base_url}/{pdb_id}.pdb")
    if response.status_code == 200:  # Check if the request was successful
        with open(pdb_path, 'wb') as file:
            file.write(response.content)
        print(f"PDB file {pdb_id}.pdb downloaded successfully.")
    else:
        raise Exception(f"Failed to download {pdb_id}.pdb. Check if the PDB ID is correct.")

sequence = extract_sequence(pdb_path, chain_id)
seq = " ".join(sequence)
seq = re.sub(r"[UZOB-]", "X", seq)

model = MultiTaskModel(TOKENIZER_PATH, 2, 3)
state_dict = torch.load(f"{MODEL_PATH}/prot_bert_mtl.bin")
model.load_state_dict(state_dict)

# tokenizer = BertTokenizerFast.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False )

# from utils.bertviz.bertviz.transformers_neuron_view import BertModel, BertTokenizer

model_type = 'bert'
# model = BertModel.from_pretrained(MODEL_PATH, output_attentions=True)

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH, do_lower_case=False )

html_neuron_view = show(model.encoder, model_type, tokenizer, seq, layer=7, head=3, html_action='return', display_mode='light')
with open("./neuron_view.html", 'w') as file:
    file.write(html_neuron_view.data)
