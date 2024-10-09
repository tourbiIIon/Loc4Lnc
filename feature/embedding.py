import torch
import pandas as pd 
import math
import numpy as np
from tqdm import tqdm
from enformer_pytorch import Enformer,seq_indices_to_one_hot,EnformerConfig
import numpy as np
from random import random

# helper functions

def exists(val):
    return val is not None

def identity(t):
    return t

def cast_list(t):
    return t if isinstance(t, list) else [t]

def coin_flip():
    return random() > 0.5

# genomic function transforms

seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord('a')] = 0
seq_indices_embed[ord('c')] = 1
seq_indices_embed[ord('g')] = 2
seq_indices_embed[ord('t')] = 3
seq_indices_embed[ord('n')] = 4
seq_indices_embed[ord('A')] = 0
seq_indices_embed[ord('C')] = 1
seq_indices_embed[ord('G')] = 2
seq_indices_embed[ord('T')] = 3
seq_indices_embed[ord('N')] = 4
seq_indices_embed[ord('.')] = -1

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord('a')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('c')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('g')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('t')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('n')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('A')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('C')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('G')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('T')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('N')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('.')] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()

def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = list(map(lambda t: np.frombuffer(t.encode('utf-8'), dtype=np.uint8), seq_strs))
    seq_chrs = list(map(lambda arr: torch.from_numpy(arr).clone(), np_seq_chrs))  # Use clone to ensure the tensor is writable
    return torch.stack(seq_chrs) if batched else seq_chrs[0]


def str_to_seq_indices(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]

def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]


tensor_path_test = "feature_independent_all.pt"
config = EnformerConfig.from_json_file("config_lnc.json")
model = Enformer.from_pretrained('EleutherAI/enformer-official-rough',config=config).cuda()


df_test = pd.read_table("../dataset/independent_test_dataset.txt") 
torch.save(torch.tensor(np.array(df_test.label)), "label_independent_all.pt")
tensor_list_train = []
tensor_list_test = []


for s in tqdm(df_test.sequence):
    with torch.no_grad():
        ind = str_to_seq_indices(s)
        if len(ind) < 10240:
            pad1 = torch.from_numpy(np.array([4]*(math.ceil((10240-len(ind))/2)))).clone()
            pad2 = torch.from_numpy(np.array([4]*((10240-len(ind))//2))).clone()
            ind = torch.cat((pad1, ind, pad2), 0)
        one_hot = seq_indices_to_one_hot(ind).cuda()
        one_hot = one_hot.unsqueeze(0)
        embeddings = model(one_hot, return_embeddings = True)[1].cuda()
        tensor_list_test.append(embeddings)


torch.save(tensor_list_test, tensor_path_test)



