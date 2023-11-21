import numpy as np
from utils import reprocess_tensor

def collate_fn_tensor(batch):
    len_arr = np.array([d["text"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // 32

    cut_list = list()
    for i in range(32):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(32):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output
