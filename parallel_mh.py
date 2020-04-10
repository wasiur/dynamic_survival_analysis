import sys
import os as os
from mpi4py import MPI
import numpy as np
import pickle
import math
import time

from my_mh import *

today = pd.to_datetime('today')
root_folder = os.getcwd()
Data_Folder = os.path.join(root_folder,'data_for_server')

fname = 'mh_dict' + today.strftime("%m%d") + '.pkl'
mh_dict = pickle.load(open(os.path.join(Data_Folder,fname), "rb"))
datafile = mh_dict['datafile']
plot_folder = mh_dict['plot_folder']
location = mh_dict['location']
burn_in = np.int(mh_dict['burn_in'])
nChains = np.int(mh_dict['nChains'])
plot_folder = os.path.join(root_folder,plot_folder)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def mh_in_parallel(start_idx, end_idx, chain_length = 10**5):
    loc_nChains = end_idx - start_idx
    res_a = np.zeros((loc_nChains,chain_length), dtype=np.float)
    res_b = np.zeros((loc_nChains,chain_length), dtype=np.float)
    res_c = np.zeros((loc_nChains,chain_length), dtype=np.float)
    for idx in range(start_idx, end_idx):
        temp = mh(epi, p, proposal, burn_in, chain_length)
        res_a[idx] = temp[:,0]
        res_b[idx] = temp[:,1]
        res_c[idx] = temp[:,2]

    res = {'a' : res_a, 'b' : res_b, 'c' : res_c}
    comm.send(res, dest=0, tag=rank)


if __name__ == '__main__':
    fname = location + '_epi_' + today.strftime("%m%d") + '.pkl'
    epi = pickle.load(open(os.path.join(plot_folder, fname), "rb"))
    fname = 'bayes' + today.strftime("%m%d")
    bayes = pickle.load(open(os.path.join(plot_folder, fname), 'rb'))
    p = bayes["p"]
    proposal = bayes["proposal"]
    chain_length = bayes["chain_length"]
    burn_in = bayes["burn_in"]
    nChains = bayes["nChains"]
    size = comm.Get_size()

    each_process = int(nChains / float(size))
    start_id = rank * each_process
    end_id = (rank + 1) * each_process
    if rank == size - 1:
        end_id = nChains

    comm.Barrier()
    mh_in_parallel(start_idx=start_id, end_idx=end_id, chain_length=chain_length)

    if rank == 0:
        chain_temp = mh(epi, p, proposal, burn_in, chain_length)
        res_a = chain_temp[:,0]
        res_b = chain_temp[:,1]
        res_c = chain_temp[:,2]
        for other_rank in range(size):
            start_id_local = other_rank * each_process
            end_id_local = (other_rank + 1) * each_process
            if other_rank == size - 1:
                end_id_local = nChains
            res_tmp = comm.recv(source=other_rank, tag=other_rank)
            a = res_tmp["a"]
            b = res_tmp["b"]
            c = res_tmp["c"]
            np.append(res_a, a)
            np.append(res_b, b)
            np.append(res_c, c)

        res = {'a': res_a, 'b': res_b, 'c': res_c}
        fname = "mh_chains" + today.strftime("%m%d")
        pickle.dump(res, open(os.path.join(plot_folder, fname), "wb"), protocol=3)
    comm.Barrier()
