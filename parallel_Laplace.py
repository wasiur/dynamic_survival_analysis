from mpi4py import MPI
import pickle
import math
import pandas as pd
import time


from epidemiccore_w import *
import os as os


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def laplace_p(theta, n, plot_T=150, bounds=[(0.1,1),(0.1,1),(1E-9,1E-4)]):
    N = min(1000,np.size(df))
    epi = Epidemic(file_or_df=df, bounds=bounds, abc=tuple(theta), plot_T=plot_T)
    epi.fit(N=N)
    epi.simulate_and_fit_parallel_laplace(N=N, n=n, rank=rank)
    fitted_parms = pd.DataFrame(
        {'a': list(f.a for f in epi.fits),
         'b': list(f.b for f in epi.fits),
         'c': list(f.c for f in epi.fits),
         'kinfty': list(f.kinfty for f in epi.fits),
         'R0': list(f.R0 for f in epi.fits),
         'n': list(f.n for f in epi.fits),
         'tau': list(f.tau for f in epi.fits),
         'rho': list(f.tau for f in epi.fits)})
    return fitted_parms


def laplace_sample_in_parallel(start_idx, end_idx):
    colNames = ['a', 'b', 'c', 'kinfty', 'R0', 'n', 'tau', 'rho']
    res = pd.DataFrame(columns=colNames)
    for index in range(start_idx, end_idx):
        temp = laplace_p(df, thetas[index], n=100, plot_T=150)
        res = res.append(temp, ignore_index=True)
    comm.send(res, dest=0, tag=rank)



if __name__ == '__main__':
    thetas = pickle.load(open("thetas", "rb"))
    df = pickle.load(open("df", "rb"))
    df_recovery = pickle.load(open("df_recovery", "rb"))
    ln = len(thetas)
    size = comm.Get_size()

    each_process = int(ln / float(size))
    start_id = rank * each_process
    end_id = (rank + 1) * each_process
    if rank == size - 1:
        end_id = ln

    comm.Barrier()

    laplace_sample_in_parallel(start_idx=start_id, end_idx=end_id)

    if rank == 0:
        colNames = ['a', 'b', 'c', 'kinfty', 'R0', 'n', 'tau', 'rho']
        res = pd.DataFrame(columns=colNames)
        for other_rank in range(size):
            start_id_local = other_rank * each_process
            end_id_local = (other_rank + 1) * each_process
            if other_rank == size - 1:
                end_id_local = ln
            res_tmp = comm.recv(source=other_rank, tag=other_rank)
            res = res.append(res_tmp, ignore_index=True)

        if os.path.exists("thetas_fitted"):
            os.remove("thetas_fitted")
        pickle.dump(res, open("thetas_fitted", "wb"), protocol=3)
    comm.Barrier()


