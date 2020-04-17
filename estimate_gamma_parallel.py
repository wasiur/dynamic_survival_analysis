from mpi4py import MPI
import pickle
import math
import time

from epidemiccore_w import *


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


today = pd.to_datetime('today')

root_folder = os.getcwd()
data_folder = os.path.join(root_folder,'data')

fname = 'dsa_dict' + today.strftime("%m%d") + '.pkl'
dsa_dict = pickle.load(open(os.path.join(data_folder,fname), "rb"))
datafile = dsa_dict['datafile']
plot_folder = dsa_dict['plot_folder']
location = dsa_dict['location']
last_date = dsa_dict['last_date']
estimate_gamma = dsa_dict['estimate_gamma']
ifsmooth = dsa_dict['ifsmooth']
ifMPI = dsa_dict['ifMPI']

plot_folder = os.path.join(root_folder,plot_folder)



def estimate_gamma_parallel(epi, start_idx, end_idx):
    nSim = end_idx - start_idx
    gamma = np.zeros(nSim, dtype=np.float64)
    offset = np.zeros(nSim, dtype=np.float64)
    m = epi.theta
    cov = epi.cov_abc()
    samples = parm_sample_correlated(m, cov, nSim)
    for index in range(nSim):
        theta = samples[index]
        gamma[index], offset[index] = epi.estimate_gamma_sample(theta, df_recovery = df_recovery, N=N, x0=(0.1, -5),
                                       bounds=[(1.0 / 25, 1.0 / 5), (-10, 0)], approach='offset')
        # res.append((index, gamma, offset))
    res = pd.DataFrame(
        {'gamma': list(gamma),
         'offset': list(offset)}
    )
    comm.send(res, dest=0, tag=rank)

if __name__ == '__main__':
    fname = location + '_epi_' + today.strftime("%m%d") + '.pkl'
    epiT = pickle.load(open(os.path.join(plot_folder, fname), "rb"))
    df_recovery = pickle.load(open("df_recovery", "rb"))
    N = 2000
    ln = len(epiT.fits)
    size = comm.Get_size()

    each_process = int(ln / float(size))
    start_id = rank * each_process
    end_id = (rank + 1) * each_process
    if rank == size - 1:
        end_id = ln

    comm.Barrier()
    estimate_gamma_parallel(epiT, start_idx=start_id, end_idx=end_id)

    if rank == 0:
        colNames = ['gamma', 'offset']
        res = pd.DataFrame(columns=colNames)
        gamma_fitted_dict = dict()
        gamma_fitted = []
        for other_rank in range(size):
            start_id_local = other_rank * each_process
            end_id_local = (other_rank + 1) * each_process
            if other_rank == size - 1:
                end_id_local = ln
            res_tmp = comm.recv(source=other_rank, tag=other_rank)
            res = res.append(res_tmp, ignore_index=True)

        fname = location + '_gamma_fitted_' + today.strftime("%m%d") + '.csv'
        res.to_csv(os.path.join(plot_folder, fname))

    comm.Barrier()
