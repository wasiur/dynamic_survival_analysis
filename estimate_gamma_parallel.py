from mpi4py import MPI
import pickle
import math
import time
from optparse import OptionParser
from dsacore import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main():
    """
    Estimates parameters of the recovery distribution. Use MPI-based parallelization.
    :return: A csv file with estimates of gamma and offsets.
    """
    usage = "Usage: mpiexec -n <threads> python estimate_gamma_parallel.py -d <dsaobj> -l <location> -o <output_folder>"
    parser = OptionParser(usage)
    parser.add_option("-d", "--dsa-object", action="store", type="string", dest="dsaobj_name",
                      help="Name of the DSA object.")
    parser.add_option("-l", "--location", action="store", type="string", dest="location",
                      default="Ohio", help="Name of the location.")
    parser.add_option("-o", "--output-folder", action="store", dest="output_folder",
                      default="plots_dsa_bayesian", help="Name of the output folder")
    parser.add_option("-N", action="store", dest="N",
                      default=2000, type="int",
                      help="Size of the random sample")
    (options, args) = parser.parse_args()

    today = pd.to_datetime('today')
    root_folder = os.getcwd()

    if options.dsaobj_name is None:
        parser.error("Please provide the DSA object")
    else:
        fname = options.dsaobj_name
        output_folder = options.output_folder
        plot_folder = os.path.join(root_folder, output_folder)
        dsaobj = pickle.load(open(os.path.join(plot_folder, fname), "rb"))
    df_recovery = pickle.load(open("df_recovery", "rb"))
    N = options.N
    location = options.location
    a_samples = list(f.a for f in dsaobj.fits)
    b_samples = list(f.b for f in dsaobj.fits)
    rho_samples = list(f.rho for f in dsaobj.fits)
    ln = len(a_samples)
    size = comm.Get_size()

    each_process = int(ln / float(size))
    start_id = rank * each_process
    end_id = (rank + 1) * each_process
    if rank == size - 1:
        end_id = ln

    comm.Barrier()
    estimate_gamma_in_parallel(dsaobj, a_samples=a_samples,
                               b_samples=b_samples, rho_samples=rho_samples,
                               df_recovery=df_recovery,
                               start_idx=start_id, end_idx=end_id, N=N)

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

        fname = location + '_gammas_fitted_' + today.strftime("%m%d") + '.csv'
        res.to_csv(os.path.join(plot_folder, fname), index=False)

    comm.Barrier()


def estimate_gamma_in_parallel(epi, a_samples, b_samples, rho_samples, df_recovery, start_idx, end_idx, N=2000):
    gammas = []
    offsets = []
    for index in range(start_idx, end_idx):
        a = a_samples[index]
        b = b_samples[index]
        rho = rho_samples[index]
        epiobj = DSA(df=epi.df, a=a, b=b, rho=rho)
        g, o = epiobj.estimate_gamma_sample(epiobj.theta, df_recovery=df_recovery, N=N, x0=(0.1, -5),
                                            bounds=[(1.0 / 25, 1.0 / 5), (-10, 0)], approach='offset')
        gammas.append(g)
        offsets.append(o)
    res = pd.DataFrame(
        {'gamma': list(gammas),
         'offset': list(offsets)}
    )
    comm.send(res, dest=0, tag=rank)


if __name__ == '__main__':
    main()
