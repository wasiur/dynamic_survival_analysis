from mpi4py import MPI
import pickle
import math
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def compute_epidemic_fit_in_parallel(epidemic_obj_arr, start_idx, end_idx):
    res = []
    for index in range(start_idx, end_idx):
        epidemic_obj = epidemic_obj_arr[index]
        epidemic_obj.fit()
        res.append((index, epidemic_obj))
    comm.send(res, dest=0, tag=rank)


if __name__ == '__main__':
    parent_rank = sys.argv[1]
    epidemic_objects_array = pickle.load(open("epidemic_objects_array_%s" % parent_rank, "rb"))

    ln = len(epidemic_objects_array)
    size = comm.Get_size()

    each_process = int(ln / float(size))
    start_id = rank * each_process
    end_id = (rank + 1) * each_process
    if rank == size - 1:
        end_id = ln

    comm.Barrier()

    compute_epidemic_fit_in_parallel(epidemic_objects_array, start_idx=start_id, end_idx=end_id)

    if rank == 0:
        epidemic_objects_array_fitted_dict = dict()
        epidemic_objects_array_fitted = []
        for other_rank in range(size):
            start_id_local = other_rank * each_process
            end_id_local = (other_rank + 1) * each_process
            if other_rank == size - 1:
                end_id_local = ln
            epidemic_obj_arr_tmp = comm.recv(source=other_rank, tag=other_rank)
            for index, obj in epidemic_obj_arr_tmp:
                epidemic_objects_array_fitted_dict[index] = obj

        for keys, vals in epidemic_objects_array_fitted_dict.items():
            epidemic_objects_array_fitted.append(vals)

        pickle.dump(epidemic_objects_array_fitted, open("epidemic_objects_array_fitted_%s" % parent_rank, "wb"),
                    protocol=3)

    comm.Barrier()
