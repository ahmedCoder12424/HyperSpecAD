import os, time, logging, math
from tqdm import tqdm

import numpy as np
np.random.seed(0)

import numba as nb
from numba import cuda
from numba.typed import List
from typing import Callable, Iterator, List, Optional, Tuple

import cupy as cp
import cuml, rmm
rmm.reinitialize(pool_allocator=False, managed_memory=True)

import pandas as pd
import scipy.sparse as ss
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
import fastcluster
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from config import Config
from joblib import Parallel, delayed

import time
import atexit

import os, sys




ANOMALY_EPS_PERCENTILE = 50

# Global dictionary of accumulators
_profiled_times = {}


def export_distance_metric(distance_metric, file_name):
    np.savetxt(file_name, distance_metric, delimiter=',')
   #logger.info("Exporting distance calculation to distance_metric.csv")
    print("exported distance metric "+ file_name)

def accumulate_time(func):
    """Decorator to accumulate total time spent in a function."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _profiled_times[func.__name__] = _profiled_times.get(func.__name__, 0.0) + elapsed
        return result
    return wrapper

# Print totals at program exit
@atexit.register
def report_times():
    print("\n--- Profiling summary ---")
    for name, total in _profiled_times.items():
        print(f"{name}: {total:.3f} seconds")


def gen_lvs(D: int, Q: int):
    base = np.ones(D)
    base[:D//2] = -1.0
    l0 = np.random.permutation(base)
    levels = list()
    for i in range(Q+1):
        flip = int(int(i/float(Q) * D) / 2)
        li = np.copy(l0)
        li[:flip] = l0[:flip] * -1
        levels.append(list(li))
    return cp.array(levels, dtype=cp.float32).ravel()


def gen_idhvs(D: int, totalFeatures: int, flip_factor: float):
    nFlip = int(D//flip_factor)

    mu = 0
    sigma = 1
    bases = np.random.normal(mu, sigma, D)

    import copy
    generated_hvs = [copy.copy(bases)]

    for _ in range(totalFeatures-1):        
        idx_to_flip = np.random.randint(0, D, size=nFlip)
        bases[idx_to_flip] *= (-1)
        generated_hvs.append(copy.copy(bases))

    return cp.array(generated_hvs, dtype=cp.float32).ravel()


def gen_lv_id_hvs(
    D: int,
    Q: int,
    bin_len: int,
    id_flip_factor: float,
    logger: logging
):
    lv_id_hvs_file = 'lv_id_hvs_D_{}_Q_{}_bin_{}_flip_{}.npz'.format(D, Q, bin_len, id_flip_factor)
    if os.path.exists(lv_id_hvs_file):
        logger.info("Load existing {} file for HD".format(lv_id_hvs_file))
        data = cp.load(lv_id_hvs_file)
        lv_hvs, id_hvs = data['lv_hvs'], data['id_hvs']
    else:
        lv_hvs = gen_lvs(D, Q)
        lv_hvs = cuda_bit_packing(lv_hvs, Q+1, D)
        id_hvs = gen_idhvs(D, bin_len, id_flip_factor)
        id_hvs = cuda_bit_packing(id_hvs, bin_len, D)
        cp.savez(lv_id_hvs_file, lv_hvs=lv_hvs, id_hvs=id_hvs)
    return lv_hvs, id_hvs


def cuda_bit_packing(orig_vecs, N, D):
    pack_len = (D+32-1)//32
    packed_vecs = cp.zeros(N * pack_len, dtype=cp.uint32)
    packing_cuda_kernel = cp.RawKernel(r'''
                    extern "C" __global__
                    void packing(unsigned int* output, float* arr, int origLength, int packLength, int numVec) {
                        int i = blockDim.x * blockIdx.x + threadIdx.x;
                        if (i >= origLength)
                            return;
                        for (int sample_idx = blockIdx.y; sample_idx < numVec; sample_idx += blockDim.y * gridDim.y) 
                        {
                            int tid = threadIdx.x;
                            int lane = tid % warpSize;
                            int bitPattern=0;
                            if (i < origLength)
                                bitPattern = __brev(__ballot_sync(0xFFFFFFFF, arr[sample_idx*origLength+i] > 0));
                            if (lane == 0) {
                                output[sample_idx*packLength+ (i / warpSize)] = bitPattern;
                            }
                        }
                    }
                    ''', 'packing')
    threads = 1024
    packing_cuda_kernel(((D + threads - 1) // threads, N), (threads,), (packed_vecs, orig_vecs, D, pack_len, N))

    return packed_vecs.reshape(N, pack_len)


def hd_encode_spectra_packed(spectra_intensity, spectra_mz, id_hvs_packed, lv_hvs_packed, N, D, Q, output_type):
    packed_dim = (D + 32 - 1) // 32
    encoded_spectra = cp.zeros(N * packed_dim, dtype=cp.uint32)
    
    max_peaks_used = spectra_intensity.shape[1]
    spectra_intensity = cp.array(spectra_intensity, dtype=cp.float32).ravel()
    spectra_mz = cp.array(spectra_mz, dtype=cp.int32).ravel()
    
    hd_enc_lvid_packed_cuda_kernel = cp.RawKernel(r'''
                __device__ float* get2df(float* p, const int x, int y, const int stride) {
                    return (float*)((char*)p + x*stride) + y;
                }
                __device__ char get2d_bin(unsigned int* p, const int i, const int DIM, const int d) {
                    unsigned int v = ((*(p + i * ((DIM + 32-1)/32) + d/32)) >> ((32-1) - d % 32)) & 0x01;
                    if (v == 0) {
                        return -1;
                    } else {
                        return 1;
                    }
                }
                extern "C" __global__
                void hd_enc_lvid_packed_cuda(
                    unsigned int* __restrict__ id_hvs_packed, unsigned int* __restrict__ level_hvs_packed, 
                    int* __restrict__ feature_indices, float* __restrict__ feature_values, 
                    int max_peaks_used, unsigned int* hv_matrix, 
                    int N, int Q, int D, int packLength) {
                    const int d = threadIdx.x + blockIdx.x * blockDim.x;
                    if (d >= D)
                        return;
                    for (int sample_idx = blockIdx.y; sample_idx < N; sample_idx += blockDim.y * gridDim.y) 
                    {
                        // we traverse [start, end-1]
                        float encoded_hv_e = 0.0;
                        unsigned int start_range = sample_idx*max_peaks_used;
                        unsigned int end_range = (sample_idx + 1)*max_peaks_used;
                        #pragma unroll 1
                        for (int f = start_range; f < end_range; ++f) {
                            if(feature_values[f] != -1)
                                encoded_hv_e += get2d_bin(level_hvs_packed, (int)(feature_values[f] * Q), D, d) * \
                                                get2d_bin(id_hvs_packed, feature_indices[f], D, d);
                        }
                        
                        // hv_matrix[sample_idx*D+d] = (encoded_hv_e > 0)? 1 : -1;
                        int tid = threadIdx.x;
                        int lane = tid % warpSize;
                        int bitPattern=0;
                        if (d < D)
                            bitPattern = __ballot_sync(0xFFFFFFFF, encoded_hv_e > 0);
                        if (lane == 0) {
                            hv_matrix[sample_idx * packLength + (d / warpSize)] = bitPattern;
                        }
                    }
                }
                ''', 'hd_enc_lvid_packed_cuda')
                
    threads = 1024
    max_block = cp.cuda.runtime.getDeviceProperties(0)['maxGridSize'][1]
    hd_enc_lvid_packed_cuda_kernel(
        ((D + threads - 1) // threads, min(N, max_block)), (threads,), 
        (id_hvs_packed, lv_hvs_packed, spectra_mz, spectra_intensity, max_peaks_used, encoded_spectra, N, Q, D, packed_dim))

    if output_type=='numpy':
        return encoded_spectra.reshape(N, packed_dim).get()
    elif output_type=='cupy':
        return encoded_spectra.reshape(N, packed_dim)


@cuda.jit('float32(uint32, uint32)', device=True, inline=True)
def fast_hamming_op(a, b):
    return nb.float32(cuda.libdevice.popc(a^b))

TPB = 32
TPB1 = 33

@cuda.jit('void(uint32[:,:], float32[:,:], float32[:], float32, int32, int32)')
def fast_pw_dist_cosine_mask_packed(A, D, prec_mz, prec_tol, N, pack_len):
    """
        Pair-wise cosine distance
    """
    sA = cuda.shared.array((TPB, TPB1), dtype=nb.uint32)
    sB = cuda.shared.array((TPB, TPB1), dtype=nb.uint32)

    x, y = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    bx = cuda.blockIdx.x

    tmp = nb.float32(.0)
    for i in range((pack_len+TPB-1) // TPB):
        if y < N and (i*TPB+tx) < pack_len:
            sA[ty, tx] = A[y, i*TPB+tx]
        else:
            sA[ty, tx] = .0

        if (TPB*bx+ty) < N and (i*TPB+tx) < pack_len:
            sB[ty, tx] = A[TPB*bx+ty, i*TPB+tx]
        else:
            sB[ty, tx] = .0  
        cuda.syncthreads()

        for j in range(TPB):
            tmp += fast_hamming_op(sA[ty, j], sB[tx, j])

        cuda.syncthreads()

    if x<N and y<N and y>x:
        if cuda.libdevice.fabsf((prec_mz[x]-prec_mz[y])/prec_mz[y])>=prec_tol:
            D[x,y] = 1.0
            D[y,x] = 1.0
        else:
            tmp/=(32*pack_len)
            D[x,y] = tmp
            D[y,x] = tmp

@accumulate_time
def fast_nb_cosine_dist_mask(hvs, prec_mz, prec_tol, output_type, stream=None):
    N, pack_len = hvs.shape

    # start = time.time()
    # ss = cp.cuda.Stream(non_blocking=True)
    # with stream:
    hvs_d = cp.array(hvs)
    prec_mz_d = cp.array(prec_mz.ravel())
    prec_tol_d = nb.float32(prec_tol/1e6)
    dist_d = cp.zeros((N,N), dtype=cp.float32)
    # print("Data loading time: ", time.time()-start)

    TPB = 32
    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # start = time.time()
    fast_pw_dist_cosine_mask_packed[blockspergrid, threadsperblock]\
        (hvs_d, dist_d, prec_mz_d, prec_tol_d, N, pack_len)
    cuda.synchronize()
    # print("CUDA computing time: ", time.time()-start)

    # start = time.time()
    if output_type=='cupy':
        dist = dist_d
    else:
        dist = dist_d.get()
    # print("Data fetching time: ", time.time()-start)

    return dist


# Condense pw_dist computation function with improved performance
@cuda.jit('void(uint32[:,:], float32[:], float32[:], float32, int32, int32)')
def fast_pw_dist_cosine_mask_packed_condense(A, D, prec_mz, prec_tol, N, pack_len):
    """
        Pair-wise cosine distance
    """
    sA = cuda.shared.array((TPB, TPB1), dtype=nb.uint32)
    sB = cuda.shared.array((TPB, TPB1), dtype=nb.uint32)

    x, y = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    bx = cuda.blockIdx.x

    tmp = nb.float32(.0)
    for i in range((pack_len+TPB-1) // TPB):
        if y < N and (i*TPB+tx) < pack_len:
            sA[ty, tx] = A[y, i*TPB+tx]
        else:
            sA[ty, tx] = .0

        if (TPB*bx+ty) < N and (i*TPB+tx) < pack_len:
            sB[ty, tx] = A[TPB*bx+ty, i*TPB+tx]
        else:
            sB[ty, tx] = .0  
        cuda.syncthreads()

        for j in range(TPB):
            tmp += fast_hamming_op(sA[ty, j], sB[tx, j])

        cuda.syncthreads()

    if x<N and y<N and y>x:
        if cuda.libdevice.fabsf((prec_mz[x]-prec_mz[y])/prec_mz[y])>=prec_tol:
            D[int(N*x-(x*x+x)/2+y-x-1)] = 1.0
        else:
            tmp/=(32*pack_len)
            D[int(N*x-(x*x+x)/2+y-x-1)] = tmp
           

def fast_nb_cosine_dist_condense(hvs, prec_mz, prec_tol, output_type, stream=None):
    N, pack_len = hvs.shape
    
    hvs_d = cp.array(hvs)
    prec_mz_d = cp.array(prec_mz.ravel())
    prec_tol_d = nb.float32(prec_tol/1e6)
    dist_d = cp.zeros(int(N*(N-1)/2), dtype=cp.float32)

    TPB = 32
    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    fast_pw_dist_cosine_mask_packed_condense[blockspergrid, threadsperblock]\
        (hvs_d, dist_d, prec_mz_d, prec_tol_d, N, pack_len)
    cuda.synchronize()

    if output_type=='cupy':
        dist = dist_d
    else:
        dist = dist_d.get()

    return dist


def get_dim(min_mz: float, max_mz: float, bin_size: float) \
        -> Tuple[int, float, float]:
    """
    Compute the number of bins over the given mass range for the given bin
    size.

    Parameters
    ----------
    min_mz : float
        The minimum mass in the mass range (inclusive).
    max_mz : float
        The maximum mass in the mass range (inclusive).
    bin_size : float
        The bin size (in Da).

    Returns
    -------
        A tuple containing (i) the number of bins over the given mass range for
        the given bin size, (ii) the highest multiple of bin size lower than
        the minimum mass, (iii) the lowest multiple of the bin size greater
        than the maximum mass. These two final values are the true boundaries
        of the mass range (inclusive min, exclusive max).
    """
    start_dim = min_mz - min_mz % bin_size
    end_dim = max_mz + bin_size - max_mz % bin_size
    # print(start_dim, end_dim, min_mz, max_mz, bin_size, math.ceil((end_dim - start_dim) / bin_size))
    return math.ceil((end_dim - start_dim) / bin_size), start_dim, end_dim


# @nb.jit(cache=True)
def _to_csr_vector(
    spectra: pd.DataFrame, 
    min_mz: float, 
    bin_size: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    mz = spectra['mz'].to_numpy()
    intensity = spectra['intensity'].to_numpy()

    mz = np.floor((np.vstack(mz)-min_mz)/bin_size)
    intensity = np.vstack(intensity)

    return intensity, mz 


def encode_cluster_spectra(
    spectra_by_charge_df: pd.DataFrame,
    config: Config,
    logger: logging,
    bin_len: int,
    lv_hvs: cp.array,
    id_hvs: cp.array
):
    # Encode spectra
    logger.info("Start encoding")
    encoded_spectra_hv = encode_preprocessed_spectra(
            spectra_df=spectra_by_charge_df, 
            config=config, dim=bin_len, logger=logger,
            lv_hvs_packed=lv_hvs, id_hvs_packed=id_hvs,
            output_type='numpy')

    # Cluster encoded spectra
    logger.info("Start clustering")    
    cluster_labels, representative_masks = cluster_encoded_spectra(
        spectra_by_charge_df=spectra_by_charge_df,
        encoded_spectra_hv=encoded_spectra_hv,
        config=config, logger=logger)

    return cluster_labels, representative_masks 



# TODO
def encode_cluster_spectra_bucket(
    spectra_df: pd.DataFrame, 
    config: Config,
    dim: int,
    lv_hvs_packed: cp.array,
    id_hvs_packed: cp.array,
    logger: logging,
    batch_size: int = 5000,
    output_type: str='numpy'
)-> List:
    start = time.time()

    num_batch = len(spectra_df)//batch_size+1

    lv_hvs = cp.asnumpy(lv_hvs_packed).ravel()
    id_hvs = cp.asnumpy(id_hvs_packed).ravel()

    print('time 1: ', time.time()-start)
    
    intensity, mz = _to_csr_vector(
        spectra_df, config.min_mz, config.fragment_tol)
    
    print('time 2: ', time.time()-start)

    spectra_df.drop(columns=['mz', 'intensity'], inplace=True)

    print('time 3: ', time.time()-start)

    data_dict = {
        'lv_hvs': lv_hvs, 'id_hvs': id_hvs, 
        'intensity': intensity, 'mz': mz}

    encoded_spectra = [encode_func(
        [i*batch_size, min((i+1)*batch_size, len(spectra_df))], 
        data_dict, config.hd_dim, config.hd_Q, dim, output_type) for i in tqdm(range(num_batch)) ] 
                    
    encoded_spectra = np.concatenate(encoded_spectra, dtype=np.uint32)\
        if output_type=='numpy' else encoded_spectra

    logger.info("Encode {} spectra in {:.4f}s".format(len(encoded_spectra), time.time()-start))

    return encoded_spectra



def encode_func(
    slice_idx: tuple,
    data_dict: dict,
    D: int,
    Q: int,
    dim: int,
    output_type: str
) -> np.ndarray:
    intensity, mz = data_dict['intensity'][slice_idx[0]: slice_idx[1]], data_dict['mz'][slice_idx[0]: slice_idx[1]]

    lv_hvs, id_hvs = cp.array(data_dict['lv_hvs']), cp.array(data_dict['id_hvs'])

    batch_size = slice_idx[1] - slice_idx[0]
    
    return hd_encode_spectra_packed(intensity, mz, id_hvs, lv_hvs, batch_size, D, Q, output_type)


def encode_preprocessed_spectra(
    spectra_df: pd.DataFrame, 
    config: Config,
    dim: int,
    lv_hvs_packed: cp.array,
    id_hvs_packed: cp.array,
    logger: logging,
    batch_size: int = 5000,
    output_type: str='numpy'
)-> List:
    start = time.time()

    num_spectra = len(spectra_df)
    num_batch = num_spectra//batch_size+1

    lv_hvs = cp.asnumpy(lv_hvs_packed).ravel()
    id_hvs = cp.asnumpy(id_hvs_packed).ravel()

    print('time 1: ', time.time()-start)
    
    intensity, mz = _to_csr_vector(
        spectra_df, config.min_mz, config.fragment_tol)

    print('time 2: ', time.time()-start)

    spectra_df.drop(columns=['mz', 'intensity'], inplace=True)

    print('time 3: ', time.time()-start)

    data_dict = {
        'lv_hvs': lv_hvs, 'id_hvs': id_hvs, 
        'intensity': intensity, 'mz': mz}

    encoded_spectra = [ encode_func(
        [i*batch_size, min((i+1)*batch_size, num_spectra)], 
        data_dict, config.hd_dim, config.hd_Q, dim, output_type) for i in tqdm(range(num_batch)) ] 
                    
    encoded_spectra = np.concatenate(encoded_spectra, dtype=np.uint32)\
        if output_type=='numpy' else encoded_spectra

    logger.info("Encode {} spectra in {:.4f}s".format(len(encoded_spectra), time.time()-start))

    return encoded_spectra


def encode_spectra(
    spectra_mz: np.ndarray, 
    spectra_intensity: np.ndarray, 
    config: Config,
    logger: logging,
    batch_size: int = 5000,
    output_type: str='numpy'
)-> np.ndarray:
    start = time.time()

    # Generate LV-ID hypervectors
    bin_len, min_mz, max_mz = get_dim(config.min_mz, config.max_mz, config.fragment_tol)
    
    lv_hvs, id_hvs = gen_lv_id_hvs(config.hd_dim, config.hd_Q, bin_len, config.hd_id_flip_factor, logger)
    
    data_dict = {
        'lv_hvs': cp.asnumpy(lv_hvs).ravel(), 
        'id_hvs': cp.asnumpy(id_hvs).ravel(), 
        'intensity': spectra_intensity, 'mz': spectra_mz}

    num_spectra = spectra_mz.shape[0]
    num_batch = num_spectra//batch_size+1

    # Encode spectra on GPU
    encoded_spectra = [ encode_func(
        [i*batch_size, min((i+1)*batch_size, num_spectra)], 
        data_dict, config.hd_dim, config.hd_Q, bin_len, output_type) for i in tqdm(range(num_batch)) ] 
                    
    encoded_spectra = np.concatenate(encoded_spectra, dtype=np.uint32)\
        if output_type=='numpy' else encoded_spectra

    logger.info("Encode {} spectra in {:.4f}s".format(len(encoded_spectra), time.time()-start))

    return encoded_spectra


def _get_bucket_idx_list(
    spectra_by_charge_df: pd.DataFrame,
    logger: logging
):
    # Get bucket list
    buckets = spectra_by_charge_df.bucket.unique()
    num_bucket = len(buckets)

    bucket_idx_arr = np.zeros((num_bucket ,2), dtype=np.int32)
    bucket_size_arr = np.zeros(num_bucket, dtype=np.int32)
    for i, b_i in enumerate(buckets):
        bucket_idx_i = (spectra_by_charge_df.bucket==b_i).to_numpy()
        bucket_idx_i = np.argwhere(bucket_idx_i==True).flatten()
        bucket_idx_arr[i, :] = [bucket_idx_i[0], bucket_idx_i[-1]]
        bucket_size_arr[i] = bucket_idx_i[-1]-bucket_idx_i[0]+1
    
    hist, bins = np.histogram(bucket_size_arr, bins=[0, 300, 1000, 5000, 10000, 20000, 30000], density=False)

    logger.info("There are {} buckets. Maximum bucket size = {}".format(num_bucket, max(bucket_size_arr)))
    logger.info("Bucket size distribution:")
    for i in range(len(bins)-1):
        logger.info("{:.2f}% of bucket size between {} and {}".format(hist[i]/num_bucket*100, bins[i], bins[i+1]))

    return bucket_idx_arr, bucket_size_arr


def schedule_bucket(
    spectra_by_charge_df: pd.DataFrame,
    logger: logging
):
    bucket_idx_arr, bucket_size_arr = _get_bucket_idx_list(spectra_by_charge_df, logger)

    # Sort the buckets based on their sizes
    sort_idx = np.argsort(-bucket_size_arr)
    sorted_bucket_idx_arr = bucket_idx_arr[sort_idx]

    reorder_idx = np.argsort(sort_idx)

    return {
        'sort_bucket_idx_arr': sorted_bucket_idx_arr, 
        'reorder_idx': reorder_idx}



def export_distance_metric(distance_metric, file_name):

    np.savetxt(file_name, distance_metric, delimiter=',')

import csv
def load_csv_as_2d_list(filename):
    matrix = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # convert non-empty cells to floats
            values = [float(x) for x in row if x.strip() != ""]
            if values:
                matrix.append(values)
    return matrix


def cluster_bucket (
    bucket_slice: tuple, 
    data_dict: dict, 
    config: Config,
    cluster_func: Callable,
    output_type: str='numpy'
):


    if bucket_slice[1]-bucket_slice[0]==0:
        return [np.array([-1]), np.array([True])]
    else:
        bucket_slice[1] += 1
        bucket_hv = data_dict['hv'][bucket_slice[0]: bucket_slice[1]]
        bucket_prec_mz = data_dict['prec_mz'][bucket_slice[0]: bucket_slice[1]]
        bucket_rt_time = data_dict['rt_time'][bucket_slice[0]: bucket_slice[1]]
        
        pw_dist = fast_nb_cosine_dist_mask(bucket_hv, bucket_prec_mz, config.precursor_tol[0], output_type)
        export_distance_metric(pw_dist, "1302_distance_metrics.csv")

        cluster_func.fit(pw_dist) #
        cluster_func_labels = cluster_func.labels_

        cluster_labels_refined = refine_cluster(
            bucket_cluster_label = cluster_func_labels, 
            bucket_precursor_mzs = bucket_prec_mz,
            bucket_rts = bucket_rt_time,
            precursor_tol_mass = config.precursor_tol[0], 
            precursor_tol_mode = config.precursor_tol[1], 
            rt_tol = config.rt_tol)
       
       # print("REFINED CLUSTER LABELS", cluster_labels_refined)
        #sys.exit(0)
        representative_mask = get_cluster_representative(
            cluster_labels=cluster_labels_refined, pw_dist=pw_dist)



        return [cluster_labels_refined, representative_mask]

def cluster_bucket_incr(
    bucket_slice: tuple, 
    data_dict: dict, 
    config: Config,
    cluster_func: Callable,
    output_type: str = 'numpy'
):
    initial_percentage = 0.7
    incremental_eps = 0.31

    if bucket_slice[1] - bucket_slice[0] == 0:
        return [np.array([-1]), np.array([True])]

    bucket_slice = (bucket_slice[0], bucket_slice[1] + 1)
    bucket_hv = data_dict['hv'][bucket_slice[0]: bucket_slice[1]]
    bucket_prec_mz = data_dict['prec_mz'][bucket_slice[0]: bucket_slice[1]]
    bucket_rt_time = data_dict['rt_time'][bucket_slice[0]: bucket_slice[1]]

    total_len = bucket_hv.shape[0]

    # ===  Step 0: Skip incremental if too few points ===
    if total_len < 10:
        pw_dist = fast_nb_cosine_dist_mask(bucket_hv, bucket_prec_mz, config.precursor_tol[0], output_type)
        cluster_func.fit(pw_dist)
        initial_labels = cluster_func.labels_

        cluster_labels_refined = refine_cluster(
            bucket_cluster_label=initial_labels, 
            bucket_precursor_mzs=bucket_prec_mz,
            bucket_rts=bucket_rt_time,
            precursor_tol_mass=config.precursor_tol[0], 
            precursor_tol_mode=config.precursor_tol[1], 
            rt_tol=config.rt_tol
        )

        representative_mask = get_cluster_representative(
            cluster_labels=cluster_labels_refined, pw_dist=pw_dist
        )

        return [cluster_labels_refined, representative_mask]

    # === Step 1: Compute full pairwise distance matrix ===
    pw_dist = fast_nb_cosine_dist_mask(bucket_hv, bucket_prec_mz, config.precursor_tol[0], output_type)

    split_index = int(initial_percentage * total_len)
    pw_dist_initial = pw_dist[:split_index, :split_index]
    cluster_func.fit(pw_dist_initial)
    initial_labels = cluster_func.labels_

    # === Step 2: Refine clusters from initial set ===
    cluster_labels_refined = refine_cluster(
        bucket_cluster_label=initial_labels, 
        bucket_precursor_mzs=bucket_prec_mz[:split_index],
        bucket_rts=bucket_rt_time[:split_index],
        precursor_tol_mass=config.precursor_tol[0], 
        precursor_tol_mode=config.precursor_tol[1], 
        rt_tol=config.rt_tol
    )

    # === Step 3: Prepare cluster metadata ===
    unique_clusters = np.unique(cluster_labels_refined)
    next_cluster_id = max(unique_clusters[unique_clusters != -1], default=-1) + 1

    final_labels = np.full(total_len, -1, dtype=int)
    final_labels[:split_index] = cluster_labels_refined

    cluster_representative_indices = {
        cid: np.random.choice(np.where(cluster_labels_refined == cid)[0])
        for cid in unique_clusters if cid != -1
    }

    rep_ids = np.array(list(cluster_representative_indices.keys()))
    rep_indices = np.array(list(cluster_representative_indices.values()), dtype=int)


    if len(rep_indices) == 0:
        # Treat all points as noise
        final_labels = np.full(total_len, -1, dtype=int)
        representative_mask = np.ones(total_len, dtype=bool)  # all singleton
  # === Step 4: Batch incremental clustering ===
    dist_matrix = pw_dist[split_index:, rep_indices]
    best_idx = np.argmin(dist_matrix, axis=1)
    best_dists = dist_matrix[np.arange(dist_matrix.shape[0]), best_idx]

    current_next_cluster_id = next_cluster_id
    for j, (best_cluster_idx, dist) in enumerate(zip(best_idx, best_dists)):
        global_i = split_index + j
        if dist <= incremental_eps:
            final_labels[global_i] = rep_ids[best_cluster_idx]
        else:
            final_labels[global_i] = current_next_cluster_id
            rep_ids = np.append(rep_ids, current_next_cluster_id)
            rep_indices = np.append(rep_indices, global_i)
            current_next_cluster_id += 1

    # === Step 5: Compute final representative mask ===
    representative_mask = get_cluster_representative(
        cluster_labels=final_labels, pw_dist=pw_dist
    )

    return [final_labels, representative_mask]


#jit
def get_cluster_indices(cluster, clusters):
        #scluster_subset = cluster_results.loc[cluster_results['cluster']==cluster]
      # print("cluster subset index", cluster_subset.index)
       # hvs_subset =  spectra_hvs[cluster_subset.index] 
        return np.where(clusters == cluster)[0]


def bundle(input_hv):
# input_hv = cp.asarray(input_hv.view(np.uint8))

    N, word_len = input_hv.shape

    unpacked_bundle = cp.zeros((word_len,), dtype=cp.uint32)
    threadsperblock = (32,32)    # 32 bits �~W 8 threads scanning rows
    blockspergrid = (word_len)

    fast_majority_nounpack[blockspergrid, threadsperblock](input_hv, unpacked_bundle)

    bundled_hv = unpacked_bundle #cp.packbits(unpacked_bundle).view(cp.uint32)
  # print("UNPACKED BUNDLE", bundled_hv)
  # print("Comparison", input_hv[1][0], bundled_hv[0])
    return bundled_hv
@cuda.jit("void(uint32[:,:], uint32[:])")
def fast_majority_nounpack(A, D):

    smem = cuda.shared.array((32,32), dtype=nb.int32)
    bit_index  = cuda.threadIdx.x
    word_index = cuda.blockIdx.x;
    block_dim = cuda.blockDim.y
    # tx = cuda.threadIdx.x
    # bx = cuda.blockIdx.x
    N, total_columns = A.shape

    mask = 1 << bit_index
    s = 0
    if (word_index >= total_columns or  bit_index >= 32):
        return


    for row in range(cuda.threadIdx.y, N, cuda.blockDim.y):
        val = A[row, word_index]
        s += (val & mask) >> bit_index

    x = cuda.threadIdx.y;
    smem[bit_index,x] = s
    cuda.syncthreads()

    stride = block_dim // 2
    while stride > 0:
        if x < stride:
            smem[bit_index,x] += smem[bit_index, x + stride]
        cuda.syncthreads()
        stride //= 2

   # if (x == 0):
    #    col = word_index * 32 + bit_index
     #   count = smem[0]
      #  D[col] = 1 if count > (N // 2) else 0
    if x == 0:  # one thread writes
        count = smem[bit_index, 0]
        if count > (N // 2):
          # D[word_index] = D[word_index] | mask
            cuda.atomic.or_(D, word_index, mask)
def get_all_cluster_reps(clusters, hvs, prev_prec_mz, bucket=None):

    print("BEGINNING FINDING CLUSTER REP")
   # clusters = cluster_results.drop_duplicates(subset='cluster')["cluster"].to_numpy()
   # if (bucket is not None):
    #    clusters = cluster_results[cluster_results['bucket'] == bucket].drop_duplicates(subset='cluster')["cluster"].to_numpy()
    clusters = np.array(clusters)
    results = []
    spectra_hvs_gpu = cp.asarray(hvs)
   #spectra_hvs_gpu = cp.asarray(hvs.view(np.uint8))
    print("BUNDLING")


    spectra_vals = []
    rep_hv_list = []
    for cluster in tqdm(clusters, desc="Bundling clusters"):
        indices = get_cluster_indices(clusters, cluster) #p.where(clusters == cluster)[0]
        hvs_subset = spectra_hvs_gpu[indices]
        rep_hv = bundle(hvs_subset)  # returns cupy.ndarray
        results.append({"cluster": cluster, "rep_hv": rep_hv})
        spectra_vals.append(np.random.choice(prev_prec_mz[indices]))
        rep_hv = cp.asnumpy(rep_hv) 
        rep_hv_list.append(rep_hv)

    # Convert rep_hv to numpy only when building DataFrame
    for r in results:
        r["rep_hv"] = cp.asnumpy(r["rep_hv"])


    representatives = pd.DataFrame(results)
    rep_hv_array = np.vstack(rep_hv_list)
    spectra_vals_array = np.array(spectra_vals).reshape(-1, 1)
    
    return rep_hv_array, clusters, spectra_vals_array


def get_bucket_slice(prev_meta_df, bucket):

    mask = prev_meta_df['bucket'] == bucket
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return (None, None)

    return  (indices[0], indices[-1] + 1) 

def get_bucket_indices(meta_df, bucket):
    mask = meta_df['bucket'].to_numpy() == bucket
    indices = np.where(mask)[0]
    return indices

def cluster_bucket_incr_3(
    bucket_slice: tuple,
    data_dict,
    prev_hvs,
    prev_clusters,
    config,
    prev_prec_mz,
    output_type,
    prev_meta_df, 
    bucket,
    cluster_func
):

    incremental_eps = 0.35
    # print("RUNNIGN cluster_bucket_incr_2 ")
    if bucket_slice[1] - bucket_slice[0] == 0:
        # print("bucket slice 0")
        return [np.array([-1]), np.array([True]),[],  np.array([True])]

    bucket_slice = (bucket_slice[0], bucket_slice[1])
    bucket_hv = data_dict['hv'][bucket_slice[0]: bucket_slice[1]+1]
    bucket_prec_mz = data_dict['prec_mz'][bucket_slice[0]: bucket_slice[1]+1]
    bucket_rt_time = data_dict['rt_time'][bucket_slice[0]: bucket_slice[1]+1]

    
    pbucket_slice = get_bucket_slice(prev_meta_df, bucket)
    pbucket_idx = get_bucket_indices(prev_meta_df, bucket)

    # bucket_prev_hv = prev_hvs[pbucket_slice[0]: pbucket_slice[1]]

    bucket_prev_hv = prev_hvs[pbucket_idx]




    clusters_p, count_p = np.unique(prev_clusters['cluster'].to_numpy(), return_counts=True)

    bucket_clusters = prev_clusters[prev_clusters['bucket']==bucket].reset_index(drop=True)
    # print(len(bucket_clusters), len(bucket_prev_hv))


    clusters_p, count_p = np.unique(bucket_clusters['cluster'].to_numpy(), return_counts=True)




    prev_rep_mask = prev_clusters['is_representative'].to_numpy()
    bucket_prev_prec_mz = prev_prec_mz[pbucket_slice[0]: pbucket_slice[1]]

    cluster_reps = bucket_clusters.loc[bucket_clusters['is_representative'], 'cluster'].to_numpy()
    cluster_rep_indices = bucket_clusters.index[bucket_clusters['is_representative']].to_numpy()
    cluster_rep_hvs = bucket_prev_hv[cluster_rep_indices]
    clusters = bucket_clusters['cluster'].to_numpy()
     
 
    if (pbucket_slice[0]==None or pbucket_slice[1]==None or len(cluster_reps) == 0):
        # print("returning early1")
        output = (cluster_bucket(
                bucket_slice = np.array(bucket_slice),
                data_dict = data_dict,
                config = config,
                cluster_func = cluster_func,
                output_type = 'cupy' if config.use_gpu_cluster else 'numpy'))
        output.append([])
        bucket_len = bucket_hv.shape[0] 
        output.append(np.ones(bucket_len, dtype=bool))
        return output

    cluster_rep_mz = np.vstack(bucket_prev_prec_mz[cluster_rep_indices])
    cluster_rep_indices = np.where(bucket_clusters['is_representative'])[0]  # relative indices
    
    clusters = bucket_clusters['cluster'].to_numpy()
    rep_ids = clusters[cluster_rep_indices]

    similarity_metrics = []
    beta = 0.5
    # print("cluster frequencies")
    # cluster_counts = bucket_clusters['cluster'].value_counts()
    # print(cluster_counts)   
    # for rep_id in rep_ids:
    #     ids = np.where(bucket_clusters['cluster'].values == rep_id)[0]
    #     # if (len(ids)>1):
    #     #     # print("ids of cluster", ids)
    #     # print(rep_id)
    #     # print("ids", ids)
    #     cluster_hvs = bucket_prev_hv[ids]
    #     # print(len(cluster_hvs))
    #     cluster_prec_mz = bucket_prev_prec_mz[ids]
    #     # print(len(cluster_prec_mz))
    #     pw_dist = fast_nb_cosine_dist_mask(cluster_hvs, cluster_prec_mz, config.precursor_tol[0], output_type)
    #     # print(pw_dist.shape)
    #     # print(pw_dist)
    #     mean = float(pw_dist.mean())
    #     std = float(pw_dist.std())
    #     # print("mean", mean, "std", std)
    #     score = mean - std*beta
    #     # if (score > 0.20):
    #     #     # print(f"score = {score:.20f}", score ==0)
    #     #    # sys.exit(0)
    #     if (score <=0.0):
    #         score = incremental_eps
    #     similarity_metrics.append(score)

    # print(similarity_metrics)


    
    full_hvs = np.concatenate((cluster_rep_hvs, bucket_hv), axis=0)

    bucket_prec_mz = np.concatenate([cluster_rep_mz, bucket_prec_mz])
    rep_indices = np.arange(len(cluster_rep_indices))
    split_index = len(cluster_rep_hvs)

    total_len =  len(bucket_hv)
    final_labels = np.full(total_len, -1, dtype=int)
    
    pw_dist = fast_nb_cosine_dist_mask(full_hvs, bucket_prec_mz, config.precursor_tol[0], output_type)
    if config.use_gpu_cluster or output_type == "cupy":
        pw_dist = cp.asnumpy(pw_dist)

    unique_clusters = np.unique(clusters)
    # print(unique_clusters)
    next_cluster_id = max(unique_clusters[unique_clusters != -1], default=-1) + 1

    if rep_indices.size  == 0:
        final_labels = np.full(total_len, -1, dtype=int)
        representative_mask = np.ones(total_len, dtype=bool)  
        anomaly_mask = np.zeros(len(bucket_hv), dtype=bool)
        prev_rep_mask = bucket_clusters['is_representative'].to_numpy()
        return [final_labels, representative_mask, prev_rep_mask, anomaly_mask]


    dist_matrix = pw_dist[len(cluster_reps):, rep_indices]  
    # print("pw_dis", pw_dist.shape, dist_matrix.shape)
    # sys.exit(0)
    best_idx = np.argmin(dist_matrix, axis=1)
    best_dists = dist_matrix[np.arange(dist_matrix.shape[0]), best_idx]
 
    current_next_cluster_id = next_cluster_id

    representative_mask = np.zeros(len(bucket_hv), dtype=bool)
    anomaly_mask = np.zeros(len(bucket_hv), dtype=bool)
    singleton_indices = []
    cluster_id_before_incr = current_next_cluster_id
    anomaly_eps = incremental_eps = 0.42


    
    for j, (best_cluster_idx, dist) in enumerate(zip(best_idx, best_dists)):
        global_i = best_cluster_idx
        anomaly_eps = incremental_eps #similarity_metrics[best_cluster_idx]
        if dist <= anomaly_eps:
            # print("dist", dist, "anomaly_eps", anomaly_eps)
            final_labels[j] = rep_ids[best_cluster_idx] 
            bucket_rep_relative_index = rep_indices[best_cluster_idx]
            # original_prev_index = pbucket_slice[0] + bucket_rep_relative_index
            if (prev_clusters.at[original_prev_index,'anomaly']==True):
                print("joining anomaly cluster")
            #     anomaly_mask[j] = True
            # print("not anomaly")
            # prev_rep_row = cluster_rep_indices[best_cluster_idx]
            # if bucket_clusters.iloc[prev_rep_row]['anomaly']:
            #     anomaly_mask[j] = True

        else:
            final_labels[j] = current_next_cluster_id
            representative_mask[j] = True
            anomaly_mask[j] = True 
            # print("marking as anomaly", dist, anomaly_eps)
            current_next_cluster_id += 1
            singleton_indices.append(len(cluster_rep_hvs)+j)

    clusters_p, count_p = np.unique(bucket_clusters['cluster'].to_numpy(), return_counts=True)
    
    unique_clusters_f, counts_f = np.unique(final_labels, return_counts=True)
   
    rep_mask = np.isin(unique_clusters_f, rep_ids)

    old_count_dict = dict(zip(clusters_p, count_p))
    counts_old_aligned = np.array([old_count_dict.get(c, 0) for c in unique_clusters_f])

    # Compute new counts
    counts_new = counts_f + counts_old_aligned
    # print("comparing rep and anomaly mask size",  len(representative_mask), len(anomaly_mask))
    return [final_labels, representative_mask, prev_rep_mask, anomaly_mask]

def weighted_percentile(values, weights, percentile):
    values = np.asarray(values)
    weights = np.asarray(weights)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cum_weights = np.cumsum(weights)
    cutoff = percentile / 100.0 * cum_weights[-1]

    return values[np.searchsorted(cum_weights, cutoff)]


def cluster_bucket_incr_2(
    bucket_slice: tuple,
    data_dict,
    prev_hvs,
    prev_clusters,
    config,
    prev_prec_mz,
    output_type,
    prev_meta_df, 
    bucket,
    cluster_func
):

    incremental_eps = 0.35
    # print("RUNNIGN cluster_bucket_incr_2 ")
    # print("BUCKET", bucket, bucket_slice[1] - bucket_slice[0])
    if bucket_slice[1] - bucket_slice[0] < 0:
        # print("bucket slice 0")
        # print("returning since bucket slice is zero", bucket)

        # print(data_dict['prec_mz'][bucket_slice[0]],prev_meta_df['precursor_mz'].max(), len(prev_meta_df))

        bucket_len = bucket_slice[1] - bucket_slice[0] + 1

        if (data_dict['prec_mz'][bucket_slice[0]] > prev_meta_df['precursor_mz'].max()):
            #return [np.array([-1]), np.array([True]),[],  np.array([True])]
            return [np.full(bucket_len, -1, dtype=int), np.ones(bucket_len, dtype=bool),[], np.ones(bucket_len, dtype=bool)]
        else:
            # return [np.array([-1]), np.array([True]),[],  np.array([False])]
            return [np.full(bucket_len, -1, dtype=int), np.ones(bucket_len, dtype=bool),[], np.zeros(bucket_len, dtype=bool)]


    bucket_slice = (bucket_slice[0], bucket_slice[1])
  
    bucket_hv = data_dict['hv'][bucket_slice[0]: bucket_slice[1]+1]
    bucket_prec_mz = data_dict['prec_mz'][bucket_slice[0]: bucket_slice[1]+1]
    bucket_rt_time = data_dict['rt_time'][bucket_slice[0]: bucket_slice[1]+1]
    metadata_df = data_dict['meta_data']
    metadata_df = metadata_df[metadata_df['bucket']==bucket].reset_index(drop=True)

    # n = len(metadata_df[(metadata_df['cluster']== 18723)]) + len(metadata_df[(metadata_df['cluster']==28038)]) + len(metadata_df[(metadata_df['cluster']== 22736)])

 
    # n = len(metadata_df[metadata_df['cluster']==511196])

    # print("N is ", n)

    # print("bucket:", bucket)
    # print("cluster 511196 anywhere in current batch:",
    #     (data_dict["meta_data"]["cluster"].astype(str) == "511196").sum())

    # print("cluster 511196 in this bucket:",
    #     ((data_dict["meta_data"]["cluster"].astype(str) == "511196") &
    #     (data_dict["meta_data"]["bucket"] == bucket)).sum())

    # if (n>0):
    #     print("THIS BUCKET HAS ANOMALIES")

    # print("entry to bucket func:",
    #   (data_dict["meta_data"]["cluster"] == 511196).sum())
    # if (data_dict["meta_data"]["cluster"] == 511196).sum() >=1:

    #     rows = data_dict["meta_data"][data_dict["meta_data"]["cluster"] == 511196]

        # print(rows[["bucket", "cluster", "identifier", "scan"]])

  

    # print(metadata_df.head())


    # print(bucket)
    # print(len(prev_meta_df[prev_meta_df['bucket']==bucket]))

    # pbucket_idx = get_bucket_indices(prev_meta_df, bucket)

    # pbucket_slice = get_bucket_slice(prev_meta_df, bucket)
    # bucket_prev_hv = prev_hvs[pbucket_slice[0]: pbucket_slice[1]+1]


        # REPLACE with:
    pbucket_idx = get_bucket_indices(prev_meta_df, bucket)
    bucket_prev_hv = prev_hvs[pbucket_idx]
    bucket_prev_prec_mz = prev_prec_mz[pbucket_idx]
    bucket_prev_meta_df = prev_meta_df.iloc[pbucket_idx]
    bucket_prev_meta_df = bucket_prev_meta_df.reset_index(drop=True)

    # Keep pbucket_slice ONLY for the None checks below:
    pbucket_slice = get_bucket_slice(prev_meta_df, bucket)


    


    # print("previous data")
    # print(prev_meta_df.iloc[pbucket_slice[0]: pbucket_slice[1]+1])

    
    # bucket_prev_hv = prev_hvs[pbucket_idx]




    clusters_p, count_p = np.unique(prev_clusters['cluster'].to_numpy(), return_counts=True)

  #  bucket_clusters = prev_clusters.iloc[pbucket_slice[0]:pbucket_slice[1]].reset_index(drop=True)
    bucket_clusters = prev_clusters[prev_clusters['bucket']==bucket].reset_index(drop=True)
    key_cols = ["bucket", "precursor_charge", "identifier", "scan", "retention_time"]

    bucket_clusters = (
                    bucket_prev_meta_df[key_cols]
                    .merge(bucket_clusters, on=key_cols, how="left")
                )
#     print('bucket clusters')
#     print(bucket_clusters['retention_time'])
#     print("bucket prev meta df")
#     print(bucket_prev_meta_df['retention_time'])
#     rt_compare = pd.concat(
#     [
#         bucket_clusters['retention_time'].reset_index(drop=True).rename('cluster_rt'),
#         bucket_prev_meta_df['retention_time'].reset_index(drop=True).rename('prev_rt')
#     ],
#     axis=1
# )

#     print(rt_compare)
#     assert bucket_clusters[["scan", "identifier"]].reset_index(drop=True).equals(
#     bucket_prev_meta_df[["scan", "identifier"]].reset_index(drop=True)
# ), "MISALIGN: bucket_clusters rows do not match bucket_prev_hv rows"


#Commented out to see more logs 
    # print("comparing clusters and hv length", len(bucket_prev_hv), len(bucket_clusters))


    # print(len(bucket_clusters), len(bucket_prev_hv))

    # print('previous cluster datafrane')
    # print(prev_clusters.iloc[pbucket_slice[0]:pbucket_slice[1]])

    # print("bucket clusters")
    # print(bucket_clusters)

    # print(len(bucket_prev_hv), len(bucket_prev_hv))



    # clusters_p, count_p = np.unique(bucket_clusters['cluster'].to_numpy(), return_counts=True)


 
    prev_rep_mask = prev_clusters['is_representative'].to_numpy()
    # bucket_prev_prec_mz = prev_prec_mz[pbucket_slice[0]: pbucket_slice[1]]

    cluster_reps = bucket_clusters.loc[bucket_clusters['is_representative'], 'cluster'].to_numpy()



    cluster_rep_indices = bucket_clusters.index[bucket_clusters['is_representative']].to_numpy()
    clusters = bucket_clusters['cluster'].to_numpy()
    # print("all clusters", np.unique(clusters))
    # print("cluster_reps", cluster_reps)
    clusters = bucket_clusters['cluster'].to_numpy()

    #Fill in missing cluster reps using positional indices (already correct)
    for c in np.unique(clusters):
        if c not in clusters[cluster_rep_indices]:  # check by cluster ID, not index
            idx = np.random.choice(np.where(clusters == c)[0])  # positional ✓
            cluster_rep_indices = np.append(cluster_rep_indices, idx)

    # Now these are both positional and aligned:
    # print("cluster_rep_indices dtype:", cluster_rep_indices.dtype, cluster_rep_indices[:5])
    if (len(cluster_rep_indices) == 0):
        # print("returning since no cluster_rep_indices in bucket", bucket)

        # print(data_dict['prec_mz'][bucket_slice[0]],prev_meta_df['precursor_mz'].max(), len(prev_meta_df))
        bucket_len = bucket_slice[1] - bucket_slice[0] + 1
        if (data_dict['prec_mz'][bucket_slice[0]] > prev_meta_df['precursor_mz'].max()):
            # return [np.array([-1]), np.array([True]),[],  np.array([True])]
            return [np.full(bucket_len, -1, dtype=int), np.ones(bucket_len, dtype=bool),[], np.ones(bucket_len, dtype=bool)]
            
        else:
            # return [np.array([-1]), np.array([True]),[],  np.array([False])]
            return [np.full(bucket_len, -1, dtype=int), np.ones(bucket_len, dtype=bool),[], np.zeros(bucket_len, dtype=bool)]



    rep_ids = clusters[cluster_rep_indices]              
    # print("after adidng remaining clusters",len(cluster_rep_indices))
    



    # rep_rows = bucket_clusters.iloc[cluster_rep_indices]
    # print("rep_rows")
    # print(rep_rows)
    # print("metadata_df")
    # print(prev_meta_df)
    # prev_meta_df = prev_meta_df.reset_index().rename(columns={"index": "meta_idx"})
    # print("rep rows", len(rep_rows))
    # match_cols = ["scan", "identifier", "retention_time"]

    # rep_rows["identifier"] = rep_rows["identifier"].astype(str).str.strip()
    # prev_meta_df["identifier"] =  prev_meta_df["identifier"].astype(str).str.strip()

    # rep_rows["scan"] = rep_rows["scan"].astype(int)
    # prev_meta_df["scan"] =     prev_meta_df["scan"].astype(int)

    # rep_rows["retention_time"] = rep_rows["retention_time"].astype(float).round(4)
    # prev_meta_df["retention_time"] =     prev_meta_df["retention_time"].astype(float).round(4)
    
    # matched =     prev_meta_df.merge(
    # rep_rows[match_cols],
    # on=match_cols,
    # how="inner")

    # matched_indices = matched['meta_idx']

    # print("matched", matched_indices)
    # print("cluster rep indices", cluster_rep_indices)
    # sys.exit(0)

    #add clusters without reps 

    cluster_rep_hvs = bucket_prev_hv[cluster_rep_indices]
    # count spectra/items in each cluster
    cluster_sizes = pd.Series(clusters).value_counts()

    # frequency for each representative's cluster, aligned with cluster_rep_indices
    cluster_rep_freqs = cluster_sizes.loc[rep_ids].to_numpy()/len(bucket_prev_hv)


    

  
    # if ( (pbucket_slice[0]==None or pbucket_slice[1]==None) or  len(cluster_reps) == 0):
    #     print("returning early1", pbucket_slice[0], pbucket_slice[1],  len(cluster_reps))
    #     print("marking anomalies",  bucket_hv.shape[0])
    #     output = (cluster_bucket(
    #             bucket_slice = np.array(bucket_slice),
    #             data_dict = data_dict,
    #             config = config,
    #             cluster_func = cluster_func,
    #             output_type = 'cupy' if config.use_gpu_cluster else 'numpy'))
    #     output.append([])
    #     bucket_len = bucket_hv.shape[0] 
    #     output.append(np.ones(bucket_len, dtype=bool))
    #     return output


    if (pbucket_slice[0]==None and pbucket_slice[1]==None):
        # print("returning early1")
        # if (n > 0):
        #     pass
        #     # print("returning early1 real anomaly cluster", pbucket_slice[0], pbucket_slice[1],  len(cluster_reps), "bucket", bucket)
        #     # print("adding this many", bucket_hv.shape[0] )
        #     # print(metadata_df)
        # else:
        #     # print("returning early1 false anomaly cluster", pbucket_slice[0], pbucket_slice[1],  len(cluster_reps))
        #     pass
        output = (cluster_bucket(
                bucket_slice = np.array(bucket_slice),
                data_dict = data_dict,
                config = config,
                cluster_func = cluster_func,
                output_type = 'cupy' if config.use_gpu_cluster else 'numpy'))
        output.append([])
        bucket_len = bucket_hv.shape[0] 
        output.append(np.zeros(bucket_len, dtype=bool))
        return output


    if (pbucket_slice[0]==None or pbucket_slice[1]==None or len(cluster_reps) == 0):
        # print("returning early1, not marked as anomalies")
        # if (n > 0):
        #     pass
        #     # print("returning early1 real anomaly cluster", pbucket_slice[0], pbucket_slice[1],  len(cluster_reps), "bucket", bucket)
        #     # print("adding this many", bucket_hv.shape[0] )
        #     # print(metadata_df)
        # else:
        #     # print("returning early1 false anomaly cluster", pbucket_slice[0], pbucket_slice[1],  len(cluster_reps))
        #     pass
        output = (cluster_bucket(
                bucket_slice = np.array(bucket_slice),
                data_dict = data_dict,
                config = config,
                cluster_func = cluster_func,
                output_type = 'cupy' if config.use_gpu_cluster else 'numpy'))
        output.append([])
        bucket_len = bucket_hv.shape[0] 
        output.append(np.zeros(bucket_len, dtype=bool))
        return output
    # elif ((pbucket_slice[0]==None or pbucket_slice[1]==None) or len(cluster_reps) == 0):
    #     print("returning early1", pbucket_slice[0], pbucket_slice[1],  len(cluster_reps))
    #     output = (cluster_bucket(
    #             bucket_slice = np.array(bucket_slice),
    #             data_dict = data_dict,
    #             config = config,
    #             cluster_func = cluster_func,
    #             output_type = 'cupy' if config.use_gpu_cluster else 'numpy'))
    #     output.append([])
    #     bucket_len = bucket_hv.shape[0] 
    #     output.append(np.zeros(bucket_len, dtype=bool))
    #     return output



    cluster_rep_mz = np.vstack(bucket_prev_prec_mz[cluster_rep_indices])
    # cluster_rep_indices = np.where(bucket_clusters['is_representative'])[0]  # relative indices
    
   
    rep_ids = clusters[cluster_rep_indices]
    # print(rep_ids)
    # print(bucket_clusters['cluster'].unique())
    # print(rep_ids, cluster_rep_indices)

    similarity_metrics = []
    beta = 0.5
    # print("cluster frequencies")
    # cluster_counts = bucket_clusters['cluster'].value_counts()
    # print(cluster_counts)  
    # 
    # 
    # change this to find average distance within cluster, using rep 
  
    # for rep_id in rep_ids:
    #     ids = np.where(bucket_clusters['cluster'].values == rep_id)[0]
    #     # if (len(ids)>1):
    #     #     # print("ids of cluster", ids)
    #     # print(rep_id)
    #     # print("ids", ids)
    #     cluster_hvs = bucket_prev_hv[ids]
    #     # print(len(cluster_hvs))
    #     cluster_prec_mz = bucket_prev_prec_mz[ids]
    #     # print(len(cluster_prec_mz))
    #     pw_dist = fast_nb_cosine_dist_mask(cluster_hvs, cluster_prec_mz, config.precursor_tol[0], output_type)
    #     # print(pw_dist.shape)
    #     # print(pw_dist)
    #     mean = float(pw_dist.mean())
    #     #cprint("average distance within ", rep_id, mean, len(cluster_hvs))
    #     std = float(pw_dist.std())
    #     # print("mean", mean, "std", std)
    #     score = mean# - std*beta
    #     # if (score > 0.20):
    #     #     # print(f"score = {score:.20f}", score ==0)
    #     #    # sys.exit(0)
    #     # if (score <=0.0):
    #     #     score = incremental_eps
    #     similarity_metrics.append(score)

    # print(similarity_metrics)
    # print("debugging hypervectors")
    # for i in cluster_rep_indices:
    #     if i < len(cluster_rep_hvs) and i < len(rep_ids):
    #         print(rep_ids[i],cluster_rep_hvs[i])

    # print("pbucket_slice:", pbucket_slice)
    # print("len bucket_prev_hv:", len(bucket_prev_hv))
    # print("len bucket_clusters:", len(bucket_clusters))
    # print("prev_meta_df bucket at slice positions:")
    # print(prev_meta_df.iloc[pbucket_slice[0]:pbucket_slice[1]+1]['bucket'].value_counts())

    pw_dist_rep = fast_nb_cosine_dist_mask(
    cluster_rep_hvs,
    cluster_rep_mz,
    config.precursor_tol[0],
    output_type
)

    # if pw_dist_rep.shape[0] > 1:
    #     d = pw_dist_rep.copy()
    #     np.fill_diagonal(d, np.inf)

    #     closest_dist_per_rep = d.min(axis=1)
    #     avg_closest_dist = closest_dist_per_rep.mean()
    #     med_closest_dist = np.median(closest_dist_per_rep)
    #     max_closest_dist = np.percentile(closest_dist_per_rep, 100)
    #     mid_dist = np.percentile(closest_dist_per_rep, 90)
    #     std_closest_dist = closest_dist_per_rep.std()
    #     anomaly_eps = min(max_closest_dist+0.01, 0.9)

    # else:
    #     avg_closest_dist = np.nan
    #     med_closest_dist = np.nan
    #     std_closest_dist = np.nan
    #     mid_dist = np.nan
    #     max_closest_dist = 1.0
    #     anomaly_eps = 0.9
        

    #weighted 

    if pw_dist_rep.shape[0] > 1:
        d = pw_dist_rep.copy()
        np.fill_diagonal(d, np.inf)

        closest_dist_per_rep = d.min(axis=1)

                # convert CuPy -> NumPy if needed
        if hasattr(closest_dist_per_rep, "get"):
            closest_dist_per_rep_np = closest_dist_per_rep.get()
        else:
            closest_dist_per_rep_np = closest_dist_per_rep

        weights = np.asarray(cluster_rep_freqs, dtype=float)
        weights = weights / weights.sum()

        avg_closest_dist = np.average(closest_dist_per_rep, weights=weights)
        med_closest_dist = weighted_percentile(closest_dist_per_rep_np, weights, 50)
        mid_dist = weighted_percentile(closest_dist_per_rep_np, weights, 90)
        max_closest_dist = weighted_percentile(closest_dist_per_rep_np, weights, ANOMALY_EPS_PERCENTILE)
        max_closest_dist_unweighted = np.percentile(closest_dist_per_rep, 100)
        print("weighted max distance", max_closest_dist, "unweighted max distance", max_closest_dist_unweighted)
        std_closest_dist = np.sqrt(
            np.average((closest_dist_per_rep - avg_closest_dist) ** 2, weights=weights)
        )

        anomaly_eps = min(max_closest_dist + 0.01, 0.9)

    else:
        avg_closest_dist = np.nan
        med_closest_dist = np.nan
        std_closest_dist = np.nan
        mid_dist = np.nan
        max_closest_dist = 1.0
        anomaly_eps = 0.9


    # print("AVG CLOSEST DIST", avg_closest_dist)
    # print("MED CLOSEST DIST",  med_closest_dist)
    # print("MAX CLOSEST DIST", max_closest_dist )
    # print("STD CLOSEST DIST",std_closest_dist )
    # print("MID DIST", mid_dist)
    avg_closest_dist = avg_closest_dist
    max_closest_dist = min(max_closest_dist+0.01, 0.9)
    # print("AVG CLOSEST DIST", avg_closest_dist,"MAX CLOSEST DIST",max_closest_dist)


    full_hvs = np.concatenate((cluster_rep_hvs, bucket_hv), axis=0)
  
    bucket_prec_mz = np.concatenate([cluster_rep_mz, bucket_prec_mz])

    rep_indices = np.arange(len(cluster_rep_indices))
    split_index = len(cluster_rep_hvs)

    total_len =  len(bucket_hv)
    final_labels = np.full(total_len, -1, dtype=int)

    pw_dist = fast_nb_cosine_dist_mask(full_hvs, bucket_prec_mz, config.precursor_tol[0], output_type)

 
    if config.use_gpu_cluster or output_type == "cupy":
        pw_dist = cp.asnumpy(pw_dist)

    unique_clusters = np.unique(clusters)
    next_cluster_id = max(unique_clusters[unique_clusters != -1], default=-1) + 1

    if rep_indices.size  == 0:
        final_labels = np.full(total_len, -1, dtype=int)
        representative_mask = np.ones(total_len, dtype=bool)  
        anomaly_mask = np.zeros(len(bucket_hv), dtype=bool)
        prev_rep_mask = bucket_clusters['is_representative'].to_numpy()
        return [final_labels, representative_mask, prev_rep_mask, anomaly_mask]


    # dist_matrix = pw_dist[len(cluster_reps):, rep_indices]  
    # print("rep_ids:", rep_ids)
    # print("rep_indices:", cluster_rep_indices)
    # for i, (rid, ridx) in enumerate(zip(rep_ids, cluster_rep_indices)):
    #     print(f"  slot {i}: cluster={rid}, hv_index={ridx}, "
    #         f"bucket_cluster_at_index={clusters[ridx] if ridx < len(clusters) else 'OOB'}")

    # print("cluster rep mz values:")
    # for i, (rid, mz) in enumerate(zip(rep_ids, cluster_rep_mz.flatten())):
    #     print(f"  cluster={rid}, mz={mz:.4f}")

    # Also print the new spectrum's mz
    # print("new spectrum mz:", bucket_prec_mz[split_index:split_index+3])
    dist_matrix = pw_dist[split_index:, :split_index]
    # print("dist_matrix", dist_matrix.shape)
    # # print(dist_matrix)
    # print("lenght of bucket hvs", len(bucket_hv))
    # print("lenght of prev bucket hvs", len(bucket_prev_hv))
    # print("full_hvs", full_hvs.shape, bucket_hv.shape, cluster_rep_hvs.shape)
    # print("dist_matrix", dist_matrix.shape, "pw_dist", pw_dist.shape)
  
    best_idx = np.argmin(dist_matrix, axis=1)
    # print(dist_matrix)
    best_dists = dist_matrix[np.arange(dist_matrix.shape[0]), best_idx]

    # print(best_idx[0])
    # print(best_dists[0])

   # "distance look too high for some rows"

 
    current_next_cluster_id = next_cluster_id

    representative_mask = np.zeros(len(bucket_hv), dtype=bool)
    anomaly_mask = np.zeros(len(bucket_hv), dtype=bool)
    singleton_indices = []
    cluster_id_before_incr = current_next_cluster_id
    q = 1
    # anomaly_eps = incremental_eps = np.percentile(closest_dist_per_rep, 65) #avg_closest_dist    #max_closest_dist #0.45

    

    # if 3090 in bucket_clusters['cluster'].unique():
    #     print('found 3090')
    #     sys.exit(0)
    for j, (best_cluster_idx, dist) in enumerate(zip(best_idx, best_dists)):
        global_i = j
        # anomaly_eps = similarity_metrics[best_cluster_idx]
        54, 89, 85
        # if (metadata_df.iloc[j]['cluster'] == 54 or metadata_df.iloc[j]['cluster']==89 or metadata_df.iloc[j]['cluster']==85):
        #     # sys.exit(0)
        # print("J",j)
        threshold = anomaly_eps
        # if (similarity_metrics[best_cluster_idx] >0): 
        #     threshold = similarity_metrics[best_cluster_idx]
        if dist <= threshold:
            # print("dist", dist, "anomaly_eps", anomaly_eps)
            
            final_labels[j] = rep_ids[best_cluster_idx] 
            bucket_rep_relative_index = rep_indices[best_cluster_idx]
            original_prev_index = pbucket_slice[0] + bucket_rep_relative_index
            
            if (prev_clusters.at[original_prev_index,'anomaly']==True):
                print("joining anomaly cluster")
                # anomaly_mask[j] = True
            # print("not anomaly")
            # prev_rep_row = cluster_rep_indices[best_cluster_idx]
            # if bucket_clusters.iloc[j]['anomaly']:
            #     anomaly_mask[j] = True
            # else:
            # print("adding to cluster", rep_ids[best_cluster_idx] , rep_ids, j)
            # print('real cluster ',metadata_df.iloc[j]['cluster'],rep_ids, bucket)
            # if (metadata_df.iloc[j]['cluster'] == 18723.0 or metadata_df.iloc[j]['cluster']==28038 or metadata_df.iloc[j]['cluster']==22736):

#49504
# 21307, 20530, 40322
#4 34903, 34, 5261
#2 55198, 35003, 28
#30410, 85845,165617
            if ("anomal" in metadata_df.iloc[j]['identifier']):
            # if (metadata_df.iloc[j]['cluster'] == 30412 or metadata_df.iloc[j]['cluster']==84977 or metadata_df.iloc[j]['cluster']==162980):
            #if (metadata_df.iloc[j]['cluster']==326516):
            # if (metadata_df.iloc[j]['cluster'] == 10854):
                # print("NOT MARKING AS ANOMALY but is anomaly incorrect", rep_ids[best_cluster_idx], "dist", dist, "anomaly_eps", anomaly_eps, j, j==(bucket_slice[1]+1), bucket_slice[1]+1)
                # print('real cluster ',metadata_df.iloc[j]['cluster'],rep_ids, bucket)
                # print("number of spectras in cluster", len(bucket_clusters[bucket_clusters['cluster']== rep_ids[best_cluster_idx]]))
                # print("anomaly spectra hypervector")
                # print("rep hypervector", cluster_rep_hvs[best_cluster_idx])
                # print("current hypervector", bucket_hv[j])
                # print("rep hypervector")
                rep_cluster_id = rep_ids[best_cluster_idx]
                rep_relative_idx = rep_indices[best_cluster_idx]
                rep_original_idx = pbucket_slice[0] + rep_relative_idx
                rep_metadata = prev_meta_df.iloc[rep_original_idx]
                # print("rep hypervector", cluster_rep_hvs[best_cluster_idx])
                # print("current hypervector", bucket_hv[j])
                # print(f"\nCluster {rep_cluster_id} representative metadata:")
                # print(rep_metadata.to_dict())
              
    

           
        else:
            final_labels[j] = current_next_cluster_id
            representative_mask[j] = True
            anomaly_mask[j] = True 
            # if (metadata_df.iloc[j]['cluster'] == 18723.0 or metadata_df.iloc[j]['cluster']==28038 or metadata_df.iloc[j]['cluster']==22736):

            #54, 89, 85
    

            #if (metadata_df.iloc[j]['cluster'] == 30412 or metadata_df.iloc[j]['cluster']==84977 or metadata_df.iloc[j]['cluster']==162980):
            # if ("anomal" in metadata_df.iloc[j]['identifier']):
            # #if (metadata_df.iloc[j]['cluster']==326516):
            #     # print("marking as anomaly correct",  rep_ids[best_cluster_idx], dist, anomaly_eps, j, j==(bucket_slice[1]+1), bucket_slice[1]+1)
            #     # print('real cluster ',metadata_df.iloc[j],rep_ids, bucket)
            #     rep_cluster_id = rep_ids[best_cluster_idx]
            #     rep_relative_idx = rep_indices[best_cluster_idx]
            #     rep_original_idx = pbucket_slice[0] + rep_relative_idx
            #     rep_metadata = prev_meta_df.iloc[rep_original_idx]

            
                # print(f"\nCluster {rep_cluster_id} representative metadata:")
                # print(rep_metadata.to_dict())
    
            # else:
                # print("Farhat has changed threshold to be 1.0 for debugging")
                # print("marking as anomaly but wrong",  rep_ids[best_cluster_idx], "dist", dist, "anomaly eps", anomaly_eps, j, j==(bucket_slice[1]+1), bucket_slice[1]+1)
                # print("curent_next_cluster_id", current_next_cluster_id)
                # print("bucket", bucket)
                # print("index of 3090", np.where(rep_ids==3090))
                # print("best_dists", best_dists)
   
           
            
                # print('real cluster ',metadata_df.iloc[j],"all clusters with reps", rep_ids, "all unique clusters", bucket_clusters['cluster'].unique())
                # print("number of spectras in prev_clusters", len(prev_clusters[prev_clusters['cluster']==metadata_df.iloc[j]['cluster']]))
                # print(prev_clusters[prev_clusters['cluster']==metadata_df.iloc[j]['cluster']].head())
                # print("SANITY CHECK")
                # tol = 0.6
                # print(prev_clusters[(prev_clusters['retention_time']<  6826.967285  -tol) & (prev_clusters['retention_time']>  6826.967285   -tol) ])
                # print(prev_clusters[prev_clusters['scan']==30848])
                # print("current hyper_vector", bucket_hv[j])


                # key_cols = ["bucket", "precursor_charge", "identifier", "scan", "retention_time"]

                # # bucket_clusters = (
                # #     bucket_prev_meta_df[key_cols]
                # #     .merge(bucket_clusters, on=key_cols, how="left")
                # # )
                # # print(bucket_clusters)


            #     mask2 = (
            #     (bucket_prev_meta_df['bucket'] == 807) &
            #     (bucket_prev_meta_df['scan'] == 30848) &
            #     (bucket_prev_meta_df['identifier'] == 'b1929_293T_proteinID_09A_QE3_122212')
            # )


            #     mask3 = (
            #     (bucket_clusters['bucket'] == 807) &
            #     (bucket_clusters['scan'] == 30848) &
            #     (bucket_clusters['identifier'] == 'b1929_293T_proteinID_09A_QE3_122212')
            # )


            #     idx2 = bucket_prev_meta_df.index[mask2][0]
            #     idxc = bucket_clusters.index[mask3][0]
            #     print(idxc, idx2)
            #     print(len(prev_clusters), len(prev_meta_df),len(prev_hvs))
            #     print(prev_clusters.iloc[idxc])
            #     hv2 = bucket_prev_hv[idx2]
            #     print("Row 2 (scan 30848):", hv2)

            #     expected_hv = np.array([
            #     2639494158, 1913736923, 2811366689, 441464881, 1592326922, 440449055,
            #     3256846091, 428866961, 3930818428, 353285938, 2510482082, 1541591334,
            #     1513591703, 2506424925, 4127222681, 1264434451, 2220083608, 3939867457,
            #     2960932736, 2632371182, 623423894, 3610463554, 41775151, 1143420472,
            #     2352861657, 45502789, 1448068190, 3538224192, 3262448791, 3770978441,
            #     3572500864, 68439244, 2310123756, 2348162116, 2703039760, 1849435181,
            #     3579086695, 241657792, 417611126, 4286702408, 1578887641, 1836224872,
            #     1801528603, 727701706, 160955076, 512232037, 1394542736, 2147949590,
            #     2655672356, 809206011, 876199161, 2164027774, 2662295725, 72029633,
            #     1881314566, 3203881984, 941859473, 3516014685, 2153063622, 2598974161,
            #     1217013485, 2172869892, 2893156857, 791885495
            # ])

            #     print(np.array_equal(expected_hv,hv2))
            #     print(hv2)
            #     print(expected_hv)
        
            #     print(bucket_clusters)
            #     print(bucket_prev_meta_df)
             
               # sys.exit(0)

              
                # print("represenative bucket clusters")
                # print(bucket_clusters[bucket_clusters['is_representative']==True])
               #sys.exit(0)

            current_next_cluster_id += 1
            singleton_indices.append(len(cluster_rep_hvs)+j)

    
    
    # print(metadata_df.iloc[0:3])
    # sys.exit(0)
    # clusters_p, count_p = np.unique(bucket_clusters['cluster'].to_numpy(), return_counts=True)
    
    unique_clusters_f, counts_f = np.unique(final_labels, return_counts=True)
   
    rep_mask = np.isin(unique_clusters_f, rep_ids)

    old_count_dict = dict(zip(clusters_p, count_p))
    counts_old_aligned = np.array([old_count_dict.get(c, 0) for c in unique_clusters_f])

    # Compute new counts
    counts_new = counts_f + counts_old_aligned
    # print("comparing rep and anomaly mask size",  len(representative_mask), len(anomaly_mask))

    # if (n > 0):
    #     print(n)
    #     metadata_df['anomaly'] = anomaly_mask
    #     print(metadata_df)
    #     print(anomaly_mask)

    return [final_labels, representative_mask, prev_rep_mask, anomaly_mask]


def hcluster_bucket(
    bucket_slice: tuple, 
    data_dict: dict, 
    linkage: str,
    config: Config,
    output_type: str='numpy'
):
    if bucket_slice[1]-bucket_slice[0]==0:
        return [np.array([-1]), np.array([True], dtype=np.bool)]
    else:
        bucket_slice[1] += 1
        bucket_hv = data_dict['hv'][bucket_slice[0]: bucket_slice[1]]
        bucket_prec_mz = data_dict['prec_mz'][bucket_slice[0]: bucket_slice[1]]
        bucket_rt_time = data_dict['rt_time'][bucket_slice[0]: bucket_slice[1]]

        # s = time.time()
        pw_dist = fast_nb_cosine_dist_condense(bucket_hv, bucket_prec_mz, config.precursor_tol[0], output_type)
        # pw_dist = squareform(pw_dist).astype(np.float32)
        # e = time.time()
        # print("Time pw_dist: ", e-s)
        
        # s = time.time()
        lk = fastcluster.linkage(pw_dist, linkage)
        # e = time.time()
        # print("Time linkage: ", e-s)

        # s = time.time()
        L = fcluster(lk, config.eps, 'distance') - 1
        # e = time.time()
        # print("Time cluster: ", e-s)
        
        cluster_labels_refined = refine_cluster(
            bucket_cluster_label = L, 
            bucket_precursor_mzs = bucket_prec_mz,
            bucket_rts = bucket_rt_time,
            precursor_tol_mass = config.precursor_tol[0], 
            precursor_tol_mode = config.precursor_tol[1], 
            rt_tol = config.rt_tol)
        
        pw_dist = squareform(pw_dist).astype(np.float32)
        representative_mask = get_cluster_representative(
            cluster_labels=cluster_labels_refined, pw_dist=pw_dist)
        
        return [cluster_labels_refined, representative_mask]
 

def hcluster_par_bucket(
    bucket_slice: tuple, 
    bucket_hv: np.ndarray,
    bucket_prec_mz: np.ndarray,
    bucket_rt_time: np.ndarray,
    linkage: str,
    precursor_tol: list,
    eps: float,
    rt_tol: float,
    output_type: str='numpy'
):
    if bucket_slice[1]-bucket_slice[0]==0:
        return [np.array([-1]), np.array([True], dtype=np.bool)]
    else:
        pw_dist = fast_nb_cosine_dist_condense(bucket_hv, bucket_prec_mz, precursor_tol[0], output_type)

        lk = fastcluster.linkage(pw_dist, linkage)

        L = fcluster(lk, eps, 'distance') - 1

        cluster_labels_refined = refine_cluster(
            bucket_cluster_label = L, 
            bucket_precursor_mzs = bucket_prec_mz,
            bucket_rts = bucket_rt_time,
            precursor_tol_mass = precursor_tol[0], 
            precursor_tol_mode = precursor_tol[1], 
            rt_tol = rt_tol)
                
        pw_dist = squareform(pw_dist).astype(np.float32)
        representative_mask = get_cluster_representative(
            cluster_labels=cluster_labels_refined, pw_dist=pw_dist)
        
        return [cluster_labels_refined, representative_mask]
    
def cluster_spectra_incr(
    spectra_by_charge_df: pd.DataFrame,
    encoded_spectra_hv: np.array,
    prev_spectra_by_charge_df: pd.DataFrame,
    prev_encoded_spectra_hv: np.array,
    prev_cluster_results: pd.DataFrame,
    config: Config,
    logger: logging):
     # Save data to shared memory
    start = time.time()

    print("PREVIUS CLUSTR RESULTS LENGTH IN CLUSTER_SPECTRA_INCR", len(prev_cluster_results))

    print("entry to cluster_spectra_incr:",
      (spectra_by_charge_df["cluster"] == 511196).sum())


    data_dict = {
        'hv': encoded_spectra_hv,
        'prec_mz': np.vstack(spectra_by_charge_df.precursor_mz).astype(np.float32),
        'rt_time': np.vstack(spectra_by_charge_df.retention_time).astype(np.float32),
        'meta_data':spectra_by_charge_df}

    ## Start clustering in GPU or CPU #

    #pectra_by_charge_df = pd.concat([prev_spectra_by_charge_df, spectra_by_charge_df], ignore_index=True)
    print("CLUSER INCR")
    # print(spectra_by_charge_df.head())
    bucket_idx_dict = schedule_bucket(spectra_by_charge_df, logger)
   #print(bucket_idx_dict[0][0])
   #print(bucket_idx_dict)
  #  test_bucket = spectra_by_charge_df.loc[bucket_idx_dict[0][0], 'bucket']
  #  print(bucket_idx_dict)

   # print("TEST BUCKET", test_bucket)
    np.set_printoptions(threshold=np.inf)
    print("checkign for 2897 bucket", bucket_idx_dict['sort_bucket_idx_arr'])
    target_bucket = 2987

    buckets = spectra_by_charge_df["bucket"].unique()

    print("bucket exists?", target_bucket in buckets)
    print("matching rows:", (spectra_by_charge_df["bucket"] == target_bucket).sum())
                 
    mask = spectra_by_charge_df["cluster"] == 511196
    if( mask.sum() >= 1):
        print("FOUND 511196 rows in spec_df_by_charge:", mask.sum())
        print("FOUUND BUCKET AND CLUSTER WHERE THE FUCK ARE YOU?")
 


   

    cluster_device = 'CPU'
    if config.cluster_alg == 'dbscan':
        if config.use_gpu_cluster:
            # DBSCAN clustering on GPU
            cluster_func = cuml.DBSCAN(
                    eps=config.eps, min_samples=1, metric='precomputed',
                calc_core_sample_indices=False, output_type='numpy')

            cluster_device = 'GPU'
        else:
            # DBSCAN clustering on CPU
            cluster_func = DBSCAN(eps=config.eps, min_samples=1, metric='precomputed', n_jobs=config.cpu_core_cluster)
#switch incremental clustering
        if config.incre_mode:
             
            cluster_results = [cluster_bucket_incr_2(
                bucket_slice = b_slice_i,
                data_dict = data_dict,
                prev_clusters = prev_cluster_results,
                prev_hvs = prev_encoded_spectra_hv,
                config = config,
                prev_prec_mz = prev_spectra_by_charge_df.precursor_mz.to_numpy().astype(np.float32),
                output_type = 'cupy' if config.use_gpu_cluster else 'numpy',
                prev_meta_df = prev_spectra_by_charge_df,
                bucket = spectra_by_charge_df['bucket'].iloc[b_slice_i[0]],
                cluster_func = cluster_func)
                for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]
        else:
            cluster_results = [cluster_bucket(
                bucket_slice = b_slice_i,
                data_dict = data_dict,
                config = config,
                cluster_func = cluster_func,
                output_type = 'cupy' if config.use_gpu_cluster else 'numpy')
                for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]

    elif config.cluster_alg == 'louvain':
        if config.use_gpu_cluster:
            # DBSCAN clustering on GPU
            cluster_func = "louvain"
            cluster_device = 'GPU'
        else:
            # DBSCAN clustering on CPU
            cluster_func = "louvain"
#switch incremental clustering
        if config.incre_mode:
            

            cluster_results = [cluster_bucket_incr_2(
                bucket_slice = b_slice_i,
                data_dict = data_dict,
                prev_clusters = prev_cluster_results,
                prev_hvs = prev_encoded_spectra_hv,
                config = config,
                prev_prec_mz = prev_spectra_by_charge_df.precursor_mz.to_numpy().astype(np.float32),
                output_type = 'cupy' if config.use_gpu_cluster else 'numpy',
                prev_meta_df = prev_spectra_by_charge_df,
                bucket = spectra_by_charge_df['bucket'].iloc[b_slice_i[0]],
                cluster_func = cluster_func)
                for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]
        else:
            cluster_results = [cluster_bucket(
                bucket_slice = b_slice_i,
                data_dict = data_dict,
                config = config,
                cluster_func = cluster_func,
                output_type = 'cupy' if config.use_gpu_cluster else 'numpy')
                for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]

    elif config.cluster_alg in ['hc_single', 'hc_complete', 'hc_average']:
        with Parallel(n_jobs=config.cpu_core_cluster) as parallel:
            cluster_results = parallel(delayed(hcluster_par_bucket)(
                b_slice_i,
                data_dict['hv'][b_slice_i[0]: b_slice_i[1]+1],
                data_dict['prec_mz'][b_slice_i[0]: b_slice_i[1]+1],
                data_dict['rt_time'][b_slice_i[0]: b_slice_i[1]+1],
                config.cluster_alg[3:], config.precursor_tol, config.eps, config.rt_tol, 'numpy')
                    for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr']))

        # cluster_results = [hcluster_bucket(
        #     bucket_slice=b_slice_i,
        #     data_dict=data_dict,
        #     linkage=config.cluster_alg[3:],
        #     config=config,
        #     output_type='numpy')
        #     for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]

    else:
        raise Exception("Error clustering algorithm: " + config.cluster_alg)



    #rint("LENGTH OF CLUSTER RESULTS after bucke functio", len(cluster_results))
    #print(len(bucket_idx_dict['reorder_idx']))
    #print(len(cluster_results))
    tot = 0
  #  for i in range(len(cluster_results)):
   #     tot += len(cluster_results[i][0])
   # print("TOTAL CLUSTERS", tot)
   # print(bucket_idx_dict['reorder_idx'])

    spectra_by_charge_df
    prev_cluster_labels = np.array(prev_cluster_results['cluster'])
    prev_rep_mask = np.array(prev_cluster_results['is_representative'])
    prev_anomaly_mask = np.array(prev_cluster_results['anomaly'])
    print("checking prev_cluster_results in cluster_spec_incr")
    # print(prev_cluster_results[(prev_cluster_results['bucket']==1249) & (prev_cluster_results['identifier']=='b1927_293T_proteinID_07A_QE3_122212')])
   
    # sys.exit(0)
    
    # Re-order cluster results
    cluster_results = [cluster_results[i] for i in bucket_idx_dict['reorder_idx']]



    cluster_labels = [res_i[0] for res_i in cluster_results]
    cluster_labels = assign_unique_cluster_labels(cluster_labels)
    cluster_labels = np.hstack(cluster_labels)

    representative_mask = np.hstack([res_i[1] for res_i in cluster_results])
    anomaly_mask = np.hstack([res_i[3] for res_i in cluster_results])
    print("anomaly mask length", len(anomaly_mask))
    print("rep mask length", len(representative_mask))
    #find way to return anomaly mask in legible/easy way 
    
    rep_clusters_new = set(cluster_labels[representative_mask])
    mask_to_update = np.isin(prev_cluster_labels, list(rep_clusters_new))
    prev_rep_mask[mask_to_update] = False
    
    #print(prev_rep_mask) 

    logger.info("{} clustering in {:.4f} s".format(cluster_device, time.time()-start))


    if (config.incre_mode):
        # cluster_labels = np.concatenate((prev_cluster_labels, cluster_labels))
        print("prev_cluster size", len(prev_cluster_labels), "new clusters label size", len(cluster_labels))
        # representative_mask = np.concatenate((prev_rep_mask, representative_mask))
        # anomaly_mask = np.concatenate((prev_anomaly_mask, anomaly_mask))
        
        return cluster_labels, representative_mask, anomaly_mask, cluster_labels

    print("NUMBER OF -1 VALUES in cluster labels", len(prev_cluster_labels), np.sum(np.array(prev_cluster_labels) == -1), len(cluster_labels), np.sum(np.array(prev_cluster_labels) == -1))
    print(cluster_labels, representative_mask)
    return cluster_labels, representative_mask

def cluster_spectra(
    spectra_by_charge_df: pd.DataFrame,
    encoded_spectra_hv: np.ndarray,
    config: Config,
    logger: logging
):
    # Save data to shared memory
    start = time.time()
   
    data_dict = {
        'hv': encoded_spectra_hv, 
        'prec_mz': np.vstack(spectra_by_charge_df.precursor_mz).astype(np.float32),
        'rt_time': np.vstack(spectra_by_charge_df.retention_time).astype(np.float32)}
    
    ## Start clustering in GPU or CPU ##
    bucket_idx_dict = schedule_bucket(spectra_by_charge_df, logger)
    
    cluster_device = 'CPU'
    if config.cluster_alg == 'dbscan':
        if config.use_gpu_cluster:
            # DBSCAN clustering on GPU
            cluster_func = cuml.DBSCAN(
                eps=config.eps, min_samples=2, metric='precomputed',
                calc_core_sample_indices=False, output_type='numpy')


            cluster_device = 'GPU'
        else:
            # DBSCAN clustering on CPU
            cluster_func = DBSCAN(eps=config.eps, min_samples=2, metric='precomputed', n_jobs=config.cpu_core_cluster)
#switch incremental clustering
        if config.incre_mode:
            cluster_results = [cluster_bucket_incr_batch(
                bucket_slice = b_slice_i, 
                data_dict = data_dict,
                config = config,
                cluster_func = cluster_func,
                output_type = 'cupy' if config.use_gpu_cluster else 'numpy') 
                for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]
        else:
            cluster_results = [cluster_bucket(
                bucket_slice = b_slice_i, 
                data_dict = data_dict,
                config = config,
                cluster_func = cluster_func,
                output_type = 'cupy' if config.use_gpu_cluster else 'numpy') 
                for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]

    elif config.cluster_alg == 'louvain':
        if config.use_gpu_cluster:
            # DBSCAN clustering on GPU
            cluster_func = "louvain"
            cluster_device = 'GPU'
        else:
            # DBSCAN clustering on CPU
            cluster_func = "louvain"
#switch incremental clustering
        if config.incre_mode:


            cluster_results = [cluster_bucket_incr_2(
                bucket_slice = b_slice_i,
                data_dict = data_dict,
                prev_clusters = prev_cluster_results,
                prev_hvs = prev_encoded_spectra_hv,
                config = config,
                prev_prec_mz = prev_spectra_by_charge_df.precursor_mz.to_numpy().astype(np.float32),
                output_type = 'cupy' if config.use_gpu_cluster else 'numpy',
                prev_meta_df = prev_spectra_by_charge_df,
                bucket = spectra_by_charge_df['bucket'].iloc[b_slice_i[0]],
                cluster_func = cluster_func)
                for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]
        else:
            cluster_results = [cluster_bucket(
                bucket_slice = b_slice_i,
                data_dict = data_dict,
                config = config,
                cluster_func = cluster_func,
                output_type = 'cupy' if config.use_gpu_cluster else 'numpy')
                for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]

    elif config.cluster_alg in ['hc_single', 'hc_complete', 'hc_average']:
        with Parallel(n_jobs=config.cpu_core_cluster) as parallel:
            cluster_results = parallel(delayed(hcluster_par_bucket)(
                b_slice_i, 
                data_dict['hv'][b_slice_i[0]: b_slice_i[1]+1],
                data_dict['prec_mz'][b_slice_i[0]: b_slice_i[1]+1],
                data_dict['rt_time'][b_slice_i[0]: b_slice_i[1]+1],
                config.cluster_alg[3:], config.precursor_tol, config.eps, config.rt_tol, 'numpy')
                    for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr']))
                   
        # cluster_results = [hcluster_bucket(
        #     bucket_slice=b_slice_i, 
        #     data_dict=data_dict,
        #     linkage=config.cluster_alg[3:],
        #     config=config, 
        #     output_type='numpy') 
        #     for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]
        
    else:
        raise Exception("Error clustering algorithm: " + config.cluster_alg)
 
 
    # Re-order cluster results
    cluster_results = [cluster_results[i] for i in bucket_idx_dict['reorder_idx']]
    
    cluster_labels = [res_i[0] for res_i in cluster_results]
    cluster_labels = assign_unique_cluster_labels(cluster_labels)
    cluster_labels = np.hstack(cluster_labels)
        
    representative_mask = np.hstack([res_i[1] for res_i in cluster_results])
    
    logger.info("{} clustering in {:.4f} s".format(cluster_device, time.time()-start))

    return cluster_labels, representative_mask


    
def cluster_encoded_spectra(
    spectra_by_charge_df: pd.DataFrame,
    encoded_spectra_hv: np.array,
    config: Config,
    logger: logging
):
    # Save data to shared memory
    start = time.time()
    
    data_dict = {
        'hv': encoded_spectra_hv, 
        'prec_mz': np.vstack(spectra_by_charge_df.precursor_mz).astype(np.float32),
        'rt_time': np.vstack(spectra_by_charge_df.retention_time).astype(np.float32)
        }
    
    ## Start clustering in GPU or CPU ##
    bucket_idx_dict = schedule_bucket(spectra_by_charge_df, logger)

    cluster_device = 'CPU'
    if config.cluster_alg == 'dbscan':
        if config.use_gpu_cluster:
            # DBSCAN clustering on GPU
            cluster_func = cuml.DBSCAN(
                eps=config.eps, min_samples=2, metric='precomputed',
                calc_core_sample_indices=False, output_type='numpy')

            cluster_device = 'GPU'
        else:
            # DBSCAN clustering on CPU
            cluster_func = DBSCAN(eps=config.eps, min_samples=2, metric='precomputed', n_jobs=config.cpu_core_cluster)

        if config.incre_mode:
            cluster_results = [cluster_bucket_incr_batch(
                bucket_slice = b_slice_i, 
                data_dict = data_dict,
                config = config,
                cluster_func = cluster_func,
                output_type = 'cupy' if config.use_gpu_cluster else 'numpy') 
                for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]
        else:
            cluster_results = [cluster_bucket(
                bucket_slice = b_slice_i, 
                data_dict = data_dict,
                config = config,
                cluster_func = cluster_func,
                output_type = 'cupy' if config.use_gpu_cluster else 'numpy') 
                for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]
            
    elif config.cluster_alg in ['hc_single', 'hc_complete', 'hc_average']:
        with Parallel(n_jobs=config.cpu_core_cluster) as parallel:
            cluster_results = parallel(delayed(hcluster_par_bucket)(
                b_slice_i, 
                data_dict['hv'][b_slice_i[0]: b_slice_i[1]+1],
                data_dict['prec_mz'][b_slice_i[0]: b_slice_i[1]+1],
                data_dict['rt_time'][b_slice_i[0]: b_slice_i[1]+1],
                config.cluster_alg[3:], config.precursor_tol, config.eps, config.rt_tol, 'numpy')
                    for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr']))
                
        # raise Exception("Un-updated clustering functions: " + config.cluster_alg)
   
        # cluster_results = [hcluster_bucket(
        #     bucket_slice=b_slice_i, 
        #     data_dict=data_dict,
        #     linkage=config.cluster_alg[3:],
        #     config=config, 
        #     output_type='numpy') 
        #     for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]
        
    else:
        raise Exception("Error clustering algorithm: " + config.cluster_alg)
 
    # Re-order cluster results
    cluster_results = [cluster_results[i] for i in bucket_idx_dict['reorder_idx']]

    cluster_labels = [res_i[0] for res_i in cluster_results]
    cluster_labels = assign_unique_cluster_labels(cluster_labels)
    cluster_labels = np.hstack(cluster_labels)
        
    representative_mask = np.hstack([res_i[1] for res_i in cluster_results])
    
    logger.info("{} clustering in {:.4f} s".format(cluster_device, time.time()-start))

    return cluster_labels, representative_mask


def refine_cluster(
    bucket_cluster_label, 
    bucket_precursor_mzs, 
    bucket_rts,
    precursor_tol_mass, 
    precursor_tol_mode, 
    rt_tol, 
    min_samples =2):
    '''
        Refine initial clusters to make sure spectra within a cluster don't 
        have an excessive precursor m/z difference.
    '''
    # Cluster refinement step
    bucket_cluster_label = bucket_cluster_label.flatten()
    order = np.argsort(bucket_cluster_label)
    reverse_order = np.argsort(order)
    sorted_cluster_label = bucket_cluster_label[order]

    sorted_bucket_precursor_mzs, sorted_bucket_rts =  bucket_precursor_mzs[order].flatten(), bucket_rts[order].flatten()

    if sorted_cluster_label[-1] == -1: # Only noise samples.
        n_clusters, n_noise = 0, len(sorted_cluster_label)
    else:
        group_idx = nb.typed.List(_get_cluster_group_idx(sorted_cluster_label))
        n_clusters = nb.typed.List(
            [_postprocess_cluster(
                sorted_cluster_label[start_i:stop_i], 
                sorted_bucket_precursor_mzs[start_i:stop_i], 
                sorted_bucket_rts[start_i:stop_i], 
                precursor_tol_mass, precursor_tol_mode, rt_tol, min_samples)
                for start_i, stop_i in group_idx])

        _assign_unique_cluster_labels(sorted_cluster_label, group_idx, n_clusters, min_samples)
        
    return sorted_cluster_label[reverse_order]


def assign_unique_cluster_labels(bucket_cluster_labels):
    '''
        Re-order and assign unique cluster labels for spectra from different charges
    '''
    reorder_labels, label_base = [], 0
    for idx_i, cluster_i in enumerate(bucket_cluster_labels):
        cluster_i = cluster_i.flatten()
                
        # Re-order and assign unique cluster labels
        noise_idx = cluster_i == -1
        num_clusters, num_noises = np.amax(cluster_i) + 1, np.sum(noise_idx)

        cluster_i[noise_idx] = np.arange(num_clusters, num_clusters + num_noises)
        cluster_i += label_base
        label_base += (num_clusters+num_noises)

        reorder_labels.append(cluster_i)
    
    return reorder_labels


@nb.njit
def _get_cluster_group_idx(clusters: np.ndarray) -> Iterator[Tuple[int, int]]:
    """
    Get start and stop indexes for unique cluster labels.
    Parameters
    ----------
    clusters : np.ndarray
        The ordered cluster labels (noise points are -1).
    Returns
    -------
    Iterator[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the unique cluster labels.
    """
    start_i = 0
    while clusters[start_i] == -1 and start_i < clusters.shape[0]:
        start_i += 1
    stop_i = start_i
    while stop_i < clusters.shape[0]:
        start_i, label = stop_i, clusters[stop_i]
        while stop_i < clusters.shape[0] and clusters[stop_i] == label:
            stop_i += 1
        yield start_i, stop_i


def _postprocess_cluster(
    cluster_labels: np.ndarray, 
    cluster_mzs: np.ndarray,
    cluster_rts: np.ndarray, 
    precursor_tol_mass: float,
    precursor_tol_mode: str, 
    rt_tol: float,
    min_samples: int
    ) -> int:
    """
    Agglomerative clustering of the precursor m/z's within each initial
    cluster to avoid that spectra within a cluster have an excessive precursor
    m/z difference.
    Parameters
    ----------
    cluster_labels : np.ndarray
        Array in which to write the cluster labels.
    cluster_mzs : np.ndarray
        Precursor m/z's of the samples in a single initial cluster.
    cluster_rts : np.ndarray
        Retention times of the samples in a single initial cluster.
    precursor_tol_mass : float
        Maximum precursor mass tolerance for points to be clustered together.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for points to be clustered together. If
        `None`, do not restrict the retention time.
    min_samples : int
        The minimum number of samples in a cluster.
    Returns
    -------
    int
        The number of clusters after splitting on precursor m/z.
    """
    cluster_labels[:] = -1
    # No splitting needed if there are too few items in cluster.
    # This seems to happen sometimes despite that DBSCAN requires a higher
    # `min_samples`.
    if len(cluster_labels) < min_samples:
        n_clusters = 0
    else:
        # Group items within the cluster based on their precursor m/z.
        # Precursor m/z's within a single group can't exceed the specified
        # precursor m/z tolerance (`distance_threshold`).
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like Scikit-Learn does).
        cluster_assignments = fcluster(
            _linkage(cluster_mzs, precursor_tol_mode),
            precursor_tol_mass, 'distance') - 1

        # Optionally restrict clusters by their retention time as well.
        if rt_tol is not None:
            cluster_assignments_rt = fcluster(
                _linkage(cluster_rts), rt_tol, 'distance') - 1
            # Merge cluster assignments based on precursor m/z and RT.
            # First prime factorization is used to get unique combined cluster
            # labels, after which consecutive labels are obtained.
            cluster_assignments = np.unique(
                cluster_assignments * 2 + cluster_assignments_rt * 3,
                return_inverse=True)[1]

        n_clusters = cluster_assignments.max() + 1
        # Update cluster assignments.
        if n_clusters == 1:
            # Single homogeneous cluster.
            cluster_labels[:] = 0
        elif n_clusters == cluster_mzs.shape[0]:
            # Only singletons.
            n_clusters = 0
        else:
            unique_clusters, inverse, counts = np.unique(
                cluster_assignments, return_inverse=True, return_counts=True)
            non_noise_clusters = np.where(counts >= min_samples)[0]
            labels = -np.ones_like(unique_clusters)
            labels[non_noise_clusters] = np.unique(unique_clusters[non_noise_clusters], return_inverse=True)[1]
            cluster_labels[:] = labels[inverse]
            n_clusters = len(non_noise_clusters)
    return n_clusters


@nb.njit(cache=True, fastmath=True)
def _linkage(
    values: np.ndarray, 
    tol_mode: str = None
    ) -> np.ndarray:
    """
    Perform hierarchical clustering of a one-dimensional m/z or RT array.
    Because the data is one-dimensional, no pairwise distance matrix needs to
    be computed, but rather sorting can be used.
    For information on the linkage output format, see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    Parameters
    ----------
    values : np.ndarray
        The precursor m/z's or RTs for which pairwise distances are computed.
    tol_mode : str
        The unit of the tolerance ('Da' or 'ppm' for precursor m/z, 'rt' for
        retention time).
    Returns
    -------
    np.ndarray
        The hierarchical clustering encoded as a linkage matrix.
    """
    linkage = np.zeros((values.shape[0] - 1, 4), np.double)
    # min, max, cluster index, number of cluster elements
    # noinspection PyUnresolvedReferences
    clusters = [(values[i], values[i], i, 1) for i in np.argsort(values)]
    for it in range(values.shape[0] - 1):
        min_dist, min_i = np.inf, -1
        for i in range(len(clusters) - 1):
            dist = clusters[i + 1][1] - clusters[i][0]  # Always positive.
            if tol_mode == 'ppm':
                dist = dist / clusters[i][0] * 10 ** 6
            if dist < min_dist:
                min_dist, min_i = dist, i
        n_points = clusters[min_i][3] + clusters[min_i + 1][3]
        linkage[it, :] = [clusters[min_i][2], clusters[min_i + 1][2],
                          min_dist, n_points]
        clusters[min_i] = (clusters[min_i][0], clusters[min_i + 1][1],
                           values.shape[0] + it, n_points)
        del clusters[min_i + 1]

    return linkage


@nb.njit(cache=True)
def _assign_unique_cluster_labels(
    cluster_labels: np.ndarray,
    group_idx: nb.typed.List,
    n_clusters: nb.typed.List,
    min_samples: int) -> None:
    """
    Make sure all cluster labels are unique after potential splitting of
    clusters to avoid excessive precursor m/z differences.
    Parameters
    ----------
    cluster_labels : np.ndarray
        Cluster labels per cluster grouping.
    group_idx : nb.typed.List[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the cluster groupings.
    n_clusters: nb.typed.List[int]
        The number of clusters per cluster grouping.
    min_samples : int
        The minimum number of samples in a cluster.
    """
    current_label = 0
    for (start_i, stop_i), n_cluster in zip(group_idx, n_clusters):
        if n_cluster > 0 and stop_i - start_i >= min_samples:
            current_labels = cluster_labels[start_i:stop_i]
            current_labels[current_labels != -1] += current_label
            current_label += n_cluster
        else:
            cluster_labels[start_i:stop_i] = -1
        
        # print(cluster_labels[start_i:stop_i])


# @nb.njit(cache=True, parallel=True)
def get_cluster_representative(
    cluster_labels: np.ndarray,
    pw_dist: np.ndarray
    ) -> np.ndarray:
    """
    Get indexes of the cluster representative spectra (medoids).
    Parameters
    ----------
    clusters : np.ndarray
        Cluster label assignments.
    pw_dist : np.ndarray
        The condense pairwise distance matrix with shape Nx(N-1)x2.
    Returns
    -------
    np.ndarray
        The mask of the medoid elements for all clusters.
    """
    # Find the indexes of the representatives for each unique cluster.
    # noinspection PyUnresolvedReferences
    clusters = np.unique(cluster_labels)
    representative_mask = np.zeros(len(cluster_labels), np.bool)
    for i, cluster_i in enumerate(clusters):
        cluster_idx = np.flatnonzero(cluster_labels == cluster_i)
        
        if cluster_i == -1: # noise
            representative_mask[cluster_idx] = True
        else:
            if len(cluster_idx) <= 2: # identical pw_dist
                representative_mask[cluster_idx[0]] = True
            else:
                representative_mask[int(np.argmin(pw_dist[cluster_idx, :].mean(axis=1)))] = True

                # TODO: Support for condense distance matrix
                
        
    return representative_mask
