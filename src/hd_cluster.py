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


ANOMALY_EPS_PERCENTILE = 100

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


  # print(type(bucket_slice[1]), type(bucket_slice))
    if bucket_slice[1]-bucket_slice[0]==0:
        return [np.array([-1]), np.array([True])]
    else:
        bucket_slice[1] += 1
        bucket_hv = data_dict['hv'][bucket_slice[0]: bucket_slice[1]]
        bucket_prec_mz = data_dict['prec_mz'][bucket_slice[0]: bucket_slice[1]]
        bucket_rt_time = data_dict['rt_time'][bucket_slice[0]: bucket_slice[1]]
        
        pw_dist = fast_nb_cosine_dist_mask(bucket_hv, bucket_prec_mz, config.precursor_tol[0], output_type)
        export_distance_metric(pw_dist, "1302_distance_metrics.csv")


    # print("DISTANCE METRIC FORMAT", type(pw_dist),pw_dist)
        
       # sys.exit(0)
       # print("SIZE OF DIST METRIC", pw_dist.shape)
        
        cluster_func.fit(pw_dist) #
        cluster_func_labels = cluster_func.labels_

       
    
        
        # representative_mask = get_cluster_representative(
        #     cluster_labels=cluster_labels_refined, pw_dist=pw_dist) 
        
#recluster using kmeans
 
        # labels=cluster_func.labels_
        # n_cluster = len(set(labels)) - (1 if -1 in labels else 0)

    

        
        # if n_cluster>0 and n_cluster<100:

        #     recluster_func=cuml.KMeans(n_clusters=n_cluster,max_iter=100,output_type='numpy')
        #     recluster_func.fit(pw_dist) #
        # else:
        #     recluster_func=cluster_func 


       # print("FORMAT OF CLUSTER LABELS", type(cluster_func.labels_), cluster_func.labels_)
       
        
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

def weighted_percentile(values, weights, percentile):
    values = np.asarray(values)
    weights = np.asarray(weights)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cum_weights = np.cumsum(weights)
    cutoff = percentile / 100.0 * cum_weights[-1]

    return values[np.searchsorted(cum_weights, cutoff)]

def detect_bucket_anomaly(
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
    bucket_len = bucket_slice[1] - bucket_slice[0] + 1

    if bucket_len <= 0:
        # PARITY FIX: incr_2 branches on whether this bucket's precursor m/z
        # is beyond anything previously seen before deciding anomaly_mask.
        if data_dict["prec_mz"][bucket_slice[0]] > prev_meta_df["precursor_mz"].max():
            return [
                np.full(bucket_len, -1, dtype=int),
                np.ones(bucket_len, dtype=bool),
                [],
                np.ones(bucket_len, dtype=bool),
            ]
        else:
            return [
                np.full(bucket_len, -1, dtype=int),
                np.ones(bucket_len, dtype=bool),
                [],
                np.zeros(bucket_len, dtype=bool),
            ]

    bucket_slice = (bucket_slice[0], bucket_slice[1])
    bucket_hv = data_dict["hv"][bucket_slice[0]: bucket_slice[1] + 1]
    bucket_prec_mz = data_dict["prec_mz"][bucket_slice[0]: bucket_slice[1] + 1]

    neighbor_buckets = [bucket]

    idx_list = [
        get_bucket_indices(prev_meta_df, b)
        for b in neighbor_buckets
    ]

    pbucket_idx = np.concatenate(idx_list) if len(idx_list) > 0 else np.array([], dtype=int)
    pbucket_idx = np.unique(pbucket_idx)

    if len(pbucket_idx) == 0:
        output = cluster_bucket(
            bucket_slice=np.array(bucket_slice),
            data_dict=data_dict,
            config=config,
            cluster_func=cluster_func,
            output_type="cupy" if config.use_gpu_cluster else "numpy",
        )
        output.append([])
        output.append(np.zeros(len(bucket_hv), dtype=bool))
        return output

    bucket_prev_hv = prev_hvs[pbucket_idx]
    bucket_prev_prec_mz = prev_prec_mz[pbucket_idx]

    bucket_prev_meta_df = prev_meta_df.iloc[pbucket_idx].reset_index(drop=True)

    key_cols = ["bucket", "precursor_charge", "identifier", "scan", "retention_time"]

    bucket_clusters_raw = prev_clusters[
        prev_clusters["bucket"].isin(neighbor_buckets)
    ].copy()

    bucket_clusters = bucket_prev_meta_df[key_cols].merge(
        bucket_clusters_raw,
        on=key_cols,
        how="left"
    )

    assert len(bucket_clusters) == len(bucket_prev_hv)
    assert bucket_clusters[["identifier", "scan"]].reset_index(drop=True).equals(
        bucket_prev_meta_df[["identifier", "scan"]].reset_index(drop=True)
    )

    if "cluster" not in bucket_clusters.columns or bucket_clusters["cluster"].isna().all():
        output = cluster_bucket(
            bucket_slice=np.array(bucket_slice),
            data_dict=data_dict,
            config=config,
            cluster_func=cluster_func,
            output_type="cupy" if config.use_gpu_cluster else "numpy",
        )
        output.append([])
        output.append(np.zeros(len(bucket_hv), dtype=bool))
        return output

    # Row-order-safe validity mask (kept from your existing fix — this is
    # actually MORE robust than incr_2, which doesn't guard against NaN
    # rows from the merge at all. Not reverting this.)
    valid_mask = bucket_clusters["cluster"].notna().to_numpy()

    bucket_prev_hv = bucket_prev_hv[valid_mask]
    bucket_prev_prec_mz = bucket_prev_prec_mz[valid_mask]
    pbucket_idx = pbucket_idx[valid_mask]

    bucket_clusters = bucket_clusters[valid_mask].reset_index(drop=True)
    clusters = bucket_clusters["cluster"].to_numpy()

    if "is_representative" not in bucket_clusters.columns:
        bucket_clusters["is_representative"] = False

    cluster_rep_indices = bucket_clusters.index[
        bucket_clusters["is_representative"].fillna(False)
    ].to_numpy()

    # PARITY FIX: capture BEFORE the fill-loop invents stand-in reps.
    # incr_2 checks `len(cluster_reps) == 0` (the *initial* reps) and, if
    # true, bails out to a fresh cluster_bucket() recluster rather than
    # trusting randomly-chosen fill-in reps for incremental assignment.
    had_no_initial_reps = len(cluster_rep_indices) == 0

    # Ensure every cluster has at least one representative
    for c in np.unique(clusters):
        if c == -1:
            continue
        if len(cluster_rep_indices) == 0 or c not in clusters[cluster_rep_indices]:
            ids = np.where(clusters == c)[0]
            if len(ids) > 0:
                cluster_rep_indices = np.append(cluster_rep_indices, np.random.choice(ids))

    if had_no_initial_reps:
        output = cluster_bucket(
            bucket_slice=np.array(bucket_slice),
            data_dict=data_dict,
            config=config,
            cluster_func=cluster_func,
            output_type="cupy" if config.use_gpu_cluster else "numpy",
        )
        output.append([])
        output.append(np.zeros(len(bucket_hv), dtype=bool))
        return output

    if len(cluster_rep_indices) == 0:
        return [
            np.full(len(bucket_hv), -1, dtype=int),
            np.ones(len(bucket_hv), dtype=bool),
            [],
            np.zeros(len(bucket_hv), dtype=bool),
        ]

    rep_ids = clusters[cluster_rep_indices]
    cluster_rep_hvs = bucket_prev_hv[cluster_rep_indices]
    cluster_rep_mz = np.vstack(bucket_prev_prec_mz[cluster_rep_indices])

    uniq, counts = np.unique(clusters, return_counts=True)
    size_map = dict(zip(uniq, counts))
    cluster_rep_freqs = np.array([size_map[r] for r in rep_ids]) / len(bucket_prev_hv)

    full_hvs = np.concatenate((cluster_rep_hvs, bucket_hv), axis=0)
    full_prec_mz = np.concatenate([cluster_rep_mz, bucket_prec_mz])

    split_index = len(cluster_rep_hvs)

    pw_dist = fast_nb_cosine_dist_mask(
        full_hvs,
        full_prec_mz,
        config.precursor_tol[0],
        output_type
    )

    if config.use_gpu_cluster or output_type == "cupy":
        pw_dist = cp.asnumpy(pw_dist)

    dist_matrix = pw_dist[split_index:, :split_index]
    pw_dist_rep = pw_dist[:split_index, :split_index]

    # PARITY FIX (eps math): incr_2 computes a weighted percentile via
    # config.anomaly_eps_percentile but never actually uses it for the
    # real threshold — the real anomaly_eps comes from the UNWEIGHTED
    # percentile at a hardcoded 90th percentile. Matching that exactly.
    if pw_dist_rep.shape[0] > 1:
        d = pw_dist_rep.copy()
        np.fill_diagonal(d, np.inf)

        closest_dist_per_rep = d.min(axis=1)

        weights = np.asarray(cluster_rep_freqs, dtype=float)
        weights = weights / weights.sum()

        # Kept for parity/debugging only — NOT used for anomaly_eps,
        # exactly like incr_2's max_closest_dist (weighted) is computed
        # but not actually used for the threshold.
        max_closest_dist_weighted = weighted_percentile(
            closest_dist_per_rep, weights, config.anomaly_eps_percentile
        )

        # This is what incr_2 actually uses: unweighted, hardcoded 90th pct.
        max_closest_dist_unweighted = np.percentile(closest_dist_per_rep, config.anomaly_eps_percentile)

        anomaly_eps = min(max_closest_dist_unweighted + 0.01, 0.9)
    else:
        anomaly_eps = 0.9

    best_idx = np.argmin(dist_matrix, axis=1)
    best_dists = dist_matrix[np.arange(dist_matrix.shape[0]), best_idx]

    # PARITY FIX: incr_2 derives next_cluster_id from the LOCAL bucket's
    # `clusters` array, not the global prev_clusters history.
    unique_clusters = np.unique(clusters)
    next_cluster_id = max(unique_clusters[unique_clusters != -1], default=-1) + 1
    current_next_cluster_id = next_cluster_id

    is_anomaly = best_dists > anomaly_eps

    final_labels = np.where(is_anomaly, -1, rep_ids[best_idx])
    n_anomalies = is_anomaly.sum()
    anomaly_ids = current_next_cluster_id + np.arange(n_anomalies)
    final_labels[is_anomaly] = anomaly_ids

    representative_mask = is_anomaly.copy()
    anomaly_mask = is_anomaly.copy()

    if "anomaly" in prev_clusters.columns and (~is_anomaly).any():
        joined_idx = np.where(~is_anomaly)[0]
        bucket_rep_relative_index = cluster_rep_indices[best_idx[joined_idx]]
        original_prev_index = pbucket_idx[bucket_rep_relative_index]

        anomaly_col = prev_clusters["anomaly"].to_numpy()
        joined_anomaly = np.nan_to_num(anomaly_col[original_prev_index].astype(float), nan=0.0).astype(bool)
        if joined_anomaly.any():
            print(f"joining anomaly cluster x{joined_anomaly.sum()}")

    prev_rep_mask = prev_clusters["is_representative"].to_numpy()

    return [final_labels, representative_mask, prev_rep_mask, anomaly_mask]

def _fill_missing_reps_vectorized(clusters: np.ndarray, cluster_rep_indices: np.ndarray) -> np.ndarray:
    """
    Ensure every cluster label present in `clusters` has at least one
    representative index in `cluster_rep_indices`. Picks one row per
    missing cluster uniformly at random (same distribution as the
    original per-cluster `np.random.choice` loop).

    Replaces an O(num_unique_clusters * n) loop (a fresh `np.where(clusters==c)`
    scan PER cluster, even for clusters that already have a rep) with a single
    O(n log n) sort + a loop that only touches clusters actually missing a rep.
    """
    existing = (
        np.unique(clusters[cluster_rep_indices])
        if cluster_rep_indices.size else np.array([], dtype=clusters.dtype)
    )
    all_clusters = np.unique(clusters)
    missing = all_clusters[~np.isin(all_clusters, existing)]

    if missing.size == 0:
        return cluster_rep_indices

    order = np.argsort(clusters, kind="stable")
    sorted_clusters = clusters[order]
    starts = np.searchsorted(sorted_clusters, missing, side="left")
    ends = np.searchsorted(sorted_clusters, missing, side="right")

    # one random pick per missing cluster -- unavoidable python-level loop,
    # but now bounded by len(missing) instead of len(all_clusters)
    rand_offsets = np.array([np.random.randint(s, e) for s, e in zip(starts, ends)])
    picks = order[rand_offsets]

    return np.concatenate([cluster_rep_indices, picks])


def detect_bucket_anomaly_fast(
    bucket_slice: tuple,
    data_dict,
    prev_hvs,
    prev_clusters,
    config: Config,
    prev_prec_mz,
    output_type,
    prev_meta_df,
    bucket,
    cluster_func
):
    """
    Timing-optimized, vectorized rewrite of cluster_bucket_incr_2.
    Same inputs/outputs/return shape. See summary below the function
    for exactly what changed and what was deliberately left alone.
    """

    # --- Empty-slice case (unchanged) ---
    if bucket_slice[1] - bucket_slice[0] < 0:
        bucket_len = bucket_slice[1] - bucket_slice[0] + 1
        is_beyond_max = data_dict['prec_mz'][bucket_slice[0]] > (prev_meta_df['precursor_mz'].max())
        is_beyond_min= data_dict['prec_mz'][bucket_slice[0]] < (prev_meta_df['precursor_mz'].min())
        return [
            np.full(bucket_len, -1, dtype=int),
            np.ones(bucket_len, dtype=bool),
            [],
            np.full(bucket_len, is_beyond_max or is_beyond_min, dtype=bool),
        ]

    bucket_hv = data_dict['hv'][bucket_slice[0]: bucket_slice[1] + 1]
    bucket_prec_mz = data_dict['prec_mz'][bucket_slice[0]: bucket_slice[1] + 1]

    pbucket_idx = get_bucket_indices(prev_meta_df, bucket)

    # --- No previous data at all for this bucket ---
    # Equivalent to the original's post-fill-loop `len(cluster_rep_indices)==0`
    # check, reached earlier and without doing the merge first.
    if pbucket_idx.size == 0:
        is_beyond_max = data_dict['prec_mz'][bucket_slice[0]] > prev_meta_df['precursor_mz'].max()
        is_beyond_min= data_dict['prec_mz'][bucket_slice[0]] < (prev_meta_df['precursor_mz'].min())
        return [
            np.full(bucket_hv.shape[0], -1, dtype=int),
            np.ones(bucket_hv.shape[0], dtype=bool),
            [],
            np.full(bucket_hv.shape[0], is_beyond_max or is_beyond_min, dtype=bool),
        ]

    bucket_prev_hv = prev_hvs[pbucket_idx]
    bucket_prev_prec_mz = prev_prec_mz[pbucket_idx]
    bucket_prev_meta_df = prev_meta_df.iloc[pbucket_idx].reset_index(drop=True)

    key_cols = ["bucket", "precursor_charge", "identifier", "scan", "retention_time"]
    bucket_clusters_raw = prev_clusters[prev_clusters['bucket'] == bucket].reset_index(drop=True)
    # Guard against a one-to-many merge: if prev_clusters ever contains more
    # than one row for the same spectrum (duplicate key_cols), a left-merge
    # below silently duplicates rows, which desyncs bucket_clusters from
    # bucket_prev_hv / bucket_prev_prec_mz (same length as pbucket_idx) and
    # produces out-of-bounds indices later (e.g. cluster_rep_hvs = bucket_prev_hv[cluster_rep_indices]).
    bucket_clusters_raw = bucket_clusters_raw.drop_duplicates(subset=key_cols, keep="first")
    bucket_clusters = bucket_prev_meta_df[key_cols].merge(bucket_clusters_raw, on=key_cols, how="left")
    assert len(bucket_clusters) == len(bucket_prev_hv), (
        f"bucket_clusters merge produced {len(bucket_clusters)} rows but "
        f"bucket_prev_hv has {len(bucket_prev_hv)} rows (bucket={bucket}). "
        "This means prev_clusters has duplicate key_cols rows for this bucket."
    )

    prev_rep_mask = prev_clusters['is_representative'].to_numpy()

    cluster_reps = bucket_clusters.loc[bucket_clusters['is_representative'], 'cluster'].to_numpy()
    cluster_rep_indices = bucket_clusters.index[bucket_clusters['is_representative']].to_numpy()
    clusters = bucket_clusters['cluster'].to_numpy()

    # --- No originally-flagged representatives -> fall back to fresh clustering ---
    # This collapses the original's two identical-action early-return blocks
    # (the redundant "both None" check was a strict subset of the "or" check).
    if len(cluster_reps) == 0:
        output = cluster_bucket(
            bucket_slice=np.array(bucket_slice),
            data_dict=data_dict,
            config=config,
            cluster_func=cluster_func,
            output_type='cupy' if config.use_gpu_cluster else 'numpy')
        output.append([])
        output.append(np.zeros(bucket_hv.shape[0], dtype=bool))
        return output

    # --- Ensure every cluster has a rep (vectorized) ---
    cluster_rep_indices = _fill_missing_reps_vectorized(clusters, cluster_rep_indices)

    rep_ids = clusters[cluster_rep_indices]
    cluster_rep_hvs = bucket_prev_hv[cluster_rep_indices]
    cluster_rep_mz = np.vstack(bucket_prev_prec_mz[cluster_rep_indices])

    full_hvs = np.concatenate((cluster_rep_hvs, bucket_hv), axis=0)
    full_prec_mz = np.concatenate([cluster_rep_mz, bucket_prec_mz])
    split_index = len(cluster_rep_hvs)

    # Single distance computation instead of two (reps-only + reps+bucket).
    pw_dist = fast_nb_cosine_dist_mask(full_hvs, full_prec_mz, config.precursor_tol[0], output_type)
    if config.use_gpu_cluster or output_type == "cupy":
        pw_dist = cp.asnumpy(pw_dist)

    pw_dist_rep = pw_dist[:split_index, :split_index]
    dist_matrix = pw_dist[split_index:, :split_index]
    if pw_dist_rep.shape[0] > 1:
        d = pw_dist_rep.copy()
        np.fill_diagonal(d, np.inf)
        closest_dist_per_rep = d.min(axis=1)
        anomaly_eps = min(np.percentile(closest_dist_per_rep, 90) + 0.01, 0.9)
    else:
        anomaly_eps = 0.9

    best_idx = np.argmin(dist_matrix, axis=1)
    best_dists = dist_matrix[np.arange(dist_matrix.shape[0]), best_idx]

    unique_clusters = np.unique(clusters)
    next_cluster_id = max(unique_clusters[unique_clusters != -1], default=-1) + 1

    is_new = best_dists > anomaly_eps
    final_labels = np.where(is_new, -1, rep_ids[best_idx])
    n_new = int(is_new.sum())
    final_labels[is_new] = next_cluster_id + np.arange(n_new)
    
    representative_mask = is_new.copy()
    anomaly_mask = is_new.copy()

    return [final_labels, representative_mask, prev_rep_mask, anomaly_mask]

def detect_bucket_anomaly_alternative(
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
    # print("detect bucket anomaly function")
    bucket_len = bucket_slice[1] - bucket_slice[0] + 1

    if bucket_len <= 0:
        return [
            np.full(bucket_len, -1, dtype=int),
            np.ones(bucket_len, dtype=bool),
            [],
            np.ones(bucket_len, dtype=bool)
        ]

    # Current bucket only
    bucket_slice = (bucket_slice[0], bucket_slice[1])
    bucket_hv = data_dict["hv"][bucket_slice[0]: bucket_slice[1] + 1]
    bucket_prec_mz = data_dict["prec_mz"][bucket_slice[0]: bucket_slice[1] + 1]
    # bucket_rt_time = data_dict["rt_time"][bucket_slice[0]: bucket_slice[1] + 1]

    # metadata_df = data_dict["meta_data"]
    # metadata_df = metadata_df[metadata_df["bucket"] == bucket].reset_index(drop=True)

    # Previous adjacent buckets
    neighbor_buckets = [bucket]

    idx_list = [
        get_bucket_indices(prev_meta_df, b)
        for b in neighbor_buckets
    ]

    pbucket_idx = np.concatenate(idx_list) if len(idx_list) > 0 else np.array([], dtype=int)
    pbucket_idx = np.unique(pbucket_idx)

    # No previous spectra in this bucket or adjacent buckets
    if len(pbucket_idx) == 0:
        output = cluster_bucket(
            bucket_slice=np.array(bucket_slice),
            data_dict=data_dict,
            config=config,
            cluster_func=cluster_func,
            output_type="cupy" if config.use_gpu_cluster else "numpy",
        )
        output.append([])
        output.append(np.zeros(len(bucket_hv), dtype=bool))
        return output

    bucket_prev_hv = prev_hvs[pbucket_idx]
    bucket_prev_prec_mz = prev_prec_mz[pbucket_idx]
    # bucket_prev_meta_df = prev_meta_df.iloc[pbucket_idx].reset_index(drop=True)

    # bucket_clusters = prev_clusters[
    #     prev_clusters["bucket"].isin(neighbor_buckets)
    # ].reset_index(drop=True)


    bucket_prev_meta_df = prev_meta_df.iloc[pbucket_idx].reset_index(drop=True)

    key_cols = ["bucket", "precursor_charge", "identifier", "scan", "retention_time"]

    bucket_clusters_raw = prev_clusters[
        prev_clusters["bucket"].isin(neighbor_buckets)
    ].copy()

    bucket_clusters = bucket_prev_meta_df[key_cols].merge(
        bucket_clusters_raw,
        on=key_cols,
        how="left"
    )

    assert len(bucket_clusters) == len(bucket_prev_hv)
    assert bucket_clusters[["identifier", "scan"]].reset_index(drop=True).equals(
        bucket_prev_meta_df[["identifier", "scan"]].reset_index(drop=True)
)

    # key_cols = ["bucket", "precursor_charge", "identifier", "scan", "retention_time"]
    # print("checking alignment")
    # print(bucket_prev_meta_df.head())
    # print(bucket_clusters.head())
    
    # bucket_clusters = (
    #     bucket_prev_meta_df[key_cols]
    #     .merge(bucket_clusters, on=key_cols, how="left")
    # )

    # If merge failed or no cluster info, fallback to normal clustering
    if "cluster" not in bucket_clusters.columns or bucket_clusters["cluster"].isna().all():
        output = cluster_bucket(
            bucket_slice=np.array(bucket_slice),
            data_dict=data_dict,
            config=config,
            cluster_func=cluster_func,
            output_type="cupy" if config.use_gpu_cluster else "numpy",
        )
        output.append([])
        output.append(np.zeros(len(bucket_hv), dtype=bool))
        return output

    # --- FIX: compute the valid-row mask BEFORE reset_index, so it lines up
    # positionally with bucket_prev_hv / bucket_prev_prec_mz / pbucket_idx,
    # which all still use the pre-dropna row order. The previous code computed
    # bucket_clusters.index.to_numpy() *after* reset_index(drop=True), which
    # is always [0..n-1] and silently misaligns these arrays whenever any
    # row had a NaN cluster from the merge. ---
    valid_mask = bucket_clusters["cluster"].notna().to_numpy()

    bucket_prev_hv = bucket_prev_hv[valid_mask]
    bucket_prev_prec_mz = bucket_prev_prec_mz[valid_mask]
    pbucket_idx = pbucket_idx[valid_mask]

    bucket_clusters = bucket_clusters[valid_mask].reset_index(drop=True)
    clusters = bucket_clusters["cluster"].to_numpy()

    if "is_representative" not in bucket_clusters.columns:
        bucket_clusters["is_representative"] = False

    cluster_rep_indices = bucket_clusters.index[
        bucket_clusters["is_representative"].fillna(False)
    ].to_numpy()

    # Ensure every cluster has at least one representative
    for c in np.unique(clusters):
        if c == -1:
            continue
        if len(cluster_rep_indices) == 0 or c not in clusters[cluster_rep_indices]:
            ids = np.where(clusters == c)[0]
            if len(ids) > 0:
                cluster_rep_indices = np.append(cluster_rep_indices, np.random.choice(ids))

    if len(cluster_rep_indices) == 0:
        return [
            np.full(len(bucket_hv), -1, dtype=int),
            np.ones(len(bucket_hv), dtype=bool),
            [],
            np.zeros(len(bucket_hv), dtype=bool),
        ]

    rep_ids = clusters[cluster_rep_indices]
    cluster_rep_hvs = bucket_prev_hv[cluster_rep_indices]
    cluster_rep_mz = np.vstack(bucket_prev_prec_mz[cluster_rep_indices])

    # cluster_sizes = pd.Series(clusters).value_counts()
    # cluster_rep_freqs = cluster_sizes.loc[rep_ids].to_numpy() / len(bucket_prev_hv)


    uniq, counts = np.unique(clusters, return_counts=True)
    size_map = dict(zip(uniq, counts))
    cluster_rep_freqs = np.array([size_map[r] for r in rep_ids]) / len(bucket_prev_hv)

    # Adaptive threshold from previous reps
    # pw_dist_rep = fast_nb_cosine_dist_mask(
    #     cluster_rep_hvs,
    #     cluster_rep_mz,
    #     config.precursor_tol[0],
    #     output_type
    # )

    # if config.use_gpu_cluster or output_type == "cupy":
    #     pw_dist_rep = cp.asnumpy(pw_dist_rep)

    # if pw_dist_rep.shape[0] > 1:
    #     d = pw_dist_rep.copy()
    #     np.fill_diagonal(d, np.inf)

    #     closest_dist_per_rep = d.min(axis=1)

    #     weights = np.asarray(cluster_rep_freqs, dtype=float)
    #     weights = weights / weights.sum()

    #     max_closest_dist = weighted_percentile(
    #         closest_dist_per_rep,
    #         weights,
    #         ANOMALY_EPS_PERCENTILE
    #     )

    #     anomaly_eps = min(max_closest_dist + 0.01, 0.9)
    # else:
    #     anomaly_eps = 0.9
    # if pw_dist_rep.shape[0] > 1:
    #     d = pw_dist_rep.copy()
    #     np.fill_diagonal(d, np.inf)

    #     closest_dist_per_rep = d.min(axis=1)

    #     weights = np.asarray(cluster_rep_freqs, dtype=float)
    #     weights = weights / weights.sum()

    #     # --- Robust adaptive threshold (replaces fixed weighted percentile) ---
    #     # Weighted median as the robust center
    #     med = weighted_percentile(closest_dist_per_rep, weights, 50)

    #     # Weighted MAD around that center
    #     abs_dev = np.abs(closest_dist_per_rep - med)
    #     mad = weighted_percentile(abs_dev, weights, 50)
    #     # consistency constant so MAD approximates std under normality
    #     mad_scaled = mad * 1.4826

    #     k = ANOMALY_EPS_MAD_K  # new config knob, try 2.5–3.5
    #     max_closest_dist = med + k * mad_scaled

    #     anomaly_eps = min(max_closest_dist + 0.01, 0.9)
    # else:
    #     anomaly_eps = 0.9

    # Compare current bucket spectra against reps from adjacent previous buckets
    full_hvs = np.concatenate((cluster_rep_hvs, bucket_hv), axis=0)
    full_prec_mz = np.concatenate([cluster_rep_mz, bucket_prec_mz])

    split_index = len(cluster_rep_hvs)

    pw_dist = fast_nb_cosine_dist_mask(
        full_hvs,
        full_prec_mz,
        config.precursor_tol[0],
        output_type
    )

    if config.use_gpu_cluster or output_type == "cupy":
        pw_dist = cp.asnumpy(pw_dist)

    dist_matrix = pw_dist[split_index:, :split_index]
    pw_dist_rep = pw_dist[:split_index, :split_index]

    if pw_dist_rep.shape[0] > 1:
        d = pw_dist_rep.copy()
        np.fill_diagonal(d, np.inf)

        closest_dist_per_rep = d.min(axis=1)

        weights = np.asarray(cluster_rep_freqs, dtype=float)
        weights = weights / weights.sum()

        max_closest_dist = weighted_percentile(
            closest_dist_per_rep,
            weights,
            config.anomaly_eps_percentile
        )
        
        max_closest_dist_unweighted = np.percentile(closest_dist_per_rep, config.anomaly_eps_percentile)#weighted_percentile(
        #     closest_dist_per_rep,
        #     weights,
        #     config.anomaly_eps_percentile
        # )
# max_closest_dist + 0.01,
        anomaly_eps = min(0.9,  max_closest_dist_unweighted + 0.01)
    else:
        anomaly_eps = 0.9

    best_idx = np.argmin(dist_matrix, axis=1)
    best_dists = dist_matrix[np.arange(dist_matrix.shape[0]), best_idx]

    unique_clusters = np.unique(prev_clusters["cluster"].to_numpy())
    next_cluster_id = max(unique_clusters[unique_clusters != -1], default=-1) + 1
    current_next_cluster_id = next_cluster_id

    final_labels = np.full(len(bucket_hv), -1, dtype=int)
    representative_mask = np.zeros(len(bucket_hv), dtype=bool)
    anomaly_mask = np.zeros(len(bucket_hv), dtype=bool)

    # for j, (best_cluster_idx, dist) in enumerate(zip(best_idx, best_dists)):
    #     if dist <= anomaly_eps:
    #         final_labels[j] = rep_ids[best_cluster_idx]

    #         bucket_rep_relative_index = cluster_rep_indices[best_cluster_idx]
    #         original_prev_index = pbucket_idx[bucket_rep_relative_index]

    #         if "anomaly" in prev_clusters.columns:
    #             if bool(prev_clusters.iloc[original_prev_index]["anomaly"]):
    #                 print("joining anomaly cluster")
    #     else:
    #         final_labels[j] = current_next_cluster_id
    #         representative_mask[j] = True
    #         anomaly_mask[j] = True
    #         current_next_cluster_id += 1
    is_anomaly = best_dists > anomaly_eps

    final_labels = np.where(is_anomaly, -1, rep_ids[best_idx])  # placeholder for anomaly rows, filled below
    n_anomalies = is_anomaly.sum()
    anomaly_ids = current_next_cluster_id + np.arange(n_anomalies)
    final_labels[is_anomaly] = anomaly_ids

    representative_mask = is_anomaly.copy()
    anomaly_mask = is_anomaly.copy()

    # Optional: debug print for joining an existing anomaly cluster
    if "anomaly" in prev_clusters.columns and (~is_anomaly).any():
        joined_idx = np.where(~is_anomaly)[0]
        bucket_rep_relative_index = cluster_rep_indices[best_idx[joined_idx]]
        original_prev_index = pbucket_idx[bucket_rep_relative_index]
       # joined_anomaly = prev_clusters.iloc[original_prev_index]["anomaly"].fillna(False).to_numpy().astype(bool)

        anomaly_col = prev_clusters["anomaly"].to_numpy()
        joined_anomaly = np.nan_to_num(anomaly_col[original_prev_index].astype(float), nan=0.0).astype(bool)
        if joined_anomaly.any():
            print(f"joining anomaly cluster x{joined_anomaly.sum()}")

    prev_rep_mask = prev_clusters["is_representative"].to_numpy()

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


    data_dict = { 
        'hv': encoded_spectra_hv,
        'prec_mz': np.vstack(spectra_by_charge_df.precursor_mz).astype(np.float32),
        'rt_time': np.vstack(spectra_by_charge_df.retention_time).astype(np.float32),
        'meta_data':spectra_by_charge_df}

    ## Start clustering in GPU or CPU #

  
    bucket_idx_dict = schedule_bucket(spectra_by_charge_df, logger)


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
            #detect_bucket_anomaly
            #cluster_bucket_incr
            cluster_results = [detect_bucket_anomaly_fast(
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




    tot = 0
 

    spectra_by_charge_df
    prev_cluster_labels = np.array(prev_cluster_results['cluster'])
    prev_rep_mask = np.array(prev_cluster_results['is_representative'])
    prev_anomaly_mask = np.array(prev_cluster_results['anomaly'])
    print("checking prev_cluster_results in cluster_spec_incr")
 
    cluster_results = [cluster_results[i] for i in bucket_idx_dict['reorder_idx']]



    cluster_labels = [res_i[0] for res_i in cluster_results]
    cluster_labels = assign_unique_cluster_labels(cluster_labels)
    cluster_labels = np.hstack(cluster_labels)

    representative_mask = np.hstack([res_i[1] for res_i in cluster_results])
    anomaly_mask = np.hstack([res_i[3] for res_i in cluster_results])
    print("anomaly mask length", len(anomaly_mask))
    print("rep mask length", len(representative_mask))
    
    
    rep_clusters_new = set(cluster_labels[representative_mask])
    mask_to_update = np.isin(prev_cluster_labels, list(rep_clusters_new))
    prev_rep_mask[mask_to_update] = False
    


    logger.info("{} clustering in {:.4f} s".format(cluster_device, time.time()-start))


    if (config.incre_mode):
        print("prev_cluster size", len(prev_cluster_labels), "new clusters label size", len(cluster_labels))
    
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
            key_cols = ["bucket", "precursor_charge", "identifier", "scan", "retention_time"]

            prev_clusters = (
                prev_meta_df[key_cols]
                .merge(prev_clusters, on=key_cols, how="left")
            )

            cluster_results = [detect_bucket_anomaly(
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

