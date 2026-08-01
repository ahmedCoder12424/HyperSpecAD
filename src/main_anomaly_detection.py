import sys, gc, logging
gc.enable()

import numpy as np
from typing import Union, List
from config import * 

import tqdm
import pandas as pd

import hd_preprocess, hd_cluster

logger = logging.getLogger('HyperSpecAD')
import time

def count_true_anomalies(spectra_meta_df, spectra_hvs):
    # excluded_proteins = pd.read_csv("../excluded_spectra-3.csv")
    print("nexcluded",len(excluded_proteins))
    blength = len(spectra_meta_df)
    match_cols = ["identifier", "scan"]

    spectra_meta_df["identifier"] = spectra_meta_df["identifier"].astype(str).str.strip()
    excluded_proteins["identifier"] = excluded_proteins["identifier"].astype(str).str.strip()
    spectra_meta_df["scan"] = spectra_meta_df["scan"].astype(int)
    excluded_proteins["scan"] = excluded_proteins["scan"].astype(int)
    spectra_meta_df["retention_time"] = spectra_meta_df["retention_time"].astype(float).round(4)
    excluded_proteins["retention_time"] = excluded_proteins["retention_time"].astype(float).round(4)
    
    tmp = spectra_meta_df.merge(excluded_proteins[match_cols].drop_duplicates(), on=match_cols, how="left", indicator=True)
    excluded_mask = tmp["_merge"].eq("both").to_numpy()
       # keep_mask = spectra_meta_df["_merge"] == "both"

        # filter meta
    spectra_meta_df_excluded = spectra_meta_df.loc[excluded_mask].copy()
    print("number of true anomalies in results", len(spectra_meta_df_excluded))


def include_specified_spectras(spectra_meta_df, spectra_hvs, filename):
    spectras = pd.read_csv(filename)
    blength = len(spectra_meta_df)
    match_cols = ["identifier", "scan"]


    spectra_meta_df["identifier"] = spectra_meta_df["identifier"].astype(str).str.strip()
    spectras["identifier"] = spectras["identifier"].astype(str).str.strip()
    spectra_meta_df["scan"] = spectra_meta_df["scan"].astype(int)
    spectras["scan"] = spectras["scan"].astype(int)
    spectra_meta_df["retention_time"] = spectra_meta_df["retention_time"].astype(float).round(4)
    spectras["retention_time"] = spectras["retention_time"].astype(float).round(4)

    match_cols = ["identifier", "scan"]

    spectras_dedup = (
        spectras[match_cols + ["cluster"]]
        .sort_values(match_cols)
        .drop_duplicates(subset=match_cols, keep="first")
    )

        #spectras_dedup,
        #spectras[match_cols + ["cluster"]].drop_duplicates()
    tmp = spectra_meta_df.merge(  spectras_dedup, on=match_cols, how="left", indicator=True)
    excluded_mask = tmp["_merge"].eq("both").to_numpy()
       # keep_mask = spectra_meta_df["_merge"] == "both"

        # filter meta
    spectra_meta_df_excluded = spectra_meta_df.loc[excluded_mask].copy()
    spectra_meta_df_excluded["cluster"] = tmp.loc[excluded_mask, "cluster"].values

        # filter hypervectors
    spectra_hvs_excluded = spectra_hvs[excluded_mask, :]



    spectra_meta_df = spectra_meta_df_excluded
    spectra_hvs = spectra_hvs_excluded
    return spectra_meta_df, spectra_hvs


def sort_data(meta_data, hvs):
    order = np.argsort(meta_data['bucket'].to_numpy())
    meta_data = meta_data.iloc[order].reset_index(drop=True)
    if hvs is not None:
        hvs = hvs[order]
    return meta_data, hvs


#need to add anomaly column to reclustering results 
def recluster(spectra_meta_df, spectra_hvs, cluster_results):
    config.incre_mode = False
    cluster_df = pd.DataFrame()
    for prec_charge_i in tqdm.tqdm(config.cluster_charges):
        # Select spectra with cluster charge
        idx = spectra_meta_df['precursor_charge']==prec_charge_i
        spec_df_by_charge = spectra_meta_df.loc[idx]

        logger.info("Start clustering Charge {} with {} spectra".format(prec_charge_i, len(spec_df_by_charge)))
        if(len(spec_df_by_charge) == 0):
            continue 
        cluster_labels_per_charge, cluster_representatives_per_charge = hd_cluster.cluster_spectra(
            spectra_by_charge_df=spec_df_by_charge, encoded_spectra_hv=spectra_hvs[idx],
            config=config, logger=logger)

        spec_df_by_charge = spec_df_by_charge.assign(
            cluster=list(cluster_labels_per_charge), 
            is_representative=list(cluster_representatives_per_charge))

        cluster_df = pd.concat([cluster_df, spec_df_by_charge])


    #add anomaly mask from cluster_results 

    match_cols = ['identifier', 'scan']   # add more columns if needed

    anomaly_lookup = cluster_results[match_cols + ['anomaly']].drop_duplicates()

    cluster_df = cluster_df.merge(
        anomaly_lookup,
        on=match_cols,
        how='left'
    )

 
    cluster_df['anomaly'] = cluster_df['anomaly'].fillna(False)
    match_cols = ['identifier', 'scan']  # adjust if needed

    merged = cluster_df.merge(
        cluster_results[match_cols + ['cluster']],
        on=match_cols,
        how='inner',
        suffixes=('_new', '_old')
    )

    num_different = (merged['cluster_new'] != merged['cluster_old']).sum()

    print("Number of clusters that changed:", num_different)
    return cluster_df


# @profile
def main(args: Union[str, List[str]] = None) -> int:
    total_start = time.perf_counter()
    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : '
        '{message}', style='{'))
    root.addHandler(handler)
    
    # Disable dependency non-critical log messages.
    logging.getLogger('numpy').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('cupy').setLevel(logging.WARNING)
    logging.getLogger('joblib').setLevel(logging.WARNING)

    # Load the configuration.
    config.parse(args)
   # logger.debug('input_filepath= %s', config.input_filepath)
    logger.debug('input_filepath= %s', config.input_paths)
    logger.debug('mz_interval= %s', config.mz_interval)
    print("config.input_paths", config.input_paths)
    # print("config.input_filepath", config.input_filepath)
    # logger.debug('work_dir = %s', config.work_dir)
    # logger.debug('overwrite = %s', config.overwrite)
    logger.debug('checkpoint = %s', config.checkpoint)
    logger.debug('representative_mgf = %s', config.representative_mgf)
    logger.debug('cpu_core_preprocess = %s', config.cpu_core_preprocess)
    logger.debug('cpu_core_cluster = %s', config.cpu_core_cluster)
    logger.debug('batch_size = %d', config.batch_size)
    logger.debug('use_gpu_cluster = %s', config.use_gpu_cluster)

    logger.debug('min_peaks = %d', config.min_peaks)
    logger.debug('min_mz_range = %.2f', config.min_mz_range)
    logger.debug('min_mz = %.2f', config.min_mz)
    logger.debug('max_mz = %.2f', config.max_mz)
    logger.debug('remove_precursor_tol = %.2f', config.remove_precursor_tol)
    logger.debug('min_intensity = %.2f', config.min_intensity)
    logger.debug('max_peaks_used = %d', config.max_peaks_used)
    logger.debug('scaling = %s', config.scaling)

    logger.debug('hd_dim = %d', config.hd_dim)
    logger.debug('hd_Q = %d', config.hd_Q)
    logger.debug('hd_id_flip_factor = %.1f', config.hd_id_flip_factor)
    logger.debug('cluster_charges = %s', config.cluster_charges)

    logger.debug('precursor_tol = %.2f %s', *config.precursor_tol)
    logger.debug('rt_tol = %s', config.rt_tol)
    logger.debug('cluster_alg = %s', config.cluster_alg)
    logger.debug('fragment_tol = %.2f', config.fragment_tol)
    logger.debug('eps = %.3f', config.eps)

    #anomaly detection config options
    logger.debug('init_data=%s',config.init_data)
    logger.debug('static_cluster_file=%s',config.static_cluster_file)
    logger.debug('incr_data=%s',config.incr_data)
    logger.debug('incre_eps=%.3f',config.incre_eps)
    logger.debug('incre_ratio=%.3f',config.incre_ratio)
    logger.debug('incre_batch_size=%d',config.incre_batch_size)
    logger.debug('incre_mode=%s',config.incre_mode)
    logger.debug('anomaly_file=%s', config.anomaly_file)
    logger.debug('anomaly_path=%s', config.anomaly_path)
    logger.debug('anomaly_eps_percentile=%s', config.anomaly_eps_percentile)
    logger.debug('input_paths=%s', config.input_paths)

    # Restore checkpoints
    if(not config.incre_mode):
        print("CLUSTERING OF NON-ANOMALY SET")
        spectra_meta_df, spectra_hvs = None, None
        if config.checkpoint:
            spectra_meta_df, spectra_hvs = hd_preprocess.load_checkpoint(
                config=config, logger=logger)

        if (spectra_meta_df is None) or (spectra_hvs is None):
            ###################### 1. Load and parse spectra files
            spectra_meta_df, spectra_mz, spectra_intensity = hd_preprocess.load_process_spectra_parallel(config=config, logger=logger)
            logger.info("Preserve {} spectra for cluster charges: {}".format(len(spectra_meta_df), config.cluster_charges))
        
            ###################### 2 HD Encoding for spectra
            spectra_hvs = hd_cluster.encode_spectra(
                spectra_mz=spectra_mz, spectra_intensity=spectra_intensity, config=config, logger=logger)

            # Save meta and encoding data
            if config.checkpoint:
                hd_preprocess.save_checkpoint(
                    spectra_meta=spectra_meta_df, spectra_hvs=spectra_hvs, 
                    config=config, logger=logger)

       
        # print(len(spectra_meta_df))


        cluster_df = pd.DataFrame()
      
     
        #creating train and test data from the original data for anomaly detection
        spectra_meta_df_init, spectra_hvs_init = include_specified_spectras(spectra_meta_df, spectra_hvs, config.anomaly_path + "/train_data_"+ config.anomaly_file+ ".csv")
    
        spectra_meta_df_incr, spectra_hvs_incr = include_specified_spectras(spectra_meta_df, spectra_hvs, config.anomaly_path + "/test_data_mixed_" + config.anomaly_file+ ".csv")
  
        # print(config.anomaly_path + "/test_data_mixed_" + config.anomaly_file+ ".csv")
     
        spectra_meta_df_init['precursor_charge'] = 1
        spectra_meta_df_incr['precursor_charge'] = 1

        #sorting data for speed, can cause script to be killed if not sorted
        spectra_meta_df_init, spectra_hvs_init = sort_data(spectra_meta_df_init, spectra_hvs_init)
        spectra_meta_df_incr, spectra_hvs_incr = sort_data(spectra_meta_df_incr, spectra_hvs_incr)

        #cluster the initial data without anomalies 
        spectra_meta_df = spectra_meta_df_init
        spectra_meta_df["hv_idx"] = np.arange(len(spectra_meta_df))
        spectra_hvs = spectra_hvs_init
        spectra_meta_df, spectra_hvs = sort_data(spectra_meta_df, spectra_hvs)
        for prec_charge_i in tqdm.tqdm(config.cluster_charges):
            # Select spectra with cluster charge
            idx = spectra_meta_df['precursor_charge']==prec_charge_i
            spec_df_by_charge = spectra_meta_df.loc[idx]

            logger.info("Start clustering Charge {} with {} spectra".format(prec_charge_i, len(spec_df_by_charge)))
            if(len(spec_df_by_charge) == 0):
                continue 
            cluster_labels_per_charge, cluster_representatives_per_charge = hd_cluster.cluster_spectra(
                spectra_by_charge_df=spec_df_by_charge, encoded_spectra_hv=spectra_hvs[idx],
                config=config, logger=logger)

            spec_df_by_charge = spec_df_by_charge.assign(
                cluster=list(cluster_labels_per_charge), 
                is_representative=list(cluster_representatives_per_charge))
        
            cluster_df = pd.concat([cluster_df, spec_df_by_charge])
      
        
        #setting anomaly column to false for all initial data, since we know they are not anomalies
        cluster_df['anomaly'] = False
    
        prev_spectra_hvs = spectra_hvs_init
        prev_spectra_meta_df = spectra_meta_df_init
        cluster_results = cluster_df

        #split the test data mixed with anomalies into batches for detection of anomalies to simulate time-series data
        batch_size = 10000
        batches = []

        n = len(spectra_hvs_incr)
        print("NUM BATCHES", batch_size)
        for i in range(0,n, batch_size):
            end = min(i + batch_size, n)
            hvs_batch = spectra_hvs_incr[i:end,:]
            meta_batch = spectra_meta_df_incr.iloc[i:end]
            batches.append((meta_batch, hvs_batch))

        print("ANOMALY DETECION OF MIXED ANOMALY/NON_ANOMALY SAMPLES")
        anomaly_df = pd.DataFrame()
        batch_count = 0
        total_start_anomaly_detection = time.perf_counter()
        for batch in batches:
            
            config.incre_mode = True
            spectra_meta_df = batch[0]
            spectra_hvs = batch[1]
            metadata_df =  spectra_meta_df 
            spectra_meta_df, spectra_hvs = sort_data(spectra_meta_df, spectra_hvs)

            prev_spectra_meta_df, prev_spectra_hvs = sort_data(prev_spectra_meta_df, prev_spectra_hvs)

            batch_count+=1
            batch_cluster_df = pd.DataFrame()
            batch_anomaly_df = pd.DataFrame()
            
            for prec_charge_i in tqdm.tqdm(config.cluster_charges):
                # Select spectra with cluster charge
                idx = spectra_meta_df['precursor_charge']==prec_charge_i
                spec_df_by_charge = spectra_meta_df.loc[idx]

                prev_idx = prev_spectra_meta_df['precursor_charge']==prec_charge_i
                prev_spec_df_by_charge = prev_spectra_meta_df.loc[prev_idx]
                logger.info("Start clustering Charge {} with {} spectra".format(prec_charge_i, len(spec_df_by_charge)))

                prev_cluster_results = cluster_results.loc[
                    cluster_results['precursor_charge'] == prec_charge_i
                ].copy()
      

                if(len(spec_df_by_charge) == 0):
                    continue
                
                config.incre_mode = True 
                cluster_labels_per_charge, cluster_representatives_per_charge, anomaly_mask, cluster_labels_new = hd_cluster.cluster_spectra_incr(
                    spectra_by_charge_df=spec_df_by_charge, encoded_spectra_hv=spectra_hvs[idx], prev_spectra_by_charge_df = prev_spec_df_by_charge,
                    prev_encoded_spectra_hv=prev_spectra_hvs[prev_idx], prev_cluster_results=prev_cluster_results,
                    config=config, logger=logger)
              
                spec_df_by_charge = spec_df_by_charge.assign(
                    cluster=list(cluster_labels_per_charge),
                    is_representative=list(cluster_representatives_per_charge),
                    anomaly=list(anomaly_mask))
               
                
                anomaly_by_charge = spec_df_by_charge.assign(cluster=list(cluster_labels_new), anomaly=list(anomaly_mask))
                anomaly_by_charge = anomaly_by_charge[anomaly_by_charge["anomaly"] == True]
     
                batch_cluster_df = pd.concat([batch_cluster_df, spec_df_by_charge])
                prev_length= len(batch_anomaly_df)
                batch_anomaly_df = pd.concat([batch_anomaly_df, anomaly_by_charge])
                metadata_df = batch_anomaly_df
    
            prev_length = len(anomaly_df)
            prev_clus_length = len(cluster_results)
            cluster_results = pd.concat([cluster_results, batch_cluster_df])
            anomaly_df = cluster_results[cluster_results["anomaly"] == True].copy()

            prev_spectra_meta_df = pd.concat([prev_spectra_meta_df, spectra_meta_df], ignore_index=True)
            prev_spectra_hvs =  np.vstack([prev_spectra_hvs, spectra_hvs]) 
           
    hd_preprocess.export_cluster_results(
        spectra_df=cluster_results, config=config, logger=logger)
    
    #export cluster and anomaly results to csv
    cluster_results.to_csv("cluster_result/cluster_results_"+config.anomaly_file+".csv", index=False)    
    if (anomaly_df is not None):
        num_unique_clusters = anomaly_df["cluster"].nunique()

        hd_preprocess.export_anomaly_results(
        spectra_df=anomaly_df, filename="cluster_result/anomaly_results_"+config.anomaly_file, logger=logger)

    else:
        pass
    anomaly_detection_total_runtime = time.perf_counter() - total_start_anomaly_detection
    total_runtime = time.perf_counter() - total_start
    samples_per_second = float(len(spectra_meta_df_incr))/anomaly_detection_total_runtime
  
    print(f"[TIME] total main_anomaly_detection.py runtime: {total_runtime:.3f} seconds")
    print(f"[TIME] total anomaly detection phase runtime: {anomaly_detection_total_runtime:.3f} seconds")
    print(f"[TIME] anomaly detection samples per second: {samples_per_second:.3f}")

    #export timing results to csv 
    with open("anomaly_validation_results/timing_results_"+config.anomaly_file+".csv", "a") as f:
        f.write(f"{config.anomaly_file},{total_runtime:.3f},{anomaly_detection_total_runtime:.3f},{samples_per_second:.3f}\n")

if __name__ == "__main__":
    main()

