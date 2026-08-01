import sys, gc, logging
gc.enable()

from typing import Union, List
from config import * 

import tqdm
import pandas as pd
import numpy as np
import hd_preprocess, hd_cluster_original

logger = logging.getLogger('HyperSpec')

# @profile
def main(args: Union[str, List[str]] = None) -> int:
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
    # print("config.input_filepath",  config.input_filepath)
    logger.debug('input_filepath= %s', config.input_paths)
    print("config.input_filepath",  config.input_paths)
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
    # logger.debug('use_incremental_clustering = %s', config.incremental)
    config_incremental = False
    if(not config_incremental):
        # Restore checkpoints
 
        spectra_meta_df, spectra_hvs = None, None
       # spectra_meta_df, spectra_hvs  = hd_preprocess.load_datasets('60_40_dataset', 'train', logger=logger)
        testing_portion = True
        if config.checkpoint and testing_portion:
            spectra_meta_df, spectra_hvs = hd_preprocess.load_checkpoint(
                config=config, logger=logger)

            
         
        if testing_portion and (spectra_meta_df is None) or (spectra_hvs is None):
            ###################### 1. Load and parse spectra files
            spectra_meta_df, spectra_mz, spectra_intensity = hd_preprocess.load_process_spectra_parallel(config=config, logger=logger)

            print("spectra_meta_df after preprocessing")
            print(spectra_meta_df)
            print("spectra_mz")
            print(spectra_mz)
            print("spectra_intensity")
            print(spectra_intensity)
            logger.info("Preserve {} spectra for cluster charges: {}".format(len(spectra_meta_df), config.cluster_charges))
        
            ###################### 2 HD Encoding for spectra
            spectra_hvs = hd_cluster_original.encode_spectra(
                spectra_mz=spectra_mz, spectra_intensity=spectra_intensity, config=config, logger=logger)

            print("spectra hvs after encoding")
            print(spectra_hvs)
    
        #   print("spectra hvs", spectra_hvs[0:3])
            # Save meta and encoding data
            if config.checkpoint:
                hd_preprocess.save_checkpoint(
                    spectra_meta=spectra_meta_df, spectra_hvs=spectra_hvs, 
                    config=config, logger=logger)

            print(spectra_meta_df[spectra_meta_df['identifier'].str.contains("injected")])
            print("spectra-meta_df", spectra_meta_df)
            spectra_meta_df.loc[spectra_meta_df['identifier'].str.contains("anomalies", na=False), 'precursor_charge'] = 1
            print(spectra_meta_df[spectra_meta_df['identifier'].str.contains("anomalies")])
    

  
    ###################### 3. Cluster for each charge

    print(spectra_meta_df['bucket'].value_counts())

    cluster_df = pd.DataFrame()
    all_cluster_reps = pd.DataFrame()

    for prec_charge_i in tqdm.tqdm(config.cluster_charges):
        # Select spectra with cluster charge
        idx = spectra_meta_df['precursor_charge']==prec_charge_i

        count = idx.sum()
        print(f"Charge {prec_charge_i}: {count} spectra")
        if count < 5:
            print(f"Skipping charge {prec_charge_i} (too few spectra)")
            continue
        spec_df_by_charge = spectra_meta_df.loc[idx]

        logger.info("Start clustering Charge {} with {} spectra".format(prec_charge_i, len(spec_df_by_charge)))
        
        cluster_labels_per_charge,cluster_reps, cluster_representatives_per_charge = hd_cluster_original.cluster_spectra(
            spectra_by_charge_df=spec_df_by_charge, encoded_spectra_hv=spectra_hvs[idx],
            config=config, logger=logger)

        spec_df_by_charge = spec_df_by_charge.assign(
            cluster=list(cluster_labels_per_charge), 
            is_representative=list(cluster_representatives_per_charge))
        print(spec_df_by_charge.head())
        print(spec_df_by_charge.index)
        print(len(spec_df_by_charge), spectra_hvs[idx].shape[0])
 
        cluster_df = pd.concat([cluster_df, spec_df_by_charge])
        all_cluster_reps = pd.concat([all_cluster_reps, cluster_reps])
        

    print(cluster_df.head())
    print("cluster_df", cluster_df)
    clusters_p, count_p = np.unique(cluster_df['cluster'].to_numpy(), return_counts=True)
    print(len(cluster_df), len(clusters_p))
   
    cluster_df.to_csv(config.output_filename, index=False)

    print("number of unique clusters", len(cluster_df['cluster'].unique()))
    cluster_sizes = cluster_df['cluster'].value_counts()

    # Calculate min, max, median, and print the metrics
    print("number of unique clusters", len(cluster_df['cluster'].unique()))
    print("number of unique  buckets", len(cluster_df['bucket'].unique()))
    print("Max cluster size:", cluster_sizes.max())
    print("Min cluster size:", cluster_sizes.min())
    print("Median cluster size:", cluster_sizes.median())

if __name__ == "__main__":
    main()

