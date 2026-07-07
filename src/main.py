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

            
         
      # print("spectra hvs", spectra_hvs[0:3])
       # print("TYPE", spectra_hvs.dtype)
       # print("meta data", spectra_meta_df)
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
    


       # print("spectra_meta-df", spectra_meta_df)
    else:
        
        ############### 0. load previously saved checkpoint files 
        spectra_meta_df, spectra_hvs, prev_spectra_meta_df, prev_spectra_hvs  = None, None, None, None
        if config.checkpoint:
            prev_spectra_meta_df, prev_spectra_hvs = hd_preprocess.load_checkpoint(
                config=config, logger=logger)

             
            #load clustering results
            cluster_results = hd_preprocess.load_clustering_result(config=config, logger=logger)

            #load cluster representatives
            cluster_representatives = hd_preprocess.load_clustering_rep(config=config, logger=logger)
            
            #create datastructure prevResults combing meta_data, hypervectors, and cluster results
            prevResults = hd_preprocess.StaticClusterResults(prev_spectra_meta_df, prev_spectra_hvs, cluster_results)

        #    #get cluster representatives for the bucket 
         #  cluster_reps_898 = hd_preprocess.get_all_cluster_reps(prevResults, 898)
          #  print("PRINTING CLUSTER REPRESENTATIVES FOR BUCKET 898")
           # print(cluster_reps_898[:10])

            df = pd.read_csv('comm.txt', sep=" ",        # split on spaces
                 header=None,    # no header row in file
                 index_col=False)

            df = df.drop(columns=[0])        # drop first column
            df = df.rename(columns={1: "cluster"})
            print(df.columns)


            print(df.tail())
            print("NUMBER OF ROWS in comm.txt", len(df))

                  
            print("TESTING create dataset")
            
          # hd_preprocess.create_test_data(0.6, prev_spectra_meta_df, prev_spectra_hvs, "60_40_dataset")
            meta_60_40_train, hvs_60_40_train  = hd_preprocess.load_datasets('60_40_dataset', 'train', logger=logger)
            meta_60_40_test, hvs_60_40_test  = hd_preprocess.load_datasets('60_40_dataset', 'test', logger=logger)
   
            full_dataset_hvs = np.concatenate((hvs_60_40_train, hvs_60_40_test), axis = 0)
            full_dataset_meta = pd.concat([meta_60_40_train, meta_60_40_test])

            
            print("train SIZE", len(meta_60_40_train))
            print("test SIZE",  len(meta_60_40_test))
            print("full SIZE",  len(full_dataset_meta))
            
            print("exporting hypervectors for 60_40_full, 60_40_train, 60_40_test")
            hd_cluster_original.export_sample_hvs(full_dataset_hvs,  "60_40_full")
            hd_cluster_original.export_sample_hvs( hvs_60_40_train,  "60_40_train")
            hd_cluster_original.export_sample_hvs(hvs_60_40_test ,  "60_40_test")
            print("exporting distance metrics for 60_40_full, 60_40_train, 60_40_test")
            hd_cluster_original.sample_distance(hvs_60_40_train,  meta_60_40_train , config, "60_40_train_distance_metric.csv")  
            hd_cluster_original.sample_distance(hvs_60_40_test,  meta_60_40_test , config, "60_40_test_distance_metric.csv")
            hd_cluster_original.sample_distance(full_dataset_hvs,  full_dataset_meta , config, "60_40_full_distance_metric.csv")

            

           # cluster_df  =  pd.concat([meta_60_40_train, df], axis=1)
          # print(cluster_df.tail())
          # hd_preprocess.save_checkpoint(full_dataset_meta, full_dataset_hvs, config=config, logger=logger)
        #hd_cluster.export_sample_hvs(hvs_60_40_test, "hvs_60_40_train.csv") 
           # hd_preprocess.export_cluster_results(
            #   spectra_df=cluster_df,cluster_reps =  pd.DataFrame() , config=config, logger=logger)
             
            
          #  precursor_mz_val = meta_60_40_train['precursor_mz'].values
        
           # hd_cluster.export_sample_hvs(hvs_60_40_train, "hvs_60-40_train")
            #hd_cluster.export_sample_hvs(hvs_60_40_test,  "hvs_60-40_test")
            #hd_cluster.sample_distance(hvs_60_40_train, precursor_mz_val, config)
            

            

            print("PRINTING 60-40 dataset")
            print(meta_60_40_train.head())
            print(hvs_60_40_train[:10])
          # hd_preprocess.create_test_data(0.7, prev_spectra_meta_df, prev_spectra_hvs, "70_30_dataset")
           # hd_preprocess.create_test_data(0.9, prev_spectra_meta_df, prev_spectra_hvs, "90_10_dataset")
           #hd_preprocess.create_test_data(0.95, prev_spectra_meta_df, prev_spectra_hvs, "95_5_dataset")
          #  hd_preprocess.create_test_data(0.8, prev_spectra_meta_df, prev_spectra_hvs, "80_20_dataset")
            print("META DATA")
            print(prev_spectra_meta_df.head())
            print("CLUSTER RESULTS")
            print(cluster_results.head())
            print("ALL CLUSTER REPS")
            print(cluster_representatives.head())

            print("printing results of cluster 20")
            meta_subset, hvs_subset, cluster_subset = prevResults.get_bucket_data(898)
            print(meta_subset.head())
            print(hvs_subset[:10])
            print(cluster_subset.head())

            #create dataset for bucket
            print("CHECKING DATASIZE", hvs_subset.shape[0],  hvs_60_40_test.shape[0] + hvs_60_40_train.shape[0]) 
            hd_preprocess.create_test_data(0.6, meta_subset,hvs_subset,"60-40_dataset")
      
            print("SAMPLE HV DISTANCE")
            sample_hvs = hvs_subset
            bucket_mz = meta_subset
            precursor_mz_val = meta_60_40_train['precursor_mz'].values
            #eta_subset.iloc[0]['precursor_mz']
          #  hd_cluster.export_sample_hvs(sample_hvs) 
            hd_cluster_original.sample_distance(sample_hvs[0:10], precursor_mz_val, config, "test_distance_metric.csv")

            print(len(prev_spectra_meta_df), len(prev_spectra_hvs), len(cluster_results))
           # print("printing results of bucket 598")
           # meta_subset, hvs_subset, cluster_subset = prevResults.get_bucket_data(598)
           # print(meta_subset.head())
          #  print(hvs_subset[:10])
           # print(cluster_subset.head())
       #     print("PRINTING TEST REPRESENTATIVE THROUGH BINDING for cluster 20")
        #   rep = hd_preprocess.get_representative_binding(hvs_subset)
         #   rep2 = hd_preprocess.bundle(hvs_subset)
          #  print(len(hvs_subset), len(rep))
          #  print(rep)
           # print("ORIGINAL HVS shape: ", hvs_subset.shape)
            #print("REP SHAPE: ", rep.shape)
           # print("BUNDLED REP", rep2)
            
         #  cluster_reps = hd_preprocess.retreive_cluster_representatives(prevResults, 898)
         #   print("PRINTING CLUSTER REPRESENTATIVES FOR BUCKEY 898")
        #    print(cluster_reps.columns)
        #    print(cluster_reps)
         #   cluster_reps = hd_preprocess.get_all_cluster_reps(prevResults)
         #   hd_preprocess.export_cluster_results(
         #   spectra_df=cluster_results,cluster_reps = cluster_reps, config=config, logger=logger)
          #  for cluster in len(cluster_results.unique):
           #     meta_subset, hvs_subset, cluster_subset = prevResults.get_cluster_data(cluster)
            #    binding(hv_subset, cluster_subset)
            #make data structure out of checkpoint and parquet allows retrive bucket/cluster 
            #convert back to parquet 
            #hdf5 
            # make a class for retrieving data, input in hvs, metadata, cluste 
        
            ###################### 1. Load and parse spectra files
            spectra_meta_df, spectra_mz, spectra_intensity = hd_preprocess.load_process_spectra_parallel(config=config, logger=logger)
            logger.info("Preserve {} spectra for cluster charges: {}".format(len(spectra_meta_df), config.cluster_charges))
            
            ###################### 2 HD Encoding for spectra
            spectra_hvs = hd_cluster_original.encode_spectra(
                spectra_mz=spectra_mz, spectra_intensity=spectra_intensity, config=config, logger=logger)

          # print("using incremental")
           
            if(prev_spectra_meta_df is not None and prev_spectra_meta_df is not None):
                spectra_meta_df = spectra_meta_df = pd.concat([prev_spectra_meta_df, spectra_meta_df], ignore_index=True)
                
                spectra_hvs = np.vstack([prev_spectra_hvs, spectra_hvs]) 

            # Save meta and encoding data
            if config.checkpoint:
                hd_preprocess.save_checkpoint(
                    spectra_meta=spectra_meta_df, spectra_hvs=spectra_hvs, 
                    config=config, logger=logger) # maybe save spectra_mz too 
        
  
    ###################### 3. Cluster for each charge

    print(spectra_meta_df['bucket'].value_counts())

    cluster_df = pd.DataFrame()
    all_cluster_reps = pd.DataFrame()

    #removing identifiers for cluster 

    # excluded_proteins = pd.read_csv("excluded_spectra-3.csv")
    # print("excluded",len(excluded_proteins))
    # blength = len(spectra_meta_df)
    # match_cols = ["identifier", "scan"]
    # spectra_meta_df["identifier"] = spectra_meta_df["identifier"].astype(str).str.strip()
    # excluded_proteins["identifier"] = excluded_proteins["identifier"].astype(str).str.strip()

    # spectra_meta_df["scan"] = spectra_meta_df["scan"].astype(int)
    # excluded_proteins["scan"] = excluded_proteins["scan"].astype(int)

    # spectra_meta_df["retention_time"] = spectra_meta_df["retention_time"].astype(float).round(4)
    # excluded_proteins["retention_time"] = excluded_proteins["retention_time"].astype(float).round(4)
    # print(excluded_proteins[match_cols].dtypes, spectra_meta_df[match_cols].dtypes)
    # spectra_meta_df = spectra_meta_df.merge(excluded_proteins[match_cols].drop_duplicates(), on=match_cols, how="left", indicator=True)
    # keep_mask = spectra_meta_df["_merge"] == "left_only"
    # spectra_meta_df = spectra_meta_df[spectra_meta_df["_merge"] == "left_only"].drop(columns="_merge")
    # alength  = len(spectra_meta_df)
    # print("excluding proteins", blength, alength)
    # plength = len(spectra_hvs)
    # spectra_hvs = spectra_hvs[keep_mask.to_numpy(), :]
    # print("hyper-vector length afte exlucidng proteins",plength,len(spectra_hvs))
    # print("meta_data length after excluding proteins", blength,  len(spectra_meta_df))
    #mask = (spectra_meta_df['identifier'] == 'b1927_293T_proteinID_07A_QE3_122212') & (spectra_meta_df['scan'] == 14919)  & (spectra_meta_df['retention_time'] == 3348.160156)
   # print(spectra_meta_df[mask])
    


   # meta_60_40_train, hvs_60_40_train  = hd_preprocess.load_datasets('60_40_dataset', 'train', logger=logger)
  # hd_preprocess.create_test_data(0.6, spectra_meta_df, spectra_hvs, "60_40_dataset")
    #sys.exit(0)
#   spectra_hvs = hvs_60_40_train
 #   spectra_meta_df =  meta_60_40_train

 # Get the hypervectors for these two specific rows
    # mask1 = (
    #     (spectra_meta_df['bucket'] == 807) &
    #     (spectra_meta_df['scan'] == 30561) &
    #     (spectra_meta_df['identifier'] == 'b1929_293T_proteinID_09A_QE3_122212')
    # )

    # mask2 = (
    #     (spectra_meta_df['bucket'] == 807) &
    #     (spectra_meta_df['scan'] == 30848) &
    #     (spectra_meta_df['identifier'] == 'b1929_293T_proteinID_09A_QE3_122212')
    # )

    # idx1 = spectra_meta_df.index[mask1][0]
    # idx2 = spectra_meta_df.index[mask2][0]

    # hv1 = spectra_hvs[idx1]
    # hv2 = spectra_hvs[idx2]

    # print("Row 1 (scan 30561, rep):", hv1)
    # print("Row 2 (scan 30848):", hv2)

    # # Also compute their mutual distance
    # from hd_cluster import fast_nb_cosine_dist_mask
    # import numpy as np

    # both = np.stack([hv1, hv2])
    # mzs = np.array([404.72342, 404.72458], dtype=np.float32).reshape(-1,1)
    # dist = fast_nb_cosine_dist_mask(both, mzs, 20, 'numpy')  # adjust precursor_tol as needed
    # print("Distance between them:", dist[0,1])
    print("bucket debug stats")
    print(spectra_meta_df["bucket"].value_counts().head(10))
    print(spectra_meta_df["bucket"].value_counts().describe())
   
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
       # cluster_reps = hd_cluster.get_all_cluster_reps(spec_df_by_charge.reset_index(drop=True), spectra_hvs[idx])
       # print("printing cluster reps")
        #print(cluster_reps.head())
      
        cluster_df = pd.concat([cluster_df, spec_df_by_charge])
        all_cluster_reps = pd.concat([all_cluster_reps, cluster_reps])
        
 #   staticClusterResults = hd_preprocess.StaticClusterResults(spectra_meta_df, spectra_hvs, cluster_df)
  #  cluster_reps = hd_preprocess.get_all_cluster_reps(staticClusterResults)
  # print(cluster_reps.head())
    print(cluster_df.head())
    print("cluster_df", cluster_df)
    clusters_p, count_p = np.unique(cluster_df['cluster'].to_numpy(), return_counts=True)
    print(len(cluster_df), len(clusters_p))
    hd_preprocess.export_cluster_results(
        spectra_df=cluster_df,cluster_reps =all_cluster_reps, config=config, logger=logger)
    # cluster_df.to_csv("cluster_results_main.csv", index=False)    
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

