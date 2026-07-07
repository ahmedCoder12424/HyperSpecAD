import sys, gc, logging
gc.enable()

import numpy as np
from typing import Union, List
from config import * 


import tqdm
import pandas as pd

import hd_preprocess, hd_cluster


logger = logging.getLogger('HyperSpec')
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





def exclude_proteins(spectra_meta_df, spectra_hvs):

    excluded_proteins = pd.read_csv("excluded_spectra-3.csv")
    print("excluded",len(excluded_proteins))
    blength = len(spectra_meta_df)
    match_cols = ["identifier", "scan"]
    spectra_meta_df["identifier"] = spectra_meta_df["identifier"].astype(str).str.strip()
    excluded_proteins["identifier"] = excluded_proteins["identifier"].astype(str).str.strip()

    spectra_meta_df["scan"] = spectra_meta_df["scan"].astype(int)
    excluded_proteins["scan"] = excluded_proteins["scan"].astype(int)

    spectra_meta_df["retention_time"] = spectra_meta_df["retention_time"].astype(float).round(4)
    excluded_proteins["retention_time"] = excluded_proteins["retention_time"].astype(float).round(4)
    print(excluded_proteins[match_cols].dtypes, spectra_meta_df[match_cols].dtypes)
    spectra_meta_df = spectra_meta_df.merge(excluded_proteins[match_cols].drop_duplicates(), on=match_cols, how="left", indicator=True)
    keep_mask = spectra_meta_df["_merge"] == "left_only"
    print("EXCLUDED IDX range",spectra_meta_df.index[~keep_mask].to_numpy())
    spectra_meta_df = spectra_meta_df[spectra_meta_df["_merg_e"] == "left_only"].drop(columns="_merge")
    alength  = len(spectra_meta_df)
    print("excluding proteins", blength, alength)
    plength = len(spectra_hvs)
    spectra_hvs = spectra_hvs[keep_mask.to_numpy(), :]
    print("hyper-vector length afte exlucidng proteins",plength,len(spectra_hvs))
    print("meta_data length after excluding proteins", blength,  len(spectra_meta_df))
    

    # spectra_meta_df = spectra_meta_df_excluded
    # spectra_hvs = spectra_hvs_excluded
    return spectra_meta_df, spectra_hvs

def include_only_proteins(spectra_meta_df, spectra_hvs, config):
    #excluded_proteins = pd.read_csv("excluded_spectra_smaller.csv")
    excluded_proteins = pd.read_csv(config.anomaly_path + "/anomaly_spectra_"+config.anomaly_file+".csv")
    # excluded_proteins = pd.read_csv("anomaly_var_size_files_pct/anomaly_spectra_s3.csv")
    print("NUMBER OF ANOMALIES IN ANOMALY FILE",len(excluded_proteins))
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

        # filter hypervectors
    spectra_hvs_excluded = spectra_hvs[excluded_mask, :]

    print(" EXCLUDEDD FROM WRONG DATASETexcluded meta rows:", len(spectra_meta_df_excluded))
    print("excluded hypervectors:", len(spectra_hvs_excluded))

    spectra_meta_df = spectra_meta_df_excluded
    spectra_hvs = spectra_hvs_excluded
    return spectra_meta_df, spectra_hvs


def include_specified_spectras(spectra_meta_df, spectra_hvs, filename):
    spectras = pd.read_csv(filename)
    print("number of psectras",len(spectras))
    blength = len(spectra_meta_df)
    match_cols = ["identifier", "scan"]
    print('filename', filename)
    print("spectras", spectras.head())

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

    # print(" EXCLUDEDD FROM WRONG DATASETexcluded meta rows:", len(spectra_meta_df_excluded))
    print("excluded hypervectors:", len(spectra_hvs_excluded))

    spectra_meta_df = spectra_meta_df_excluded
    spectra_hvs = spectra_hvs_excluded
    return spectra_meta_df, spectra_hvs


def include_specified_spectras_2(spectra_meta_df, spectra_hvs, filename):
    spectras = pd.read_csv(filename)
    print("number of spectras", len(spectras))
    blength = len(spectra_meta_df)
    match_cols = ["identifier", "scan"]

    spectra_meta_df = spectra_meta_df.reset_index(drop=True)
    
    spectra_meta_df["identifier"] = spectra_meta_df["identifier"].astype(str).str.strip()
    spectras["identifier"] = spectras["identifier"].astype(str).str.strip()
    spectra_meta_df["scan"] = spectra_meta_df["scan"].astype(int)
    spectras["scan"] = spectras["scan"].astype(int)
    spectra_meta_df["retention_time"] = spectra_meta_df["retention_time"].astype(float).round(4)
    spectras["retention_time"] = spectras["retention_time"].astype(float).round(4)

    tmp = spectra_meta_df.merge(
        spectras[match_cols + ["cluster"]].drop_duplicates(subset=match_cols),
        on=match_cols,
        how="left",
        indicator=True
    )

    assert len(tmp) == len(spectra_meta_df), \
        f"Merge produced duplicate rows: {len(tmp)} vs {len(spectra_meta_df)}"

    excluded_mask = tmp["_merge"].eq("both").to_numpy()

    spectra_meta_df_excluded = spectra_meta_df.loc[excluded_mask].copy()
    spectra_meta_df_excluded["cluster"] = tmp.loc[excluded_mask, "cluster"].values

    spectra_hvs_excluded = spectra_hvs[excluded_mask, :]

    print(f"Matched {len(spectra_meta_df_excluded)} of {blength} spectra")
    print(f"Excluded hypervectors: {len(spectra_hvs_excluded)}")

    assert len(spectra_meta_df_excluded) == len(spectra_hvs_excluded), \
        f"Meta/HV mismatch after filtering: {len(spectra_meta_df_excluded)} vs {len(spectra_hvs_excluded)}"

    return spectra_meta_df_excluded, spectra_hvs_excluded

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

    clusters_p, count_p = np.unique(cluster_df['cluster'].to_numpy(), return_counts=True)


    #add anomaly mask from cluster_results 

    match_cols = ['identifier', 'scan']   # add more columns if needed

    anomaly_lookup = cluster_results[match_cols + ['anomaly']].drop_duplicates()

    cluster_df = cluster_df.merge(
        anomaly_lookup,
        on=match_cols,
        how='left'
    )

    clusters_p, count_p = np.unique(cluster_df['cluster'].to_numpy(), return_counts=True)
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

#incremental config
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



    print("CONFIG PRECURSOR TOL", config.precursor_tol[0])
    print(config.output_filename)

 

    

    
    # Restore checkpoints
    print("INCRE MODE", config.incre_mode)
    if(not config.incre_mode):
        spectra_meta_df, spectra_hvs = None, None
        if config.checkpoint:
            spectra_meta_df, spectra_hvs = hd_preprocess.load_checkpoint(
                config=config, logger=logger)
    #test comment
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

       
        print(len(spectra_meta_df))


        cluster_df = pd.DataFrame()
        # print("LENGTH OF HVS", len(spectra_hvs))
        # print("LENGTH OF DF", len(spectra_meta_df))


        # metadata_df2= pd.read_csv("1468_dataset_meta.csv")
        # spectra_hvs2 = np.load("spectra_hvs.npy")

        # print("are hypervecotrs equal ", np.array_equal(spectra_hvs,   spectra_hvs2))
        # print(metadata_df2)
        # print(spectra_meta_df)

        # rt1 = spectra_meta_df['scan'].astype(int).reset_index(drop=True)
        # rt2 = metadata_df2['scan'].astype(int).reset_index(drop=True)
        # print("does metadata equal", rt1.equals(rt2))
      

         #anomaly_cluster_data/train_data_
        # print("LOAIDING", "anomaly_size_files/train_data"+ config.anomaly_file+ ".csv")
        # print(len(spectra_meta_df), len(spectra_hvs))
       # spectra_meta_df_init, spectra_hvs_init = include_specified_spectras(spectra_meta_df, spectra_hvs,"anomaly_size_files/train_data"+ config.anomaly_file+ ".csv")
      #  spectra_meta_df_init, spectra_hvs_init = include_specified_spectras(spectra_meta_df, spectra_hvs,"inorganic_anomaly_split_files_mod_2/train_data"+ config.anomaly_file+ ".csv")
        # spectra_meta_df_init, spectra_hvs_init = include_specified_spectras(spectra_meta_df, spectra_hvs,"anomaly_var_size_files_inorganic_trunc/train_data_"+ config.anomaly_file+ ".csv")
        spectra_meta_df_init, spectra_hvs_init = include_specified_spectras(spectra_meta_df, spectra_hvs, config.anomaly_path + "/train_data_"+ config.anomaly_file+ ".csv")
       # spectra_meta_df_init, spectra_hvs_init = include_specified_spectras(spectra_meta_df, spectra_hvs,"sweep_anomaly_size/sweep_anom/train_datas2.csv")
        
        # print("length of train data", len(spectra_hvs_init))
        #anomaly_cluster_data/test_data-mixed_
        #spectra_meta_df_incr, spectra_hvs_incr = include_specified_spectras(spectra_meta_df, spectra_hvs,"anomaly_size_files/test_data-mixed" + config.anomaly_file+ ".csv")
        # print(len(spectra_meta_df), len(spectra_hvs))
        #spectra_meta_df_incr, spectra_hvs_incr = include_specified_spectras(spectra_meta_df, spectra_hvs,"inorganic_anomaly_split_files_mod_2/test_data-mixed"+ config.anomaly_file+ ".csv")
        # spectra_meta_df_incr, spectra_hvs_incr = include_specified_spectras(spectra_meta_df, spectra_hvs,"anomaly_var_size_files_inorganic_trunc/test_data_mixed_"+ config.anomaly_file+ ".csv")
        spectra_meta_df_incr, spectra_hvs_incr = include_specified_spectras(spectra_meta_df, spectra_hvs, config.anomaly_path + "/test_data_mixed_" + config.anomaly_file+ ".csv")
       # spectra_meta_df_incr, spectra_hvs_incr = include_specified_spectras(spectra_meta_df, spectra_hvs,"sweep_anomaly_size/sweep_anom/test_data-mixeds2.csv")
        # print("columns of incr_data",   spectra_meta_df_incr.columns)
        print(config.anomaly_path + "/test_data_mixed_" + config.anomaly_file+ ".csv")
     

        # print("length of incr data", len(spectra_hvs_incr), len(spectra_meta_df_incr))

        idx = np.where(spectra_meta_df_incr['cluster'] == 470192)[0]
        # print(spectra_hvs_incr[idx])


        spectra_meta_df_init['precursor_charge'] = 1
        spectra_meta_df_incr['precursor_charge'] = 1

    

        hvs_here = spectra_hvs_incr[idx]
        indexes_here = idx


        # cluster_results = pd.read_csv("cluster_results_main.csv")
        # metadata_df = pd.read_csv("1468_dataset_meta.csv")
        # spectra_hvs = np.load("spectra_hvs.npy")


        # print(cluster_results['retention_time'][0:5])
        # print(metadata_df['retention_time'][0:5])



        # cluster_results[cluster_results['cluster']==495092]
        # idx = cluster_results[cluster_results['cluster'] == 495092].index
        # # print(spectra_hvs[idx])

        # index_sep = idx 
        # hvs_sep = spectra_hvs[idx]

        # print(index_sep,  indexes_here)
        # print(cluster_results.iloc[idx])
        # print(spectra_meta_df_incr.iloc[indexes_here])
        





        # print("checking if incr data includes anomaly spectras")
        spectra_meta_df_an, spectra_hvs_an = include_only_proteins(spectra_meta_df_incr, spectra_hvs_incr, config)
        print("NUMBER OF ANOMALIES IN TEST DATA", len(spectra_meta_df_an))
        print("NUMBER OF ANOMALIES using identifer", spectra_meta_df["identifier"].str.contains("anomal", na=False).sum())

        an_mz = spectra_meta_df_an['precursor_mz'].values
       
        # print(spectra_meta_df_an)






        # i = spectra_meta_df_init[spectra_meta_df_init['precursor_mz']==75.008766].index
        # mz = 75.008766
        # print(spectra_hvs_init[i])



        # print("checking distance to candidate init spectrum")

        # mz = 6.6102657

        # # Boolean mask over dataframe rows
        # mask = np.isclose(
        #     spectra_meta_df_init["precursor_mz"].to_numpy(dtype=np.float32),
        #     mz,
        #     atol=1e-6
        # )

        # # Convert mask to positional row index matching spectra_hvs_init
        # pos_matches = np.flatnonzero(mask)

        # # print("positional matches:", pos_matches)

        # candidate_pos = pos_matches[0]
        # candidate_hv = spectra_hvs_init[candidate_pos]
        # candidate_mz = spectra_meta_df_init.iloc[candidate_pos]["precursor_mz"]

        # anomaly_pos = 0
        # anomaly_hv = spectra_hvs_an[anomaly_pos]
        # anomaly_mz = spectra_meta_df_an.iloc[anomaly_pos]["precursor_mz"]
        # # print(spectra_meta_df_an.iloc[anomaly_pos])

        # both = np.stack([anomaly_hv, candidate_hv])
        # mzs = np.array([anomaly_mz, candidate_mz], dtype=np.float32).reshape(-1, 1)

        # dist = hd_cluster.fast_nb_cosine_dist_mask(both, mzs, 20, "numpy")

        # print("anomaly mz:", anomaly_mz)
        # print("candidate mz:", candidate_mz)
        # print("distance matrix:")
        # print(dist)
        # print("distance anomaly -> candidate:", dist[0, 1])

        # print("bucket 9335")
        # print(len(spectra_meta_df_incr[spectra_meta_df_incr['bucket']==9335]))
        # print(len(spectra_meta_df_init[spectra_meta_df_init['bucket']==9335]))

        # print("bucket 5750")
        # print(len(spectra_meta_df_incr[spectra_meta_df_incr['bucket']==5750]))
        # print(len(spectra_meta_df_init[spectra_meta_df_init['bucket']==5750]))



    


    






        # metadata_df =  spectra_meta_df_incr
     






        spectra_meta_df_init, spectra_hvs_init = sort_data(spectra_meta_df_init, spectra_hvs_init)
        spectra_meta_df_incr, spectra_hvs_incr = sort_data(spectra_meta_df_incr, spectra_hvs_incr)


#         mask = (
#             (spectra_meta_df_init["cluster"] == 28038) 
#         )

#         positions = np.flatnonzero(mask.to_numpy())

#         print("matches:", len(positions))

#         for pos in positions:
#             print("position:", pos)
#             print(spectra_meta_df_init.iloc[pos])
#             print("hypervector:")
#             print(spectra_hvs_init[pos])


#         mask2 = (
#             (spectra_meta_df_incr['bucket'] == 1906) &
#             (spectra_meta_df_incr['scan'] == 20020) &
#             (spectra_meta_df_incr['identifier'] == 'b1929_293T_proteinID_09A_QE3_122212')
#         )


#         idx2 = spectra_meta_df_incr.index[mask2][0]
#         # print(spectra_meta_df_incr.iloc[idx2])
#         hv2 = spectra_hvs_incr[idx2]
#         EXPECTED_HV = hv2
#         print("EXEPCTED HV" , hv2)



#         mask_anom = (
#     (spectra_meta_df_incr["bucket"] == 1027) &
#     (spectra_meta_df_incr["scan"] == 14980)
# )



#         anom_positions = np.flatnonzero(mask_anom.to_numpy())
#         # assert len(anom_positions) == 1, f"Expected 1 anomaly match, got {len(anom_positions)}"

#         anom_pos = anom_positions[0]
#         expected_hv = spectra_hvs_incr[anom_pos]
#         expected_mz = spectra_meta_df_incr.iloc[anom_pos]["precursor_mz"]

#         print("EXPECTED anomaly row:")
#         print(spectra_meta_df_incr.iloc[anom_pos])
#         print("EXPECTED HV:")
#         print(expected_hv)

        # # Compare against every spectrum in cluster 28038
        # mask_cluster = spectra_meta_df_init["cluster"] ==10891
        # cluster_positions = np.flatnonzero(mask_cluster.to_numpy())

        # print("cluster 28038 matches:", len(cluster_positions))
        # sanity_hv = spectra_hvs_init[cluster_positions[0]]
        # sanity_mz = spectra_meta_df_init.iloc[cluster_positions[0]]["precursor_mz"]
        # expected_hv=sanity_hv
        # expected_mz=sanity_mz
        # for pos in cluster_positions:
        #     candidate_hv = spectra_hvs_init[pos]
        #     candidate_mz = spectra_meta_df_init.iloc[pos]["precursor_mz"]

        #     both = np.stack([expected_hv, candidate_hv])
        #     mzs = np.array([expected_mz, candidate_mz], dtype=np.float32).reshape(-1, 1)

        #     dist = hd_cluster.fast_nb_cosine_dist_mask(both, mzs, 20, "numpy")

        #     print("\ncandidate position:", pos)
        #     print(spectra_meta_df_init.iloc[pos][
        #         ["bucket", "scan", "identifier", "precursor_mz", "retention_time", "cluster"]
        #     ])
        #     print("distance to expected anomaly:", dist[0, 1])
        #     print("candidate HV:")
        #     print(candidate_hv)


        # print(len(spectra_meta_df_init))
        # print(len(spectra_meta_df_incr))

  




        # spectra_meta_df_init= spectra_meta_df.iloc[0:int(0.3*len(spectra_meta_df))]
        # spectra_hvs_init = spectra_hvs[0:int(0.3*len(spectra_hvs)),:]
        # spectra_meta_df_excl = spectra_meta_df.drop(
        #  spectra_meta_df.index[30:100]
        # )
       
        # #make sure exluded proteins in incr
        # spectra_meta_df_incr = spectra_meta_df.drop(
        #  spectra_meta_df.index[0:int(0.3*len(spectra_meta_df))]
        # )
        # spectra_hvs_incr = np.delete(spectra_hvs, np.s_[0:int(0.3*len(spectra_meta_df))], axis=0)

        # spectra_meta_df_incr, spectra_hvs_incr = include_only_proteins(spectra_meta_df_incr, spectra_hvs_incr)


        # spectra_meta_df_init, spectra_hvs_init = exclude_proteins(spectra_meta_df_init, spectra_hvs_init)
        # print("after excluding proteins", len(spectra_meta_df), len(spectra_hvs))
        # hd_preprocess.save_data_ckp(spectra_meta_df_excl, spectra_hvs_excl, "excluded_proteins_dataset", config, logger)
        # spectra_meta_df_incl, spectra_hvs_incl = include_only_proteins(spectra_meta_df_init, hvs_init)
        # print("after_including proteins", len(spectra_meta_df_incl), len(spectra_hvs_incl))
        # hd_preprocess.save_data_ckp(spectra_meta_df_incl, spectra_hvs_incl, "initial_proteins_dataset", config, logger)

        #initial clustering 

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
      
        
        clusters_p, count_p = np.unique(cluster_df['cluster'].to_numpy(), return_counts=True)
        cluster_df['anomaly'] = False
        # print(cluster_df.head())
        hd_preprocess.export_cluster_results(
            spectra_df=cluster_df, config=config, logger=logger)

        prev_spectra_hvs = spectra_hvs_init
        prev_spectra_meta_df = spectra_meta_df_init
        cluster_results = cluster_df


#         mask_anom = (
#     (spectra_meta_df_incr["bucket"] == 1027) &
#     (spectra_meta_df_incr["scan"] == 14980)
# )

#         anom_positions = np.flatnonzero(mask_anom.to_numpy())
#         # assert len(anom_positions) == 1, f"Expected 1 anomaly match, got {len(anom_positions)}"

#         anom_pos = anom_positions[0]
#         expected_hv = spectra_hvs_incr[anom_pos]
#         expected_mz = spectra_meta_df_incr.iloc[anom_pos]["precursor_mz"]

#         print("EXPECTED anomaly row:")
#         print(spectra_meta_df_incr.iloc[anom_pos])
#         print("EXPECTED HV:")
#         print(expected_hv)

      
         
#         target_cluster = 34919

#         cluster_rows = cluster_df[cluster_df["cluster"] == target_cluster]
#         print(cluster_rows)
#         cluster_hv_indices = cluster_rows["hv_idx"].to_numpy()

#         print(f"cluster {target_cluster} matches:", len(cluster_hv_indices))

#         mask_anom = (
#     (spectra_meta_df_incr["bucket"] == 1478) &
#     (spectra_meta_df_incr["scan"] == 69638)
# )

#         anom_positions = np.flatnonzero(mask_anom.to_numpy())
#         # assert len(anom_positions) == 1, f"Expected 1 anomaly match, got {len(anom_positions)}"

#         anom_pos = anom_positions[0]
#         expected_hv = spectra_hvs_incr[anom_pos]
#         expected_mz = spectra_meta_df_incr.iloc[anom_pos]["precursor_mz"]

    

#         print("EXPECTED HV", expected_hv)
    

#         for hv_idx in cluster_hv_indices:
#             candidate_hv = spectra_hvs_init[hv_idx]
#             candidate_mz = spectra_meta_df_init.iloc[hv_idx]["precursor_mz"]

#             both = np.stack([expected_hv, candidate_hv])
#             mzs = np.array([expected_mz, candidate_mz], dtype=np.float32).reshape(-1, 1)

#             dist = hd_cluster.fast_nb_cosine_dist_mask(both, mzs, 20, "numpy")

#             print("\nhv_idx:", hv_idx)
#             print(spectra_meta_df_init.iloc[hv_idx][
#                 ["bucket", "scan", "identifier", "precursor_mz", "retention_time"]
#             ])
#             print("cluster:", target_cluster)
#             print("distance to expected anomaly:", dist[0, 1])
#             print("candidate HV:")
#             print(candidate_hv)




    
        # print("finished getting initial clusters")

        hd_preprocess.export_cluster_results(
         spectra_df=cluster_results, config=config, logger=logger)

        # print(len(cluster_results), len(spectra_meta_df_init))

        batch_size = 100000
        batches = []
#         print(len(
#     spectra_meta_df_incr[
#         (spectra_meta_df_incr['precursor_charge'] == 2) &
#         (spectra_meta_df_incr['precursor_mz'] == 625.80994) &
#         (spectra_meta_df_incr['scan'] == 67944)
#     ]
# ))      

        n = len(spectra_hvs_incr)



        
        for i in range(0,n, batch_size):
            end = min(i + batch_size, n)
            hvs_batch = spectra_hvs_incr[i:end,:]
            meta_batch = spectra_meta_df_incr.iloc[i:end]
            batches.append((meta_batch, hvs_batch))



        anomaly_df = pd.DataFrame()
   
        b = 0
        # print("NUM BATCHES ",len(batches))
        num_batches = len(batches)
        # print(int(num_batches/10))
        # sys.exit(0)
        for batch in batches:
            
            config.incre_mode = True
            spectra_meta_df = batch[0]
            spectra_hvs = batch[1]
            metadata_df =  spectra_meta_df 
            spectra_meta_df, spectra_hvs = sort_data(spectra_meta_df, spectra_hvs)

            prev_spectra_meta_df, prev_spectra_hvs = sort_data(prev_spectra_meta_df, prev_spectra_hvs)
            cluster_results_results, place = sort_data(cluster_results,None)

            
        #     mask2 = (
        #     (prev_spectra_meta_df['bucket'] == 1906) &
        #     (prev_spectra_meta_df['scan'] == 20020) &
        #     (prev_spectra_meta_df['identifier'] == 'b1929_293T_proteinID_09A_QE3_122212')
        # )

        #     if mask2.any():
        #         idx2 = prev_spectra_meta_df.index[mask2][0]
        #         hv2 = prev_spectra_hvs[idx2]
        #         # print("Row 2 (scan 30848):", hv2)


        #         expected_hv = EXPECTED_HV
        #         print("found spectra, ", np.array_equal(expected_hv,hv2))
              
        #         print("does spectra hv match", np.array_equal(expected_hv,hv2))
        #         if (np.array_equal(expected_hv,hv2)==False):
        #             print(hv2)
        #             print(expected_hv) 
        #             print("spectra hv does not match")
     
                
        #         if (np.array_equal(expected_hv,hv2)==True):
        #             print("spectra hv matches")
            
    
      

            # assert len(prev_spectra_meta_df) == len(prev_spectra_hvs), \
            #     f"MISALIGN: meta={len(prev_spectra_meta_df)} hvs={len(prev_spectra_hvs)}"

            # # Spot-check the specific spectrum we know is wrong
            # check = prev_spectra_meta_df[
            #     (prev_spectra_meta_df['scan'] == 30848) & 
            #     (prev_spectra_meta_df['identifier'] == 'b1929_293T_proteinID_09A_QE3_122212')
            # ]
            # if len(check) > 0:
            #     pos = check.index[0]  # positional after reset_index
            #     print(f"scan 30848 is at position {pos} in prev_spectra_meta_df")
            #     print(f"its HV: {prev_spectra_hvs[pos]}")
            #     print(f"expected HV: {spectra_hvs[...]}") 
      

            # print("anomaly count before reclusterng", len(cluster_results[cluster_results['anomaly']==True]))
            # cluster_results = recluster(prev_spectra_meta_df, prev_spectra_hvs, cluster_results)
            # print("anomaly count after reclusterng", len(cluster_results[cluster_results['anomaly']==True]))
            # if (len(cluster_results[cluster_results['anomaly']==True])>3):
            #     print("b", b)
            # #     sys.exit(0)
         
            # if b %int(num_batches/4) == 0 and b>0:
            #     print("RECLUSTERING", b)
            #     cluster_results = recluster(prev_spectra_meta_df, prev_spectra_hvs, cluster_results)
            #     print("anomaly count after reclusterng", len(cluster_results[cluster_results['anomaly']==True]))
            #     print("size of clusters", len(cluster_results))

                
                # sys.exit(0)


            b+=1
            # print("BATCH", b)
            # print("batch ", b ,"checking length CONSISTENCY", len(prev_spectra_hvs), len(prev_spectra_meta_df), len(cluster_df), len(cluster_results))
            j = 0
            batch_cluster_df = pd.DataFrame()
            batch_anomaly_df = pd.DataFrame()
            n = len(metadata_df[(metadata_df['cluster']== 18723)]) + len(metadata_df[(metadata_df['cluster']==28038)]) + len(metadata_df[(metadata_df['cluster']== 22736)])
            
            mask = spectra_meta_df["cluster"] == 511196
            if( mask.sum() >= 1):
                pass
                # print("FOUND 511196 rows in df:", mask.sum())
        

            for prec_charge_i in tqdm.tqdm(config.cluster_charges):
                # Select spectra with cluster charge
                
                idx = spectra_meta_df['precursor_charge']==prec_charge_i
                spec_df_by_charge = spectra_meta_df.loc[idx]
                    
                mask = spec_df_by_charge["cluster"] == 511196
                if( mask.sum() >= 1):
                    pass
                    # print("FOUND 511196 rows in spec_df_by_charge:", mask.sum())

        

                prev_idx = prev_spectra_meta_df['precursor_charge']==prec_charge_i
                prev_spec_df_by_charge = prev_spectra_meta_df.loc[prev_idx]
                logger.info("Start clustering Charge {} with {} spectra".format(prec_charge_i, len(spec_df_by_charge)))

                prev_cluster_results = cluster_results.loc[
                    cluster_results['precursor_charge'] == prec_charge_i
                ].copy()
                prev_anomaly = prev_cluster_results[prev_cluster_results['anomaly']==True]
                #before cluster_results[prev_idx]

            
                if(len(spec_df_by_charge) == 0):
                    # batch_cluster_df = pd.concat([batch_cluster_df, prev_cluster_results])
                    # batch_anomaly_df = pd.concat([batch_anomaly_df, prev_anomaly])
                    # print("appending remain cluster results", len(batch_cluster_df))
                    continue
                
                # print("checking length prev ", len(prev_spectra_hvs[prev_idx]),len(cluster_results[prev_idx]),len(prev_spec_df_by_charge))
                # print("checking length new ", len(spec_df_by_charge),len(spectra_hvs[idx]))

      
                # print( "27731 count" ,len(prev_cluster_results[prev_cluster_results['cluster']==27731]))
                config.incre_mode = True 
                cluster_labels_per_charge, cluster_representatives_per_charge, anomaly_mask, cluster_labels_new = hd_cluster.cluster_spectra_incr(
                    spectra_by_charge_df=spec_df_by_charge, encoded_spectra_hv=spectra_hvs[idx], prev_spectra_by_charge_df = prev_spec_df_by_charge,
                    prev_encoded_spectra_hv=prev_spectra_hvs[prev_idx], prev_cluster_results=prev_cluster_results,
                    config=config, logger=logger)
                #get this from cluster_spectra_incr 

                # print("len of input", len(spectra_hvs[idx]))
                # print("len of previous input", len(prev_spectra_hvs[prev_idx]))
                # print("len of prev input", len(prev_spec_df_by_charge))
                # print("len of cluster_results",cluster_results[prev_idx])

                # print("cluster_labels", len(cluster_labels_per_charge))
                # print("cluster reps", len(cluster_representatives_per_charge))
                # print("anomaly mask", (len(anomaly_mask)))
                # print("cluster_labels_new", len(cluster_labels_new))
    
    
                num_noise = np.sum(cluster_labels_per_charge == -1)
                # print("NUM NOISE", num_noise)
                # print("length of previous data", len(idx))
                # print("length of previous_cluster_results", len(cluster_results[prev_idx]))
                # print("lengt of cluster results from function", len(cluster_labels_per_charge), len(cluster_representatives_per_charge))
            
                # print("cluster_labels")
                # print(len(cluster_labels_per_charge))
                # print("cluster_reps_per_charge")
                # print(len(cluster_representatives_per_charge))
                # print("new stuff length", len(spec_df_by_charge))
                # spec_df_by_charge = pd.concat([prev_spec_df_by_charge, spec_df_by_charge], ignore_index=True)
                # spec_df_by_charge = pd.concat([prev_spec_df_by_charge, spec_df_by_charge])
                # print("length of concatted clustering table", len(spec_df_by_charge))

                # print("len(spec_df_by_charge)", len(spec_df_by_charge))
                # print("len(cluster_labels_new)", len(cluster_labels_new))
                # print(spec_df_by_charge.columns)

                # print("config mode ", config.incre_mode)
                # print(j,b)
                # print("prev cluster", len(prev_cluster_results))
                # print("new clusters",len(cluster_labels_new))
                # print("spec_df_by_charge", len(spec_df_by_charge))
                # print("prev_spec_df_by_charge", len(prev_spec_df_by_charge))
                j+=1
                # print("BATCH " , j, len( spec_df_by_charge), len(cluster_labels_per_charge))

            
                # if (b ==4):
                #     sys.exit(0)
                # print("cluster labels" ,len(cluster_labels_per_charge), "rep", len(cluster_representatives_per_charge), "anomaly", len(anomaly_mask))

                spec_df_by_charge = spec_df_by_charge.assign(
                    cluster=list(cluster_labels_per_charge),
                    is_representative=list(cluster_representatives_per_charge),
                    anomaly=list(anomaly_mask))
               
                # if (b ==4):
                #     print(spec_df_by_charge[(spec_df_by_charge['bucket']==1249) & (spec_df_by_charge['identifier']=='b1927_293T_proteinID_07A_QE3_122212')])
                #     sys.exit(0)
                    
             
                # spec_df_by_charge["cluster"] = cluster_labels_per_charge
                # spec_df_by_charge["is_representative"] = cluster_representatives_per_charge
                # spec_df_by_charge["anomaly"] = anomaly_mask
                
                anomaly_by_charge = spec_df_by_charge.assign(cluster=list(cluster_labels_new), anomaly=list(anomaly_mask))
                anomaly_by_charge = anomaly_by_charge[anomaly_by_charge["anomaly"] == True]
                # print("Anomaly ratio:", np.mean(anomaly_mask))
                # print("checking anomaly results")
                # print(anomaly_by_charge.head())
                # print(len(cluster_df),len(spec_df_by_charge))
                # print("before concat", len(batch_cluster_df), len(spec_df_by_charge))
                batch_cluster_df = pd.concat([batch_cluster_df, spec_df_by_charge])
                # print("after concat", len(batch_cluster_df))
                prev_length= len(batch_anomaly_df)
                batch_anomaly_df = pd.concat([batch_anomaly_df, anomaly_by_charge])
                metadata_df = batch_anomaly_df
              #  n = len(metadata_df[(metadata_df['cluster']== 18723)]) + len(metadata_df[(metadata_df['cluster']==28038)]) + len(metadata_df[(metadata_df['cluster']== 22736)])
             
                # count_true_anomalies(batch_anomaly_df, None)
                # print("batch anomaly length after", prev_length, len(batch_anomaly_df))
                
            # print("sanity","old anomaly lenght",len(anomaly_df),len(spectra_hvs))
            prev_length = len(anomaly_df)
            prev_clus_length = len(cluster_results)
            cluster_results = pd.concat([cluster_results, batch_cluster_df])
            # print("checking if cluster_results growing", len(cluster_results))
            anomaly_df = cluster_results[cluster_results["anomaly"] == True].copy()
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            # if (n>0):
            #     spec_df_by_charge = anomaly_df
            #     print("BATCH", b)
            #     print("checking global anomaly df ")
            #     print("bucket1249 and idnetnif b1927")
            #     print(spec_df_by_charge[(spec_df_by_charge['bucket']==1249) & (spec_df_by_charge['identifier']=='b1927_293T_proteinID_07A_QE3_122212')])
            #     # sys.exit(0)
      
          #  n = len(metadata_df[(anomaly_df['cluster']== 18723)]) + len(anomaly_df[(anomaly_df['cluster']==28038)]) + len(anomaly_df[(anomaly_df['cluster']== 22736)])

            
       
            # anomaly_df = batch_anomaly_df
            # print("cluster result length b aft",   prev_clus_length , len(cluster_results))
            # print("overall anomaly length after", prev_length, len(anomaly_df))

            # print("batch ", b, "updating cluster results length goes from ", prev_length, "to", len(cluster_results))
            prev_spectra_meta_df = pd.concat([prev_spectra_meta_df, spectra_meta_df], ignore_index=True)
            # print("prev_spectra_meta_dfs", len(prev_spectra_meta_df))
            prev_spectra_hvs =  np.vstack([prev_spectra_hvs, spectra_hvs]) 
            # print("size of prev_sepctra_meta_df", len(prev_spectra_meta_df),"size of cluster_results", len(cluster_results))
           

    hd_preprocess.export_cluster_results(
        spectra_df=cluster_results, config=config, logger=logger)
    
    cluster_results.to_csv("cluster_result/cluster_results_"+config.anomaly_file+".csv", index=False)    
    if (anomaly_df is not None):
        # print("length of anomaly df", len(anomaly_df))
        num_unique_clusters = anomaly_df["cluster"].nunique()
        # print("Number of unique clusters in anomaly_df:", num_unique_clusters)
        # excluded_proteins = pd.read_csv("excluded_spectra-3.csv")
        # num_unique_clusters = excluded_proteins["cluster"].nunique()
        # print("Number of unique clusters in excluded proteins:", num_unique_clusters)

        hd_preprocess.export_anomaly_results(
        spectra_df=anomaly_df, filename="cluster_result/anomaly_results_"+config.anomaly_file, logger=logger)

      
    else:
        # print("anomaly is None")
        pass
    total_runtime = time.perf_counter() - total_start
    print(f"[TIME] total main2.py runtime: {total_runtime:.3f} seconds")

    with open("anomaly_result_timing_best_timing_Gr_HC_Si.csv", "a") as f:
        f.write(f"{config.anomaly_file},{total_runtime:.3f}\n")

  


                

if __name__ == "__main__":
    main()

