import hd_preprocess_incr


hvs=hd_preprocess_incrload_datasets("/home/zheyu/hyperspec/Hyper-Spec/1468dataset/out_eps0.25_initial_0.6_increeps_0.3.csv.parquet")


load_checkpoint()


spectra_meta_df, spectra_hvs = load_checkpoint( file_name ,logger)
cluster_results = load_clustering_results file_name ,logger)
prevResults = hd_preprocess.StaticClusterResults(spectra_meta_df, spectra_hvs, cluster_results)
cluster_reps = hd_preprocess.get_all_cluster_reps(prevResults)
You can modify the loading functions like this for easier use
def load_clustering_result(
  file_name,
  logger: logging
  ):
  """
  Restore from previously saved checkpoint files (spectra meta and encoded hvs)
  Parameters
  ----------
  config :
    Config that defines runtime parameters
  Returns
  -------
  spectra_meta_df :
    Restored spectra meta dataframe
  spectra_hvs :
    Restored spectra hvs array
  """
  cluster_file = filename+'.parquet'
  cluster_results = pd.read_parquet(cluster_file) \
    if os.path.exists(cluster_file) else None
  if (cluster_results is not None):
    logger.info("Successfully restored cluster results from {}!".format(cluster_file))
  else:
    logger.info("No cluster results found")
  return cluster_results
def load_checkpoint(
  file_name,
  logger: logging
  ):
  """
  Restore from previously saved checkpoint files (spectra meta and encoded hvs)
  Parameters
  ----------
  config :
    Config that defines runtime parameters
  Returns
  -------
  spectra_meta_df :
    Restored spectra meta dataframe
  spectra_hvs :
    Restored spectra hvs array
  """
  ckp_parquet_file = file_name+ '_meta.ckp'
  ckp_hvs_file = file_name + '_hvs.ckp'
  spectra_meta_df = pd.read_parquet(ckp_parquet_file) \
    if os.path.exists(ckp_parquet_file) else None
  spectra_hvs = None
  if os.path.exists(ckp_hvs_file):
    with open(ckp_hvs_file, 'rb') as f:
      spectra_hvs = np.load(f)
  if (spectra_meta_df is not None) and (spectra_hvs is not None):
    logger.info("Successfully restored checkpoints from {} and {}!".format(ckp_parquet_file, ckp_hvs_file))
  else:
    logger.info("Incomplete checkpoints!")
  return spectra_meta_df, spectra_hvs