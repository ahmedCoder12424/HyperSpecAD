
Steps to Run Anomaly Detection Expirments


MS preprocessing 

use scripts in anomaly_ms_preprocessing_scripts

This step preprocesses the non LC-MS data into mgf-like files which can be inputted into Hyper-Spec pipeline

1.  create_mgf_spectras_from_single_spectrum.py
    this script creates mgf files from the pigments data which is contained in txt files with two columns mz and intensity representing one continous spectrum. Each pigment has its own files. The script samples the spectrum to create multiple spectras to make an mgf style file. 
2.  create_mgf_from_tofs_sims.py
    This parses and processes the battery tof-sims files to create mgf files 

2.  Run create_anomaly_file.py to create a directory from which anomaly files are created a various percentages. Point directory /hdd/data/ to save space.
You can adjust the range of percentages. Change the input and outputs accordingly. 


Dataset Creation

use scripts in anomaly_dataset_creation_scripts

1. Run ./anomaly_dataset_creation_scripts/cluster_anomalies.sh 
    edit the paths to point to the correct normal non-anomolous dataset directory and correct anomaly dataset directory
    set output path in /hdd/data/ to save space 
    This preprocesses and provides initial clusterings of the dataset at various anomaly percentages
2. Run python create_inorganic_datasets_var_anomaly_size.py 
    Set the INPUT Path to the location of the result of the previous step. Set the output accordingly and place it in hdd/data/
    This will create train and test splits of the dataset where the non-anomolous data is split evenly so no new non anomaly clusters 
    pop up in the train which is mixed with the anomaly spectras. The anomaly spectras are also outputted in a separate files
    

Anomaly Expirements 

use scripts in anomaly_expirement_scripts

1. ./anomaly_expirement_scripts/run_anomaly_detection_expirements_param.sh to run expirements for the datasets created in the previous step. Change the paths to point to the right raw anomaly and non anomaly dataset files, the right directory for the processed files. Feel free to add more run directories to run more expirements. Set the accuracy and output timing graphs accordingly 
    knobs you can tune 
    - you can tune the bucket width by changing mz-interval
    - you can tune the threshold percentile by changing anomaly_eps_percentile

2. ./anomaly_expirement_scripts/sweep_anomaly_mz_tradeoff_expirements.sh this script sweeps differnt m/z intervals can be used later to generate a tradeoff graph

Baselines

1. run ./baselines_param.sh to run for the baselines, change the input and output paths accordingly


Generating Graphs and Figures 

use files in anomaly_result_scripts

1. generate_plots.py - set the input paths to timing and accuracy results of hyper-spec and the baseline results and set a directory for the graphs.
 ex: 
    python3 generate_plots_with_tables.py \
    --baselines=../baseline_results/Gr_HC_Si_trunc_organic_anom_baseline_summary.txt \
    --timing=../anomaly_validation_results/best_acc_organic_timing.txt \
    --accuracy=../anomaly_validation_results/best_acc_organic_acc.txt \
    --outdir=../tables/organic_best_acc

This will produced graphs for the timing and different metrics. 
2. plot_tradeoff.py - this plots the tradeoff of different bucket widths effect on timing and accuracy results
    ex:
        python3 plot_tradeoff.py \
    --acc-file results/proteomics/accuracy_raw.txt \
    --time-file results/proteomics/timing_raw.txt \
    --outdir plots/proteomics \
    --prefix proteomics_

