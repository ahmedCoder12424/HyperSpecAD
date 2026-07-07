import pandas as pd
import hd_cluster 
df = pd.read_csv("cluster_results_main.csv")

separation_results = pd.read_csv("cluster_separation_stats_by_bucket.csv")

# separation_results = pd.read_csv("cluster_results_main.csv")

# best separated clusters first
best_clusters = separation_results.sort_values(
    "separation_score",
    ascending=False
)
print(best_clusters.head())

print(best_clusters[best_clusters['size']==25])


sizes = [1,2,3,4,6,8,10,15,25,30,35]

for size in sizes:

  print(size, best_clusters[best_clusters['size']==size][0:3])

  

# frequency_counts = df['cluster'].value_counts()
# frequency_counts = frequency_counts[frequency_counts == 50]



# #1 55253 , 55248, 55254
# #2 55198, 35003, 28
# #3 54, 89, 85
# #4 34903, 34, 5261
# #8 33198, 4937, 969
# #6 12321, 6014, 12
# #10, 43441, 24301, 5080
# #15 21307, 20530, 40322
# #25, 49504, -1,-1
# #30, 28038 , -1,-1
# #35 10854, -1,-1

# a1,a2,a3 = 49504, -1,-1

# #load hvs and metadata df 

# df_18723 = df.loc[(df['cluster'] == a1)]
# df_18723_rep = df_18723['is_representative']
# bucket = df_18723_rep['bucket']
# cluster_reps = df[df['bucket'] == ]['is_representative']


# df_18723.to_csv("anomaly_spectra_s25.csv", index=False)
# print(len(df_18723))

# df_18723 = df.loc[(df['cluster'] != a1) & (df['cluster'] != a2) & (df['cluster'] != a3)  ]
# from sklearn.model_selection import train_test_split

# # count rows per cluster
# cluster_counts = df_18723['cluster'].value_counts()

# # singleton clusters -> must go to 60%
# singleton_clusters = cluster_counts[cluster_counts <=8].index
# multi_clusters = cluster_counts[cluster_counts > 8].index

# df_singletons = df_18723[df_18723['cluster'].isin(singleton_clusters)]
# df_multi = df_18723[df_18723['cluster'].isin(multi_clusters)]

# df_train = df_singletons
# df_test = pd.DataFrame()
# for cluster in df_multi['cluster'].unique():

#     cluster_df = df_multi[df_multi['cluster'] == cluster]

#     buckets = cluster_df['bucket'].unique()
#     if (cluster == 897 or cluster == 893):
#       print(buckets, cluster)
#     for bucket in buckets:

#         cluster_bucket_df = cluster_df[cluster_df['bucket']==bucket]
#         n = len(cluster_bucket_df['bucket'].tolist())
#         if (n <=1):
#           df_train = pd.concat([df_train,cluster_bucket_df], axis=0)
#           continue


#         df_cluster_60, df_cluster_40 = train_test_split(
#         cluster_bucket_df,
#         test_size=0.4,
#         stratify=cluster_bucket_df['bucket'],
#         random_state=42
#        )
#         # print(len(df_cluster_60))
#         # print(len(df_cluster_40))
#         if(cluster == 897 or cluster == 893):
#           print("train",df_cluster_60['bucket'].tolist())
#           print("test" ,df_cluster_40['bucket'].tolist())

#         df_train = pd.concat([df_train, df_cluster_60], axis=0)
#         df_test = pd.concat([df_test, df_cluster_40], axis=0)

# df_train.to_csv("train_datas25.csv", index=False)

# excluded_protein = pd.read_csv("anomaly_spectra_s25.csv")

# # shuffle rows randomly
# excluded_shuffled = excluded_protein.sample(frac=1, random_state=42)

# # add to test set
# df_test = pd.concat([df_test, excluded_shuffled], axis=0).reset_index(drop=True)

# df_test.to_csv("test_data-mixeds25.csv", index=False)