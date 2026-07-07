import louvain_module

labels = louvain_module.run_louvain("train_data.csv", k=15)
print(labels)
