import louvain_py
result = louvain_py.run_louvain_from_csv("train_data.csv", k=15)
print(result)
