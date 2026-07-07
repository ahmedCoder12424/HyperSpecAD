import louvain_py
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




for i in range(1):
    matrix = load_csv_as_2d_list("97_bucket.csv")
   #:wprint(len(matrix), matrix)
#rint(matrix)
# Run the Louvain algorithm using the in-memory 2D list
    result = louvain_py.run_louvain(matrix, k=15)

    print(len(result))

#result = louvain_py.run_louvain_from_csv("train_data.csv", k=15)
#print(result)
