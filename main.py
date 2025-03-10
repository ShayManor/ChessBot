import csv

import kagglehub

# path = kagglehub.dataset_download("ronakbadhe/chess-evaluations")
# print("Path to dataset files:", path)
def test():
    hidden_dims_options = [256, 512, 1024]
    epocs_options = [10, 25, 30]
    layers_options = [2, 3, 4, 5, 6, 7, 10]
    idx = 0
    with open('data.csv', 'r+') as file:
        reader = list(csv.reader(file))
        data = list(reader)
        print(data)
        for layers in layers_options:
            for epocs in epocs_options:
                for hidden_dims in hidden_dims_options:
                    data[idx].extend([str(layers), str(epocs), str(hidden_dims)])
                    idx += 1
        with open('data.csv', 'w') as f:
            writer = csv.writer(f)
            for d in data:
                writer.writerow(d)


