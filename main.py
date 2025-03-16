import csv

import kagglehub


# https://chessbot-manors.s3.us-east-2.amazonaws.com/chessData.csv
# https://chessbot-manors.s3.us-east-2.amazonaws.com/choppedData.csv
# https://chessbot-manors.s3.us-east-2.amazonaws.com/choppedTest.csv
# path = kagglehub.dataset_download("ronakbadhe/chess-evaluations")
# print("Path to dataset files:", path)
def test():
    hidden_dims_options = [1024, 2048, 4096]
    epocs_options = [60]
    conv_layers_options = [6, 8]
    fc_layers_options = [6, 8, 10, 14]
    idx = 1
    with open('data.csv', 'w') as file:
        writer = csv.writer(file)
        for layers in fc_layers_options:
            for epocs in epocs_options:
                for hidden_dims in hidden_dims_options:
                    for l in conv_layers_options:
                        writer.writerow(
                            [0, 0, 0, 0, f"test{idx + 1}_model_weights.pth", str(layers), str(l), str(epocs),
                             str(hidden_dims)])
                        idx += 1


test()
