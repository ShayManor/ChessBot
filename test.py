import csv
import os
import time

import pandas as pd
import torch

from chess import fen_to_tensor, NNUE, num_layers, main


def load_model_weights(model, weights_file, device):
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model weights loaded from {weights_file}")
    return model


def score_chess_state(fen, model, device):
    """
    Given a FEN string, returns the evaluation score from the neural network.
    """
    board_tensor = fen_to_tensor(fen).to(device)
    board_tensor = board_tensor.unsqueeze(0)

    with torch.no_grad():
        score = model(board_tensor)
    return score.item()


def create_new():
    hidden_dims_options = [256, 512, 1024, 2048, 4096]
    epocs_options = [50, 100]
    layers_options = [2, 3, 4, 6, 8, 10, 12]
    with open('data.csv', 'a') as file:
        writer = csv.writer(file)
        for layers in layers_options:
            for epocs in epocs_options:
                for hidden_dims in hidden_dims_options:
                    main(layers=layers, epocs=epocs, hidden_dims=hidden_dims)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = NNUE(num_layers=layers, hidden_dim=hidden_dims)
                    i = 1
                    while os.path.exists(f"test{i}_model_weights.pth"):
                        i += 1
                    weights_file = f"test{i - 1}_model_weights.pth"
                    model = load_model_weights(model, weights_file, device)
                    data = pd.read_csv('data/choppedTest.csv')
                    start_time = time.time()
                    tot = 0
                    total_vals = 0
                    counter = 0
                    for index, row in data.iterrows():
                        counter += 1
                        fen = row['FEN']
                        exp_eval = float(row['Evaluation'].replace('#', '')) / 100
                        evaluation = float(score_chess_state(fen, model, device))
                        # print(f"Eval for {fen} expected: {exp_eval}, result: {evaluation}")
                        tot += abs(evaluation - exp_eval)
                        total_vals += abs(evaluation)
                    print(f"Error rate: {tot / len(data)}. Average evals = {total_vals / len(data)}")
                    print(
                        f"Total time to test: {time.time() - start_time}. Average time: {(time.time() - start_time) / counter}")
                    row = [tot / len(data), total_vals / len(data), time.time() - start_time,
                           (time.time() - start_time) / counter, weights_file, layers, epocs, hidden_dims]
                    writer.writerow(row)
                    file.flush()


def retest():
    total_data = []
    with open('data.csv', 'r') as f:
        next(f)
        reader = csv.reader(f)
        for line in reader:
            # main(layers=line[6], epocs=line[7], hidden_dims=line[8])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = NNUE(num_layers=int(line[5]), hidden_dim=int(line[7]))
            i = 1
            weights_file = line[4]
            model = load_model_weights(model, weights_file, device)
            dtypes = {
                'FEN': str,
                'Evaluation': str,
            }
            data = pd.read_csv('data/choppedTest.csv', dtype=dtypes)
            start_time = time.time()
            tot = 0
            total_vals = 0
            counter = 0
            for index, row in data.iterrows():
                counter += 1
                fen = row['FEN']
                exp_eval = float(str(row['Evaluation']).replace('#', '')) / 100
                evaluation = float(score_chess_state(fen, model, device))
                tot += abs(evaluation - exp_eval)
                total_vals += abs(evaluation)
                print(f"Eval for {fen} expected: {exp_eval}, result: {evaluation}")
            print(f"Error rate: {tot / len(data)}. Average evals = {total_vals / len(data)}")
            print(f"Total time to test: {time.time() - start_time}. Average time: {(time.time() - start_time) / counter}")
            row = [tot / len(data), total_vals / len(data), time.time() - start_time,
                   (time.time() - start_time) / counter, weights_file, line[5], line[6], line[7]]
            total_data.append(row)
    with open('data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerows(total_data)


if __name__ == '__main__':
    create_new()
    time.sleep(60*2)
    retest()
