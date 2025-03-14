import csv
import os
import time

import pandas as pd
import torch

from train import fen_to_tensor, NNUE, num_layers, main, ChessDataset


def load_model_weights(model, weights_file, device):
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model weights loaded from {weights_file}")
    return model


def score_chess_state(fen, model, device, mean_eval, std_eval):
    """
    Given a FEN string, returns the evaluation score in original scale.
    """
    board_tensor = fen_to_tensor(fen).to(device).unsqueeze(0)
    with torch.no_grad():
        norm_score = model(board_tensor)
    score = norm_score.item() * std_eval + mean_eval
    return score



def create_new():
    hidden_dims_options = [2048, 4096]
    epocs_options = [60]
    layers_options = [10, 12]
    with open('data.csv', 'w') as file:
        writer = csv.writer(file)
        for layers in layers_options:
            for epocs in epocs_options:
                for hidden_dims in hidden_dims_options:
                    test_dataset = ChessDataset('data/choppedTest.csv', normalize=True)
                    # Get the normalization parameters from the dataset.
                    mean_eval = test_dataset.mean_eval
                    std_eval = test_dataset.std_eval
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
                        evaluation = float(score_chess_state(fen, model, device, mean_eval, std_eval))
                        # print(f"Eval for {fen} expected: {exp_eval}, result: {evaluation}")
                        tot += abs(evaluation - exp_eval)
                        total_vals += abs(evaluation)
                    print(f"Error rate: {tot / counter}. Average evals = {total_vals / counter}")
                    print(
                        f"Total time to test: {time.time() - start_time}. Average time: {(time.time() - start_time) / counter}")
                    row = [tot / counter, total_vals / counter, time.time() - start_time,
                           (time.time() - start_time) / counter, weights_file, layers, epocs, hidden_dims]
                    writer.writerow(row)
                    file.flush()


def retest():
    total_data = []
    # Create the test dataset with normalization enabled.
    test_dataset = ChessDataset('data/choppedTest.csv', normalize=True)
    # Get the normalization parameters from the dataset.
    mean_eval = test_dataset.mean_eval
    std_eval = test_dataset.std_eval

    # For ground-truth, you can also load the DataFrame separately if needed.
    df_test = pd.read_csv('data/choppedTest.csv')

    with open('data.csv', 'r') as f:
        next(f)
        reader = csv.reader(f)
        for line in reader:
            print(line)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Use the parameters from the CSV line to create the model
            model = NNUE(num_layers=int(line[5]), hidden_dim=int(line[7])).to(device)
            weights_file = line[4]
            model = load_model_weights(model, weights_file, device)

            start_time = time.time()
            tot = 0
            total_vals = 0
            counter = 0

            # Iterate over the test dataset using the original FEN values from df_test for ground truth.
            for idx, row in df_test.iterrows():
                counter += 1
                fen = row['FEN']
                exp_eval = float(str(row['Evaluation']).replace('#', ''))
                # Use our score_chess_state function with normalization inversion.
                evaluation = float(score_chess_state(fen, model, device, mean_eval, std_eval))
                tot += abs(evaluation - exp_eval)
                total_vals += abs(evaluation)
                print(f"Eval for {fen} expected: {exp_eval}, result: {evaluation}")

            elapsed = time.time() - start_time
            print(f"Error rate: {tot / len(df_test)}. Average evals = {total_vals / counter}")
            print(f"Total time to test: {elapsed}. Average time: {elapsed / counter}")
            row_data = [tot / counter, total_vals / counter, elapsed, elapsed / counter,
                        weights_file, line[5], line[6], line[7]]
            total_data.append(row_data)

    with open('data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(total_data)


if __name__ == '__main__':
    create_new()
    # time.sleep(60*2)
    # retest()
