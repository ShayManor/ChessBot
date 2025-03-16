import csv
import os
import time

import pandas as pd
import torch

from train import fen_to_tensor, main, ChessDataset, ChessCNN, process_fen


def load_model_weights(model, weights_file, device):
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model weights loaded from {weights_file}")
    return model


def score_chess_state(fen, model, device, mean_eval, std_eval):
    """
    Given a FEN string, returns the evaluation score in original scale.
    Splits the FEN into board and extra features and passes both to the model.
    """
    # Split FEN into board and extra features
    board, extra = process_fen(fen)
    # Add batch dimension and move both to the GPU
    board = board.to(device).unsqueeze(0)
    extra = extra.to(device).unsqueeze(0)
    with torch.no_grad():
        norm_score = model(board, extra)
    score = norm_score.item() * std_eval + mean_eval
    return score


def create_new():
    hidden_dims_options = [1024, 2048, 4096]
    epocs_options = [60]
    conv_layers_options = [6, 8]
    fc_layers_options = [6, 8, 10, 14]
    with open('data.csv', 'a') as file:
        writer = csv.writer(file)
        for layers in fc_layers_options:
            for epocs in epocs_options:
                for hidden_dims in hidden_dims_options:
                    for l in conv_layers_options:
                        test_dataset = ChessDataset('data/choppedTest.csv', normalize=True)
                        # Get the normalization parameters from the dataset.
                        mean_eval = test_dataset.mean_eval
                        std_eval = test_dataset.std_eval
                        main(num_conv_layers=l, num_fc_layers=layers, num_epochs=epocs, fc_hidden_dim=hidden_dims)
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model = ChessCNN(num_fc_layers=layers, num_conv_layers=l, fc_hidden_dim=hidden_dims).to(device)


def retest():
    total_data = []
    # Create the test dataset with normalization enabled.
    test_dataset = ChessDataset('data/choppedTest.csv', normalize=True)
    # Get the normalization parameters from the dataset.
    # mean_eval = test_dataset.mean_eval
    # std_eval = test_dataset.std_eval

    # For ground-truth, you can also load the DataFrame separately if needed.
    df_test = pd.read_csv('data/choppedTest.csv')

    with open('data.csv', 'r') as f:
        next(f)
        reader = csv.reader(f)
        for line in reader:
            test_dataset = ChessDataset('data/choppedTest.csv', normalize=True)
            print(line)
            layers, l, epocs, hidden_dims = int(line[5]), int(line[6]), int(line[7]), int(line[8])
            mean_eval = test_dataset.mean_eval
            std_eval = test_dataset.std_eval
            # main(num_conv_layers=l, num_fc_layers=layers, num_epochs=epocs, fc_hidden_dim=hidden_dims)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = ChessCNN(num_fc_layers=layers, num_conv_layers=l, fc_hidden_dim=hidden_dims)
            weights_file = line[4]
            model = load_model_weights(model, weights_file, device)
            data = pd.read_csv('data/choppedTest.csv')
            start_time = time.time()
            tot = 0
            total_vals = 0
            counter = 0
            for index, row in data.iterrows():
                counter += 1
                fen = row['FEN']
                exp_eval = float(row['Evaluation'].replace('#', ''))
                evaluation = float(score_chess_state(fen, model, device, mean_eval, std_eval))
                print(f"Eval for {fen} expected: {exp_eval}, result: {evaluation}")
                tot += abs(evaluation - exp_eval)
                total_vals += abs(evaluation)
            print(f"Error rate: {tot / counter}. Average evals = {total_vals / counter}")
            print(
                f"Total time to test: {time.time() - start_time}. Average time: {(time.time() - start_time) / counter}")
            row = [tot / counter, total_vals / counter, time.time() - start_time,
                   (time.time() - start_time) / counter, weights_file, layers, l, epocs, hidden_dims]
            total_data.append(row)

    with open('data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(total_data)


if __name__ == '__main__':
    # create_new()
    # time.sleep(60*2)
    retest()
