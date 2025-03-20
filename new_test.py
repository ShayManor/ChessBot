import csv
import time

import pandas as pd
import torch

from new_train import EnhancedChessEvaluationModel
from preproces_data import parse_eval
from train import ChessDataset


def test_on_choppedTest2():
    """
    Reads the existing rows from data.csv (skips header),
    loads each specified model, then tests on 'choppedTest2.csv'.
    Appends new results to data.csv with the new test stats
    (Error rate, average eval, total test time, average test time, etc.).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing on device:", device)

    # We'll create a dataset from choppedTest2.csv.
    # Make sure your ChessDataset supports the 'normalize=True' if needed.
    test_dataset = ChessDataset('data/choppedTest2.csv', normalize=True)
    mean_eval = test_dataset.mean_eval
    std_eval = test_dataset.std_eval

    # Read existing data from data.csv (skipping the header).
    rows_for_append = []
    with open('data.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header line
        for line in reader:
            # Expect columns like:
            # 0: Error Rate
            # 1: Average Evals
            # 2: Time to test
            # 3: Average Time
            # 4: Model Weights
            # 5: Layers
            # 6: conv_layers
            # 7: Epocs
            # 8: Hidden Dimensions
            # We mostly only need columns [4..8] to reconstruct the model.

            model_weights = line[4]
            layers = int(line[5])
            conv_layers = 128
            epocs = int(line[7])
            hidden_dims = int(line[8])

            # Now we test on choppedTest2.csv using these weights.
            # 1) Create the model
            model = EnhancedChessEvaluationModel(
                input_channels=12,
                conv_channels=conv_layers,  # e.g. 128 or 256
                fc_hidden_dim=hidden_dims  # e.g. 1024 or 2048
            ).to(device)

            # 2) Load the weights
            print(f"\nLoading model weights from {model_weights}")
            model.load_state_dict(torch.load(model_weights, map_location=device))
            model.eval()

            # 3) Do a pass over choppedTest2.csv
            data_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=64,
                                                      shuffle=False,
                                                      num_workers=2)
            start_time = time.time()
            total_abs_error = 0.0
            total_abs_eval = 0.0
            count = 0

            with torch.no_grad():
                for batch in data_loader:
                    board_tensors, extras_tensors, eval_values, _ = batch
                    board_tensors = board_tensors.to(device)
                    extras_tensors = extras_tensors.to(device)

                    # Evaluate (predictions)
                    outputs = model(board_tensors, extras_tensors)  # shape [batch_size, 1]
                    outputs = outputs.squeeze(-1)  # shape [batch_size]

                    # Convert back to real scale, because we stored eval in real scale
                    # You said your 'ChessDataset' might also do normalization,
                    # so we unnormalize with (outputs * std_eval + mean_eval) if that is the correct approach.
                    # Double-check if you are using the same normalization logic.
                    real_preds = outputs * std_eval + mean_eval

                    # Now compare to actual eval_values (these are presumably also in real scale).
                    # Possibly you have to parse the '#' if your dataset does that automatically.
                    train_ds = pd.read_csv('data/choppedData.csv')
                    train_ds = train_ds['Evaluation'].apply(parse_eval)
                    train_std = train_ds.std()
                    train_mean = train_ds.mean()
                    print(f"Training mean: {train_mean}, Training std: {train_std}")
                    # Accumulate absolute differences
                    for i in range(len(eval_values)):
                        true_eval = eval_values[i].item()  # real scale
                        pred_eval = real_preds[i].item()
                        total_abs_error += abs(pred_eval - true_eval)
                        total_abs_eval += abs(pred_eval)
                        print(f"Expected Eval: {true_eval} Returned Eval: {pred_eval}")
                        count += 1

            end_time = time.time()
            total_time = end_time - start_time
            if count == 0:
                # Avoid dividing by zero; means no data in choppedTest2.csv
                error_rate = 0
                avg_eval = 0
                avg_time = 0
            else:
                error_rate = total_abs_error / count
                avg_eval = total_abs_eval / count
                avg_time = total_time / count

            print(f"Testing complete for {model_weights}:")
            print(f"  Error rate: {error_rate:.4f}")
            print(f"  Average eval: {avg_eval:.4f}")
            print(f"  Total time: {total_time:.4f}")
            print(f"  Avg time: {avg_time:.6f}")

            # Prepare a new row to append to data.csv
            new_row = [
                error_rate,  # error rate
                avg_eval,  # average eval
                total_time,  # total time
                avg_time,  # average time
                model_weights,  # same model weights
                layers,  # layers
                conv_layers,  # conv_layers
                epocs,  # epocs
                hidden_dims  # hidden dims
            ]
            rows_for_append.append(new_row)

    # Finally, append these new rows to data.csv
    with open('data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for row in rows_for_append:
            writer.writerow(row)

    print("\nFinished testing on choppedTest2.csv. New rows appended to data.csv.")


if __name__ == '__main__':
    test_on_choppedTest2()
