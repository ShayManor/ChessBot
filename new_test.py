import csv
import time
import torch
import pandas as pd
from final_train import NNUEModel, ChessEvalDataset  # Adjust the module name if needed


def test_on_choppedTest2():
    """
    Loads a trained NNUE model from a checkpoint and tests it on the data in choppedTest2.csv.
    Evaluates performance by computing the average absolute error between predicted and true evaluations.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing on device:", device)

    # Create a test dataset from choppedTest2.csv.
    test_dataset = ChessEvalDataset('data/choppedTest2.csv')
    data_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=2)

    # Load the best checkpoint (adjust the path as needed)
    model_checkpoint = "checkpoints/nnue_best.pt"
    print(f"\nLoading model weights from {model_checkpoint}")

    # Initialize the model (ensure the architecture matches your training script)
    model = NNUEModel(hidden_dim=32).to(device)
    state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    start_time = time.time()
    total_abs_error = 0.0
    total_abs_eval = 0.0
    count = 0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device).squeeze(-1)

            # Evaluate the model
            outputs = model(features).squeeze(-1)
            # If you later add normalization, you must invert it here.
            real_preds = outputs  # currently, predictions are in raw centipawn scale

            # Accumulate absolute errors
            for i in range(len(targets)):
                true_eval = targets[i].item()
                pred_eval = real_preds[i].item()
                total_abs_error += abs(pred_eval - true_eval)
                total_abs_eval += abs(pred_eval)
                print(f"Expected Eval: {true_eval}  Returned Eval: {pred_eval}")
                count += 1

    end_time = time.time()
    total_time = end_time - start_time

    if count == 0:
        error_rate = 0
        avg_eval = 0
        avg_time = 0
    else:
        error_rate = total_abs_error / count
        avg_eval = total_abs_eval / count
        avg_time = total_time / count

    print(f"\nTesting complete for {model_checkpoint}:")
    print(f"  Error rate: {error_rate:.4f}")
    print(f"  Average eval: {avg_eval:.4f}")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Avg time per sample: {avg_time:.6f} seconds")


if __name__ == '__main__':
    test_on_choppedTest2()
