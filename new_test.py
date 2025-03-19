import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import os

from new_train import ChessFeatureExtractor, NonLinearEvalTransform, EnhancedChessEvaluationModel, \
    ChessEvaluationTrainer
# Import necessary classes from your first file
from train import ChessDataset


def main():
    # Configuration options similar to create_new()
    hidden_dims_options = [1024, 2048]  # Using the fc_hidden_dim from EnhancedChessEvaluationModel
    conv_channels_options = [128, 256]  # Using the conv_channels from EnhancedChessEvaluationModel

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize feature extractor
    feature_extractor = ChessFeatureExtractor()

    # Initialize evaluation transformer
    eval_transformer = NonLinearEvalTransform(clip_value=2000)

    # Create results tracking
    results = []

    for hidden_dims in hidden_dims_options:
        for conv_channels in conv_channels_options:
            print(f"\nTraining model with hidden_dims={hidden_dims}, conv_channels={conv_channels}")

            # Initialize model
            model = EnhancedChessEvaluationModel(
                input_channels=12,  # Standard chess input (6 pieces x 2 colors)
                conv_channels=conv_channels,
                fc_hidden_dim=hidden_dims
            ).to(device)

            # Set up trainer
            trainer = ChessEvaluationTrainer(
                model=model,
                feature_extractor=feature_extractor,
                eval_transformer=eval_transformer,
                device=device
            )

            # Load data
            print("Loading training data...")
            train_dataloader = load_chess_data('data/precomputedData.pt', batch_size=64)
            val_dataloader = load_chess_data('data/choppedTest.csv', batch_size=64, is_validation=True)

            # Train model
            print("Starting model training...")
            start_time = time.time()
            trainer.train(train_dataloader, val_dataloader, num_epochs=100)
            training_time = time.time() - start_time

            # Save model weights
            model_filename = f"chess_eval_model_h{hidden_dims}_c{conv_channels}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved to {model_filename}")

            # Fine-tune on tactical positions
            print("Fine-tuning on tactical positions...")
            fine_tune_dataloader = load_chess_data('data/tactic_precomputedData.pt', batch_size=64)

            # Reload model with best weights
            model.load_state_dict(torch.load("best_chess_eval_model.pth", map_location=device))

            # Fine-tune
            start_time = time.time()
            trainer.train(fine_tune_dataloader, val_dataloader, num_epochs=50)
            fine_tuning_time = time.time() - start_time

            # Save fine-tuned model
            fine_tuned_filename = f"chess_eval_model_finetuned_h{hidden_dims}_c{conv_channels}.pth"
            torch.save(model.state_dict(), fine_tuned_filename)
            print(f"Fine-tuned model saved to {fine_tuned_filename}")

            # Record results
            results.append({
                'hidden_dims': hidden_dims,
                'conv_channels': conv_channels,
                'training_time': training_time,
                'fine_tuning_time': fine_tuning_time,
                'model_file': model_filename,
                'fine_tuned_file': fine_tuned_filename
            })

    # Print results summary
    print("\n===== Training Results Summary =====")
    for result in results:
        print(f"Model with hidden_dims={result['hidden_dims']}, conv_channels={result['conv_channels']}:")
        print(f"  Training time: {result['training_time']:.2f} seconds")
        print(f"  Fine-tuning time: {result['fine_tuning_time']:.2f} seconds")
        print(f"  Model file: {result['model_file']}")
        print(f"  Fine-tuned file: {result['fine_tuned_file']}")
        print()


def load_chess_data(data_path, batch_size=64, is_validation=False):
    """
    Load chess data from file and prepare DataLoader
    This function needs to be adapted to your specific data format
    """
    # This is a placeholder - you'll need to implement this based on your data format
    # If data_path ends with .pt, load preprocessed data
    if data_path.endswith('.pt'):
        # Load preprocessed tensor data
        data = torch.load(data_path)
        dataset = TensorDataset(data['board_tensors'], data['fen_strings'],
                                data['eval_values'], data['weights'])
    else:
        # Load from CSV and process
        dataset = ChessDataset(data_path, normalize=True)

    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=4 if not is_validation else 2,
        pin_memory=True
    )


if __name__ == "__main__":
    main()