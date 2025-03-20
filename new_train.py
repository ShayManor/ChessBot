import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class ChessDataset(Dataset):
    def __init__(self, csv_file, normalize=True):
        """
        Load chess positions and evaluations from a CSV file.

        Args:
            csv_file (str): Path to the CSV file with chess data
            normalize (bool): Whether to normalize the evaluation scores
        """
        self.data = pd.read_csv(csv_file)
        self.normalize = normalize

        # Parse the evaluation values
        self.evaluations = self.data['Evaluation'].apply(parse_eval).values

        # Calculate statistics for normalization
        self.mean_eval = np.mean(self.evaluations)
        self.std_eval = np.std(self.evaluations)

        print(f"Dataset loaded from {csv_file}")
        print(f"Total positions: {len(self.data)}")
        print(f"Evaluation stats - Mean: {self.mean_eval:.4f}, Std: {self.std_eval:.4f}")
        print(f"Min eval: {np.min(self.evaluations):.4f}, Max eval: {np.max(self.evaluations):.4f}")

        # Process the FEN strings to create board tensors
        self.board_tensors = []
        self.extra_feature_tensors = []
        self.weights = []

        for idx in range(len(self.data)):
            fen = self.data.iloc[idx]['FEN']
            board_tensor, extra_features = self._process_fen(fen)
            self.board_tensors.append(board_tensor)
            self.extra_feature_tensors.append(extra_features)

            # Calculate a weight based on the absolute evaluation value
            # Give more weight to balanced positions
            eval_value = self.evaluations[idx]
            weight = 1.0 / (abs(eval_value) * 0.1 + 1.0)
            self.weights.append(weight)

    def _process_fen(self, fen):
        """
        Process a FEN string to create tensors for the neural network.

        Args:
            fen (str): FEN string representing a chess position

        Returns:
            tuple: (board_tensor, extra_features_tensor)
        """
        # Split the FEN string to get the board part
        parts = fen.split(' ')
        board_str = parts[0]
        active_color = parts[1]
        castling_rights = parts[2]
        en_passant = parts[3]

        # Create 12-channel board representation (6 piece types * 2 colors)
        # 0-5: white pieces (P, N, B, R, Q, K)
        # 6-11: black pieces (p, n, b, r, q, k)
        board_tensor = torch.zeros(12, 8, 8)

        # Parse the board representation
        ranks = board_str.split('/')
        for i, rank in enumerate(ranks):
            j = 0
            for char in rank:
                if char.isdigit():
                    j += int(char)
                else:
                    piece_idx = "PNBRQKpnbrqk".index(char)
                    board_tensor[piece_idx, i, j] = 1.0
                    j += 1

        # Create extra features
        extra_features = torch.zeros(8)

        # Side to move
        extra_features[0] = 1.0 if active_color == 'w' else 0.0

        # Castling rights
        extra_features[1] = 1.0 if 'K' in castling_rights else 0.0
        extra_features[2] = 1.0 if 'Q' in castling_rights else 0.0
        extra_features[3] = 1.0 if 'k' in castling_rights else 0.0
        extra_features[4] = 1.0 if 'q' in castling_rights else 0.0

        # En passant
        extra_features[5] = 1.0 if en_passant != '-' else 0.0

        # Additional features (e.g., material balance, piece mobility, etc.)
        # These could be more sophisticated in a real implementation

        # For now, just add some basic material count
        white_material = torch.sum(board_tensor[0:5].sum(dim=(1, 2)) * torch.tensor([1, 3, 3, 5, 9]))
        black_material = torch.sum(board_tensor[6:11].sum(dim=(1, 2)) * torch.tensor([1, 3, 3, 5, 9]))
        material_balance = (white_material - black_material) / 39.0  # Normalize by max possible material

        extra_features[6] = material_balance

        # Phase of the game (rough estimate based on piece count)
        total_pieces = board_tensor.sum()
        extra_features[7] = total_pieces / 32.0  # Normalize by starting piece count

        return board_tensor, extra_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_tensor = self.board_tensors[idx]
        extra_features = self.extra_feature_tensors[idx]
        eval_value = torch.tensor(self.evaluations[idx], dtype=torch.float32)
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)

        # Normalize evaluation if requested
        if self.normalize:
            normalized_eval = (eval_value - self.mean_eval) / (self.std_eval + 1e-6)
            return board_tensor, extra_features, normalized_eval, weight
        else:
            return board_tensor, extra_features, eval_value, weight


def parse_eval(eval_str):
    """
    Parse the evaluation string from the CSV into a numerical value.

    Args:
        eval_str (str): Evaluation string (e.g., "0.45", "#-3", etc.)

    Returns:
        float: Numerical evaluation value
    """
    if isinstance(eval_str, float):
        return eval_str

    try:
        # Handle mate scores
        if '#' in eval_str:
            if eval_str.startswith('#'):
                # White has checkmate in n moves
                moves = int(eval_str[1:])
                return 100.0 - moves  # Return a very high score
            else:
                # Black has checkmate in n moves
                moves = int(eval_str[2:])
                return -100.0 + moves  # Return a very low score
        else:
            # Regular evaluation
            return float(eval_str)
    except:
        # Default value for unparseable evaluations
        return 0.0


class EnhancedChessEvaluationModel(nn.Module):
    def __init__(self, input_channels=12, conv_channels=128, fc_hidden_dim=1024):
        super(EnhancedChessEvaluationModel, self).__init__()

        # Convolutional layers for pattern recognition
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.LeakyReLU(0.1)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels * 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(conv_channels * 2, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels * 2),
            nn.LeakyReLU(0.1)
        )

        # Residual connections
        self.residual_conv = nn.Conv2d(input_channels, conv_channels * 2, kernel_size=1)

        # Feature size after convolution
        conv_output_size = conv_channels * 2 * 8 * 8

        # Fully connected layers for evaluation
        self.fc_block = nn.Sequential(
            nn.Linear(conv_output_size + 8, fc_hidden_dim),  # +8 for the extra features
            nn.BatchNorm1d(fc_hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.BatchNorm1d(fc_hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(fc_hidden_dim // 2, fc_hidden_dim // 4),
            nn.BatchNorm1d(fc_hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(fc_hidden_dim // 4, 1)
        )

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, board_tensor, chess_features):
        # Process board representation through CNNs
        x = self.conv_block1(board_tensor)

        # Apply second conv block with residual connection
        residual = self.residual_conv(board_tensor)
        x = self.conv_block2(x) + residual

        # Flatten and concatenate with chess-specific features
        x = x.view(x.size(0), -1)
        x = torch.cat([x, chess_features], dim=1)

        # Final evaluation through fully connected layers
        evaluation = self.fc_block(x)

        return evaluation


class ChessEvaluationTrainer:
    def __init__(self, model, device, train_mean=None, train_std=None):
        self.model = model
        self.device = device
        self.train_mean = train_mean
        self.train_std = train_std

        # Set up tensorboard writer
        self.writer = SummaryWriter(log_dir='runs/chess_eval')

        # Optimizer with different parameter groups
        self.optimizer = torch.optim.AdamW([
            {'params': model.conv_block1.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5},
            {'params': model.conv_block2.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5},
            {'params': model.residual_conv.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5},
            {'params': model.fc_block.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4}
        ])

    def custom_loss_function(self, outputs, targets, weights=None):
        """
        Custom loss function that gives higher weight to positions with smaller evaluation differences
        """
        # Basic MSE
        squared_diff = (outputs.squeeze(-1) - targets) ** 2

        # Apply per-sample weights if present
        if weights is not None:
            squared_diff = squared_diff * weights

        # Extra weighting for near-zero targets (balanced positions)
        importance_weights = 1.0 / (torch.abs(targets) * 0.1 + 1.0)
        weighted_loss = squared_diff * importance_weights

        return weighted_loss.mean()

    def train_epoch(self, dataloader, epoch, total_epochs):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (board_tensors, extras_tensors, eval_values, weights) in enumerate(dataloader):
            # Move everything to device
            board_tensors = board_tensors.to(self.device)
            extras_tensors = extras_tensors.to(self.device)
            eval_values = eval_values.to(self.device)
            weights = weights.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(board_tensors, extras_tensors)

            # Loss calculation
            loss = self.custom_loss_function(outputs, eval_values, weights)

            # Backward pass and optimize
            loss.backward()

            # Optional gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track statistics
            running_loss += loss.item()

            # Calculate predictions accuracy (sign agreement)
            outputs_squeezed = outputs.squeeze(-1)
            correct_sign = ((outputs_squeezed > 0) == (eval_values > 0)).sum().item()
            correct_predictions += correct_sign
            total_samples += eval_values.size(0)

            # Log progress
            if i % 50 == 0:
                grad_norm = self._get_grad_norm()
                print(f"Epoch {epoch}/{total_epochs}, Batch {i}, Loss: {loss.item():.6f}, Grad Norm: {grad_norm:.4f}")

                # Log to tensorboard
                global_step = epoch * len(dataloader) + i
                self.writer.add_scalar('Training/Loss', loss.item(), global_step)
                self.writer.add_scalar('Training/GradNorm', grad_norm, global_step)

                # Log a few predictions vs targets
                if i % 200 == 0 and i > 0:
                    for j in range(min(5, len(eval_values))):
                        pred = outputs_squeezed[j].item()
                        target = eval_values[j].item()

                        # If we have the statistics, convert to original scale for better interpretability
                        if self.train_mean is not None and self.train_std is not None:
                            pred_orig = pred * self.train_std + self.train_mean
                            target_orig = target * self.train_std + self.train_mean
                            print(f"  Sample {j}: Pred={pred_orig:.4f}, Target={target_orig:.4f}")
                        else:
                            print(f"  Sample {j}: Pred={pred:.4f}, Target={target:.4f}")

        # Calculate epoch statistics
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        # Log epoch statistics
        self.writer.add_scalar('Training/EpochLoss', epoch_loss, epoch)
        self.writer.add_scalar('Training/SignAccuracy', epoch_accuracy, epoch)

        return epoch_loss, epoch_accuracy

    def validate(self, val_dataloader, epoch):
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for board_tensors, extras_tensors, eval_values, weights in val_dataloader:
                board_tensors = board_tensors.to(self.device)
                extras_tensors = extras_tensors.to(self.device)
                eval_values = eval_values.to(self.device)

                # Forward pass
                outputs = self.model(board_tensors, extras_tensors)
                outputs_squeezed = outputs.squeeze(-1)

                # Loss calculation
                loss = F.mse_loss(outputs_squeezed, eval_values)
                val_loss += loss.item()

                # Collect predictions and targets for analysis
                all_preds.extend(outputs_squeezed.cpu().numpy())
                all_targets.extend(eval_values.cpu().numpy())

                # Calculate accuracy (sign agreement)
                correct_sign = ((outputs_squeezed > 0) == (eval_values > 0)).sum().item()
                correct_predictions += correct_sign
                total_samples += eval_values.size(0)

        # Calculate validation statistics
        val_loss = val_loss / len(val_dataloader)
        val_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        # Log validation statistics
        self.writer.add_scalar('Validation/Loss', val_loss, epoch)
        self.writer.add_scalar('Validation/SignAccuracy', val_accuracy, epoch)

        # Log a histogram of predictions
        self.writer.add_histogram('Validation/Predictions', np.array(all_preds), epoch)
        self.writer.add_histogram('Validation/Targets', np.array(all_targets), epoch)

        # If we have the statistics, compute MAE in original scale
        if self.train_mean is not None and self.train_std is not None:
            all_preds_orig = np.array(all_preds) * self.train_std + self.train_mean
            all_targets_orig = np.array(all_targets) * self.train_std + self.train_mean
            mae_orig = np.mean(np.abs(all_preds_orig - all_targets_orig))
            self.writer.add_scalar('Validation/MAE_Original', mae_orig, epoch)
            print(f"Validation MAE in original scale: {mae_orig:.4f}")

            # Log a few predictions vs targets in original scale
            for i in range(min(10, len(all_preds_orig))):
                print(f"  Sample {i}: Pred={all_preds_orig[i]:.4f}, Target={all_targets_orig[i]:.4f}")

        return val_loss, val_accuracy

    def _get_grad_norm(self):
        """Calculate the L2 norm of the gradients"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def train(self, train_dataloader, val_dataloader, num_epochs=50, lr_scheduler=None):
        best_val_loss = float('inf')
        patience = 0
        max_patience = 10  # Early stopping after 10 epochs without improvement

        # Store dataset statistics for later inference
        if hasattr(train_dataloader.dataset, 'mean_eval') and hasattr(train_dataloader.dataset, 'std_eval'):
            self.train_mean = train_dataloader.dataset.mean_eval
            self.train_std = train_dataloader.dataset.std_eval
            print(f"Using dataset statistics - Mean: {self.train_mean:.4f}, Std: {self.train_std:.4f}")

            # Save these statistics for later use in inference
            torch.save({
                'mean': self.train_mean,
                'std': self.train_std
            }, 'normalization_stats.pt')
            print("Saved normalization statistics to normalization_stats.pt")

        for epoch in range(num_epochs):
            # Train one epoch
            train_loss, train_accuracy = self.train_epoch(train_dataloader, epoch, num_epochs)

            # Validate
            val_loss, val_accuracy = self.validate(val_dataloader, epoch)

            # Adjust learning rate if scheduler is provided
            if lr_scheduler is not None:
                lr_scheduler.step()
                current_lr = lr_scheduler.get_last_lr()[0]
                self.writer.add_scalar('Training/LearningRate', current_lr, epoch)

            # Print epoch summary
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"Train Sign Accuracy: {train_accuracy:.4f}, Val Sign Accuracy: {val_accuracy:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # Save model (both state dict and full model)
                torch.save(self.model.state_dict(), "best_chess_eval_model.pth")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'train_mean': self.train_mean,
                    'train_std': self.train_std
                }, "best_chess_eval_model_full.pt")

                print(f"New best model saved (Val Loss: {val_loss:.6f})")
                patience = 0
            else:
                patience += 1

            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_mean': self.train_mean,
                    'train_std': self.train_std
                }, f"checkpoint_epoch_{epoch}.pt")

            # Early stopping check
            if patience >= max_patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break

        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        self.writer.close()

        # Save final model
        torch.save(self.model.state_dict(), "final_chess_eval_model.pth")

        return best_val_loss


def preprocess_and_save_data(dataset_path, output_path, batch_size=128, num_workers=4):
    """
    Load a dataset, preprocess it, and save the preprocessed tensors for faster loading
    """
    dataset = ChessDataset(dataset_path, normalize=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_boards = []
    all_extras = []
    all_evals = []
    all_weights = []

    for board_tensors, extras_tensors, eval_values, weights in dataloader:
        all_boards.append(board_tensors)
        all_extras.append(extras_tensors)
        all_evals.append(eval_values)
        all_weights.append(weights)

    # Concatenate all batches
    all_boards = torch.cat(all_boards, dim=0)
    all_extras = torch.cat(all_extras, dim=0)
    all_evals = torch.cat(all_evals, dim=0)
    all_weights = torch.cat(all_weights, dim=0)

    # Save to disk
    torch.save({
        'boards': all_boards,
        'extras': all_extras,
        'evals': all_evals,
        'weights': all_weights,
        'mean_eval': dataset.mean_eval,
        'std_eval': dataset.std_eval
    }, output_path)

    print(f"Preprocessed data saved to {output_path}")
    print(f"Total positions: {len(all_boards)}")


def create_new_model(train_data_path, val_data_path, hidden_dims=1024, conv_channels=128, num_epochs=50,
                     batch_size=128):
    """
    Create and train a new chess evaluation model
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading training data...")
    if train_data_path.endswith('.pt'):
        # Load preprocessed data
        data = torch.load(train_data_path)
        train_dataset = TensorDataset(data['boards'], data['extras'], data['evals'], data['weights'])
        train_mean = data.get('mean_eval', None)
        train_std = data.get('std_eval', None)
    else:
        # Load from CSV
        train_dataset = ChessDataset(train_data_path, normalize=True)
        train_mean = train_dataset.mean_eval
        train_std = train_dataset.std_eval

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print("Loading validation data...")
    if val_data_path.endswith('.pt'):
        # Load preprocessed data
        data = torch.load(val_data_path)
        val_dataset = TensorDataset(data['boards'], data['extras'], data['evals'], data['weights'])
    else:
        # Load from CSV
        val_dataset = ChessDataset(val_data_path, normalize=True)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Create model
    model = EnhancedChessEvaluationModel(
        input_channels=12,
        conv_channels=conv_channels,
        fc_hidden_dim=hidden_dims
    ).to(device)

    # Create trainer
    trainer = ChessEvaluationTrainer(model, device, train_mean, train_std)

    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        trainer.optimizer,
        max_lr=[1e-3, 1e-3, 1e-3, 5e-4],
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )

    # Train model
    print(f"Starting model training with hidden_dims={hidden_dims}, conv_channels={conv_channels}...")
    start_time = time.time()
    best_val_loss = trainer.train(train_dataloader, val_dataloader, num_epochs=num_epochs, lr_scheduler=lr_scheduler)
    training_time = time.time() - start_time

    # Save final model with descriptive filename
    model_filename = f"chess_eval_model_h{hidden_dims}_c{conv_channels}.pth"
    torch.save(model.state_dict(), model_filename)

    # Save complete model info
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_dims': hidden_dims,
        'conv_channels': conv_channels,
        'val_loss': best_val_loss,
        'training_time': training_time,
        'train_mean': train_mean,
        'train_std': train_std
    }, f"chess_eval_model_h{hidden_dims}_c{conv_channels}_full.pt")

    print(f"Model training completed in {training_time:.2f} seconds")
    print(f"Model saved to {model_filename}")

    # Record results to CSV
    with open('model_results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:  # File is empty, write header
            writer.writerow(['hidden_dims', 'conv_channels', 'val_loss', 'training_time', 'model_file'])
        writer.writerow([hidden_dims, conv_channels, best_val_loss, training_time, model_filename])

    return model, best_val_loss, training_time


def main():
    # Check if we need to preprocess the data first
    preprocess = False
    if preprocess:
        print("Preprocessing training data...")
        preprocess_and_save_data('data/choppedData.csv', 'data/precomputedData.pt')
        print("Preprocessing validation data...")
        preprocess_and_save_data('data/choppedTest.csv', 'data/precomputedTest.pt')
        print("Preprocessing test data...")
        preprocess_and_save_data('data/choppedTest2.csv', 'data/precomputedTest2.pt')

    # Configuration options
    hidden_dims_options = [1024, 2048]
    conv_channels_options = [128, 256]

    # Create results tracking
    results = []

    for hidden_dims in hidden_dims_options:
        for conv_channels in conv_channels_options:
            try:
                model, val_loss, training_time = create_new_model(
                    train_data_path='data/precomputedData.pt',
                    val_data_path='data/precomputedTest.pt',
                    hidden_dims=hidden_dims,
                    conv_channels=conv_channels,
                    num_epochs=50,
                    batch_size=128
                )

                results.append({
                    'hidden_dims': hidden_dims,
                    'conv_channels': conv_channels,
                    'val_loss': val_loss,
                    'training_time': training_time,
                    'model_file': f"chess_eval_model_h{hidden_dims}_c{conv_channels}.pth"
                })
            except Exception as e:
                print(f"Error training model with hidden_dims={hidden_dims}, conv_channels={conv_channels}: {e}")

    # Print results summary
    print("\n===== Training Results Summary =====")
    for result in results:
        print(f"Model with hidden_dims={result['hidden_dims']}, conv_channels={result['conv_channels']}:")
        print(f"  Validation loss: {result['val_loss']:.6f}")
        print(f"  Training time: {result['training_time'] / (60 ** 2):.2f} hours")
        print(f"  Model file: {result['model_file']}")
        print()

    # Find and report the


if __name__ == '__main__':
    main()
