import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def parse_eval(eval_str):
    """
    Parse the evaluation string from the CSV into a numerical value.
    Handling mate scores by using +/- 100.0 offsets.
    """
    if isinstance(eval_str, float):
        return eval_str

    try:
        # Handle mate scores
        if '#' in eval_str:
            if eval_str.startswith('#'):
                # White checkmate in n moves => large positive
                moves = int(eval_str[1:])
                return 100.0 - moves
            else:
                # Black checkmate in n moves => large negative
                moves = int(eval_str[2:])
                return -100.0 + moves
        else:
            # Regular evaluation
            return float(eval_str)
    except:
        # If unparseable, default to 0.0
        return 0.0


class ChessDataset(Dataset):
    def __init__(self, csv_file, normalize=True):
        """
        Load chess positions and evaluations from a CSV file.
        """
        self.data = pd.read_csv(csv_file)
        self.normalize = normalize

        # Parse the evaluation values
        self.evaluations = self.data['Evaluation'].apply(parse_eval).values

        # Calculate statistics for normalization
        self.mean_eval = np.mean(self.evaluations)
        self.std_eval = np.std(self.evaluations)
        if self.std_eval < 1e-6:
            self.std_eval = 1e-6  # avoid dividing by zero

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
        parts = fen.split(' ')
        board_str = parts[0]
        active_color = parts[1]
        castling_rights = parts[2]
        en_passant = parts[3]

        # Create 12-channel board representation
        board_tensor = torch.zeros(12, 8, 8)
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

        extra_features = torch.zeros(8)
        # Side to move
        extra_features[0] = 1.0 if active_color == 'w' else 0.0
        # Castling
        extra_features[1] = 1.0 if 'K' in castling_rights else 0.0
        extra_features[2] = 1.0 if 'Q' in castling_rights else 0.0
        extra_features[3] = 1.0 if 'k' in castling_rights else 0.0
        extra_features[4] = 1.0 if 'q' in castling_rights else 0.0
        # En passant
        extra_features[5] = 0.0 if en_passant == '-' else 1.0

        # Material balance (extremely rough)
        white_material = torch.sum(board_tensor[0:5].sum(dim=(1, 2)) * torch.tensor([1, 3, 3, 5, 9]))
        black_material = torch.sum(board_tensor[6:11].sum(dim=(1, 2)) * torch.tensor([1, 3, 3, 5, 9]))
        material_balance = (white_material - black_material) / 39.0
        extra_features[6] = material_balance

        # Phase of the game estimate
        total_pieces = board_tensor.sum()
        extra_features[7] = total_pieces / 32.0

        return board_tensor, extra_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_tensor = self.board_tensors[idx]
        extra_features = self.extra_feature_tensors[idx]
        eval_value = self.evaluations[idx]  # raw
        weight = self.weights[idx]  # raw

        # Return in normalized form if self.normalize
        if self.normalize:
            norm_eval = (eval_value - self.mean_eval) / self.std_eval
            eval_tensor = torch.tensor(norm_eval, dtype=torch.float32)
        else:
            eval_tensor = torch.tensor(eval_value, dtype=torch.float32)

        weight_tensor = torch.tensor(weight, dtype=torch.float32)
        return board_tensor, extra_features, eval_tensor, weight_tensor


class EnhancedChessEvaluationModel(nn.Module):
    def __init__(self, input_channels=12, conv_channels=128, fc_hidden_dim=1024):
        super(EnhancedChessEvaluationModel, self).__init__()

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
        # Residual
        self.residual_conv = nn.Conv2d(input_channels, conv_channels * 2, kernel_size=1)

        conv_output_size = conv_channels * 2 * 8 * 8
        self.fc_block = nn.Sequential(
            nn.Linear(conv_output_size + 8, fc_hidden_dim),
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
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, board_tensor, chess_features):
        x = self.conv_block1(board_tensor)
        residual = self.residual_conv(board_tensor)
        x = self.conv_block2(x) + residual

        x = x.view(x.size(0), -1)
        x = torch.cat([x, chess_features], dim=1)
        evaluation = self.fc_block(x)
        return evaluation


class ChessEvaluationTrainer:
    def __init__(self, model, device, train_mean=None, train_std=None):
        self.model = model
        self.device = device
        self.train_mean = train_mean
        self.train_std = train_std

        self.writer = SummaryWriter(log_dir='runs/chess_eval')

        # AdamW, but raising the LR clipping threshold
        self.optimizer = torch.optim.AdamW([
            {'params': model.conv_block1.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5},
            {'params': model.conv_block2.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5},
            {'params': model.residual_conv.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5},
            {'params': model.fc_block.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4}
        ])

        self.clip_value = 5.0  # was 1.0, increase so we don't always clip

    def custom_loss_function(self, outputs, targets, weights=None):
        squared_diff = (outputs.squeeze(-1) - targets) ** 2
        if weights is not None:
            squared_diff = squared_diff * weights
        # Weighted more for near-zero positions
        importance_weights = 1.0 / (torch.abs(targets) * 0.1 + 1.0)
        weighted_loss = squared_diff * importance_weights
        return weighted_loss.mean()

    def _get_grad_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def train_epoch(self, dataloader, epoch, total_epochs):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (board_tensors, extras_tensors, eval_values, weights) in enumerate(dataloader):
            board_tensors = board_tensors.to(self.device)
            extras_tensors = extras_tensors.to(self.device)
            eval_values = eval_values.to(self.device)
            weights = weights.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(board_tensors, extras_tensors)
            loss = self.custom_loss_function(outputs, eval_values, weights)
            loss.backward()

            # Clip grads at 5.0
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_value)
            self.optimizer.step()

            running_loss += loss.item()

            # sign-based "accuracy"
            outputs_squeezed = outputs.squeeze(-1)
            correct_sign = ((outputs_squeezed > 0) == (eval_values > 0)).sum().item()
            correct_predictions += correct_sign
            total_samples += eval_values.size(0)

            if i % 50 == 0:
                grad_norm = self._get_grad_norm()
                print(f"Epoch {epoch}/{total_epochs}, Batch {i}, Loss: {loss.item():.6f}, Grad Norm: {grad_norm:.4f}")
                global_step = epoch * len(dataloader) + i
                self.writer.add_scalar('Training/Loss', loss.item(), global_step)
                self.writer.add_scalar('Training/GradNorm', grad_norm, global_step)

                # every 200 steps, log sample predictions in original scale
                if i % 200 == 0 and i > 0:
                    for j in range(min(5, len(eval_values))):
                        pred_norm = outputs_squeezed[j].item()
                        target_norm = eval_values[j].item()
                        if self.train_mean is not None and self.train_std is not None:
                            pred_orig = pred_norm * self.train_std + self.train_mean
                            target_orig = target_norm * self.train_std + self.train_mean
                            print(f"  Sample {j}: Pred={pred_orig:.4f}, Target={target_orig:.4f}")
                        else:
                            print(f"  Sample {j}: Pred={pred_norm:.4f}, Target={target_norm:.4f}")

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

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

                outputs = self.model(board_tensors, extras_tensors)
                outputs_squeezed = outputs.squeeze(-1)

                # MSE in normalized scale
                loss = F.mse_loss(outputs_squeezed, eval_values)
                val_loss += loss.item()

                # sign-based
                correct_sign = ((outputs_squeezed > 0) == (eval_values > 0)).sum().item()
                correct_predictions += correct_sign
                total_samples += eval_values.size(0)

                all_preds.extend(outputs_squeezed.cpu().numpy())
                all_targets.extend(eval_values.cpu().numpy())

        val_loss = val_loss / len(val_dataloader)
        val_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        self.writer.add_scalar('Validation/Loss', val_loss, epoch)
        self.writer.add_scalar('Validation/SignAccuracy', val_accuracy, epoch)

        # if we have stats, compute MAE in original scale
        if self.train_mean is not None and self.train_std is not None:
            preds_orig = np.array(all_preds) * self.train_std + self.train_mean
            targets_orig = np.array(all_targets) * self.train_std + self.train_mean
            mae_orig = np.mean(np.abs(preds_orig - targets_orig))
            self.writer.add_scalar('Validation/MAE_Original', mae_orig, epoch)
            print(f"Validation MAE in original scale: {mae_orig:.4f}")

            # Print a few
            for i in range(min(5, len(preds_orig))):
                print(f"  [Val Sample {i}] Pred={preds_orig[i]:.4f}, Target={targets_orig[i]:.4f}")

        return val_loss, val_accuracy

    def train(self, train_dataloader, val_dataloader, num_epochs=50, lr_scheduler=None):
        best_val_loss = float('inf')
        patience = 0
        max_patience = 10

        # If the dataset has mean/std, store them
        if hasattr(train_dataloader.dataset, 'mean_eval') and hasattr(train_dataloader.dataset, 'std_eval'):
            self.train_mean = train_dataloader.dataset.mean_eval
            self.train_std = max(train_dataloader.dataset.std_eval, 1e-6)
            print(f"Using dataset statistics - Mean: {self.train_mean:.4f}, Std: {self.train_std:.4f}")

            # Save them
            torch.save({'mean': self.train_mean, 'std': self.train_std}, 'normalization_stats.pt')
            print("Saved normalization statistics to normalization_stats.pt")

        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.train_epoch(train_dataloader, epoch, num_epochs)
            val_loss, val_accuracy = self.validate(val_dataloader, epoch)

            if lr_scheduler is not None:
                lr_scheduler.step()
                current_lr = lr_scheduler.get_last_lr()[0]
                self.writer.add_scalar('Training/LearningRate', current_lr, epoch)

            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"Train Sign Accuracy: {train_accuracy:.4f}, Val Sign Accuracy: {val_accuracy:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_chess_eval_model.pth")
                print(f"New best model saved (Val Loss: {val_loss:.6f})")
                patience = 0
            else:
                patience += 1

            # Checkpoint every 5 epochs
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

            if patience >= max_patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break

        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        self.writer.close()

        # Save final model
        torch.save(self.model.state_dict(), "final_chess_eval_model.pth")
        return best_val_loss