import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessEvaluationTrainer:
    def __init__(self, model, feature_extractor, eval_transformer, device):
        self.model = model
        self.feature_extractor = feature_extractor
        self.eval_transformer = eval_transformer
        self.device = device

        # Optimizer with different parameter groups
        self.optimizer = torch.optim.AdamW([
            {'params': model.conv_block1.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5},
            {'params': model.conv_block2.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5},
            {'params': model.residual_conv.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5},
            {'params': model.fc_block.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4}
        ])

        # Learning rate scheduler
        # NOTE: steps_per_epoch=1 is often too small if you have multiple batches per epoch.
        # Typically, steps_per_epoch should = len(train_dataloader).
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[1e-3, 1e-3, 1e-3, 5e-4],
            steps_per_epoch=1,
            epochs=300,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100
        )

    def custom_loss_function(self, outputs, targets, weights=None):
        """
        Custom loss function that gives higher weight to positions with smaller evaluation differences
        """
        # Basic MSE
        squared_diff = (outputs - targets) ** 2

        # Apply per-sample weights if present
        if weights is not None:
            squared_diff = squared_diff * weights

        # Extra weighting for near-zero targets
        importance_weights = 1.0 / (torch.abs(targets) * 0.1 + 1.0)
        weighted_loss = squared_diff * importance_weights

        return weighted_loss.mean()

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        running_loss = 0.0

        for i, (board_tensors, extras_tensors, eval_values, weights) in enumerate(dataloader):
            # Move everything to the GPU (if available)
            board_tensors = board_tensors.to(self.device)
            extras_tensors = extras_tensors.to(self.device)
            eval_values = eval_values.to(self.device)
            weights = weights.to(self.device)

            # Transform numeric evals (e.g., centipawns) into a smaller scale
            # .item() is not strictly needed if eval_values is shape [batch_size], but let's be explicit.
            transformed_evals = torch.tensor(
                [self.eval_transformer.transform(val.item()) for val in eval_values],
                device=self.device
            )

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(board_tensors, extras_tensors)  # <--- extras_tensors are the "chess_features"

            # Loss
            loss = self.custom_loss_function(outputs, transformed_evals, weights)
            loss.backward()

            # Optional gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update
            self.optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.6f}, Grad Norm: {self._get_grad_norm():.4f}")

        # Advance the LR scheduler (caution: if you have many batches, you might want to call
        # self.scheduler.step() inside the loop or set steps_per_epoch=len(dataloader)).
        self.scheduler.step()

        return running_loss / len(dataloader)

    def validate(self, val_dataloader):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for board_tensors, extras_tensors, eval_values, weights in val_dataloader:
                board_tensors = board_tensors.to(self.device)
                extras_tensors = extras_tensors.to(self.device)
                eval_values = eval_values.to(self.device)

                # Transform
                transformed_evals = torch.tensor(
                    [self.eval_transformer.transform(val.item()) for val in eval_values],
                    device=self.device
                )

                outputs = self.model(board_tensors, extras_tensors)
                # For validation we can skip custom weighting. Just do MSE on the transformed scale:
                loss = F.mse_loss(outputs, transformed_evals)
                val_loss += loss.item()

        return val_loss / len(val_dataloader)

    def _get_grad_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def train(self, train_dataloader, val_dataloader, num_epochs=300):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Train one epoch
            train_loss = self.train_epoch(train_dataloader, epoch)
            # Validate
            val_loss = self.validate(val_dataloader)

            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_chess_eval_model.pth")
                print(f"New best model saved (Val Loss: {val_loss:.6f})")

            # Example "early stopping" if you like
            if epoch > 50 and train_loss < 1e-4:
                print("Training converged. Stopping early.")
                break

        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")