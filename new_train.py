import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessFeatureExtractor:
    def __init__(self):
        # Piece values (traditional + slight adjustments)
        self.piece_values = {
            'P': 1.0, 'N': 3.0, 'B': 3.25, 'R': 5.0, 'Q': 9.0, 'K': 0.0,
            'p': -1.0, 'n': -3.0, 'b': -3.25, 'r': -5.0, 'q': -9.0, 'k': 0.0
        }
        # Central squares have higher value
        self.square_values = torch.zeros(8, 8)
        for i in range(8):
            for j in range(8):
                # Distance from center
                dist = max(abs(i - 3.5), abs(j - 3.5))
                self.square_values[i, j] = 1.0 - dist / 3.5

    def parse_fen(self, fen):
        # Parse FEN string to get board state
        parts = fen.split(' ')
        board_str = parts[0]
        active = parts[1]
        castling = parts[2]
        en_passant = parts[3]

        # Convert to 8x8 board representation
        board = []
        for row in board_str.split('/'):
            board_row = []
            for ch in row:
                if ch.isdigit():
                    board_row.extend(['.'] * int(ch))
                else:
                    board_row.append(ch)
            board.append(board_row)
        return board, active, castling, en_passant

    def compute_material_balance(self, board):
        # Calculate material balance
        material = 0
        for i in range(8):
            for j in range(8):
                if board[i][j] in self.piece_values:
                    material += self.piece_values[board[i][j]]
        return material

    def compute_piece_mobility(self, board):
        # Simplified mobility calculation
        white_mobility = 0
        black_mobility = 0

        # Mobility approximation based on empty squares adjacent to pieces
        for i in range(8):
            for j in range(8):
                if board[i][j] == '.':
                    continue

                piece = board[i][j]
                is_white = piece.isupper()
                mobility = 0

                # Check adjacent squares (simplified)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 8 and 0 <= nj < 8 and board[ni][nj] == '.':
                            mobility += 1

                if is_white:
                    white_mobility += mobility
                else:
                    black_mobility += mobility

        return white_mobility - black_mobility

    def compute_center_control(self, board):
        # Calculate control of the center
        center_control = 0
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]

        for i, j in center_squares:
            if board[i][j] in 'PNBRQK':
                center_control += 0.5
            elif board[i][j] in 'pnbrqk':
                center_control -= 0.5

        return center_control

    def compute_pawn_structure(self, board):
        # Analyze pawn structure (doubled, isolated pawns)
        white_pawn_files = [0] * 8
        black_pawn_files = [0] * 8

        for i in range(8):
            for j in range(8):
                if board[i][j] == 'P':
                    white_pawn_files[j] += 1
                elif board[i][j] == 'p':
                    black_pawn_files[j] += 1

        # Count doubled pawns (penalty)
        white_doubled = sum(max(0, count - 1) for count in white_pawn_files)
        black_doubled = sum(max(0, count - 1) for count in black_pawn_files)

        # Count isolated pawns
        white_isolated = sum(1 for j in range(8) if white_pawn_files[j] > 0 and
                             (j == 0 or white_pawn_files[j - 1] == 0) and
                             (j == 7 or white_pawn_files[j + 1] == 0))
        black_isolated = sum(1 for j in range(8) if black_pawn_files[j] > 0 and
                             (j == 0 or black_pawn_files[j - 1] == 0) and
                             (j == 7 or black_pawn_files[j + 1] == 0))

        # Calculate pawn structure score
        pawn_structure = -0.3 * (white_doubled - black_doubled) - 0.2 * (white_isolated - black_isolated)
        return pawn_structure

    def extract_features(self, fen):
        # Extract comprehensive chess features from FEN
        board, active, castling, en_passant = self.parse_fen(fen)

        # Calculate features
        material = self.compute_material_balance(board)
        mobility = self.compute_piece_mobility(board)
        center_control = self.compute_center_control(board)
        pawn_structure = self.compute_pawn_structure(board)

        # King safety (simplified)
        king_safety = 0
        for i in range(8):
            for j in range(8):
                if board[i][j] == 'K':
                    # White king prefers to be behind pawns
                    if i > 0 and board[i - 1][j] == 'P':
                        king_safety += 0.3
                elif board[i][j] == 'k':
                    # Black king prefers to be behind pawns
                    if i < 7 and board[i + 1][j] == 'p':
                        king_safety -= 0.3

        # Castling rights
        castling_value = 0
        if 'K' in castling or 'Q' in castling:
            castling_value += 0.2
        if 'k' in castling or 'q' in castling:
            castling_value -= 0.2

        # Side to move
        tempo = 0.2 if active == 'w' else -0.2

        # Combine all features into a feature vector
        features = torch.tensor([
            material,
            mobility * 0.1,
            center_control * 0.5,
            pawn_structure,
            king_safety,
            castling_value,
            tempo
        ])

        return features


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
            nn.Linear(conv_output_size + 7, fc_hidden_dim),  # +7 for the chess-specific features
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


class NonLinearEvalTransform:
    """
    Transforms chess evaluations non-linearly to better preserve meaning
    """

    def __init__(self, clip_value=2000):
        self.clip_value = clip_value

    def transform(self, eval_value):
        # Clip extreme values
        eval_value = max(min(eval_value, self.clip_value), -self.clip_value)

        # Apply sigmoid-like transformation that preserves zero
        if eval_value >= 0:
            return 2.0 / (1.0 + torch.exp(-eval_value / 300)) - 1.0
        else:
            return -2.0 / (1.0 + torch.exp(eval_value / 300)) + 1.0

    def inverse_transform(self, transformed_value):
        # Inverse transformation to get back the original scale
        if transformed_value >= 0:
            return -300 * torch.log(2.0 / (transformed_value + 1.0) - 1.0)
        else:
            return 300 * torch.log(2.0 / (-transformed_value + 1.0) - 1.0)


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
        # Basic MSE loss
        squared_diff = (outputs - targets) ** 2

        # Apply importance weights if provided
        if weights is not None:
            squared_diff = squared_diff * weights

        # Add regularization based on evaluation size
        # Positions closer to 0 (equal) should be predicted more accurately
        importance_weights = 1.0 / (torch.abs(targets) * 0.1 + 1.0)

        # Combine losses
        weighted_loss = squared_diff * importance_weights
        return weighted_loss.mean()

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        running_loss = 0.0

        for i, (board_tensors, fen_strings, eval_values, weights) in enumerate(dataloader):
            # Move data to device
            board_tensors = board_tensors.to(self.device)
            eval_values = eval_values.to(self.device)
            weights = weights.to(self.device)

            # Extract additional chess features
            chess_features = torch.zeros(len(fen_strings), 7, device=self.device)
            for j, fen in enumerate(fen_strings):
                chess_features[j] = self.feature_extractor.extract_features(fen).to(self.device)

            # Transform evaluation values
            transformed_evals = torch.tensor([self.eval_transformer.transform(e) for e in eval_values],
                                             device=self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(board_tensors, chess_features)

            # Calculate loss
            loss = self.custom_loss_function(outputs, transformed_evals, weights)

            # Backpropagation
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Log progress
            running_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.6f}, "
                      f"Grad Norm: {self._get_grad_norm():.4f}")

        # Update learning rate
        self.scheduler.step()

        # Return average loss for the epoch
        return running_loss / len(dataloader)

    def _get_grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def validate(self, val_dataloader):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for board_tensors, fen_strings, eval_values, _ in val_dataloader:
                # Move data to device
                board_tensors = board_tensors.to(self.device)
                eval_values = eval_values.to(self.device)

                # Extract additional chess features
                chess_features = torch.zeros(len(fen_strings), 7, device=self.device)
                for j, fen in enumerate(fen_strings):
                    chess_features[j] = self.feature_extractor.extract_features(fen).to(self.device)

                # Transform evaluation values
                transformed_evals = torch.tensor([self.eval_transformer.transform(e) for e in eval_values],
                                                 device=self.device)

                # Forward pass
                outputs = self.model(board_tensors, chess_features)

                # Calculate loss (without weights for validation)
                loss = F.mse_loss(outputs, transformed_evals)
                val_loss += loss.item()

        return val_loss / len(val_dataloader)

    def train(self, train_dataloader, val_dataloader, num_epochs=300):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_dataloader, epoch)

            # Validate the model
            val_loss = self.validate(val_dataloader)

            # Print epoch summary
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"best_chess_eval_model.pth")
                print(f"New best model saved (Val Loss: {val_loss:.6f})")

            # Early stopping check
            if epoch > 50 and train_loss < 1e-4:
                print("Training converged. Stopping early.")
                break

        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")