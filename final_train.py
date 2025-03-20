import os
import math
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Ensure that the custom Ranger optimizer (ranger.py) is in the same directory.
from ranger import Ranger


##############################################
# 1. Dataset and Feature Extraction
##############################################

def _flip_vertical(square):
    # Flip square index vertically (rank inversion) to maintain orientation
    return square ^ 56  # XOR with 56 flips the board vertically (0..63)


class ChessEvalDataset(Dataset):
    """
    Dataset for chess positions in FEN format with evaluations.
    Expected CSV format: each line is "FEN,evaluation"
    evaluation: centipawn value or mate in moves (e.g. "#5" or "#-3")
    """

    def __init__(self, data_file):
        self.samples = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split the line into FEN and evaluation
                try:
                    fen, eval_str = line.split(',')
                except Exception as e:
                    print("Error parsing line:", line)
                    continue
                if fen.lower() == "fen" or eval_str.lower() == "evaluation":
                    continue
                target = self.parse_eval(eval_str)
                self.samples.append((fen, target))

    def parse_eval(self, eval_str):
        """Convert evaluation string to numeric centipawn value."""
        eval_str = eval_str.strip()
        if eval_str.startswith('#'):
            # Mate score conversion. For mate in N, assign a large value:
            # e.g., mate in 1 becomes 32000-100, mate in 2 becomes 32000-200, etc.
            mate_moves = int(eval_str.replace('#', '').replace('+', '').replace('-', ''))
            sign = -1 if '-' in eval_str else 1
            return sign * (32000 - mate_moves * 100)
        else:
            return float(eval_str)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, target = self.samples[idx]
        board = chess.Board(fen)
        # Build NNUE features:
        features = self.board_to_nnue_features(board)
        # Determine side-to-move flag: 1 for white, 0 for black.
        stm = 1.0 if board.turn == chess.WHITE else 0.0
        # Append stm flag as last element of white half features
        # For simplicity, we assume the white half is modified to include this flag.
        # We'll split the features into two halves later.
        # Here features is a 2 x (feature_dim) tensor.
        # We add stm flag to the first half's last element.
        features[0, -1] = stm
        return features.view(-1), torch.tensor([target], dtype=torch.float32)

    def board_to_nnue_features(self, board):
        """
        Construct a simplified NNUE feature vector.
        We'll construct two halves (white perspective and black perspective),
        each of dimension: 64 (king square) * 64 (board square) * 11 (piece types).
        This is a simplified version and may be adjusted.
        """
        NUM_SQ = 64
        NUM_PIECE_TYPES = 11  # simplified: index 0..10 for pieces (pawn through queen and extra for king)
        # feature dimension for one half:
        feat_dim = NUM_SQ * NUM_SQ * NUM_PIECE_TYPES
        # Create two halves: one for white, one for black
        features = torch.zeros((2, feat_dim), dtype=torch.float32)

        # Identify king positions (if missing, default to 0)
        wk_sq = board.king(chess.WHITE)
        bk_sq = board.king(chess.BLACK)
        if wk_sq is None: wk_sq = 0
        if bk_sq is None: bk_sq = 0

        # For each piece on board (excluding kings):
        for square, piece in board.piece_map().items():
            if piece.piece_type == chess.KING:
                continue  # Skip kings in this simple feature representation
            # Map piece type to index 0..(NUM_PIECE_TYPES-1)
            pt = piece.piece_type - 1  # pawn->0, knight->1, bishop->2, rook->3, queen->4, king->5 but skip king
            # For this simplified version, if piece is a king, we can map to an extra index if needed.
            # Here, we assume only non-king pieces are encoded.
            # Determine color index: 0 for white, 1 for black (this doubles the effective index)
            color_idx = 0 if piece.color == chess.WHITE else 1
            # For white perspective half:
            # Use king square (wk_sq) as a major index, board square as minor.
            idx_w = (wk_sq * NUM_SQ * NUM_PIECE_TYPES) + ((pt * 2 + color_idx) * NUM_SQ) + square
            # For black perspective half:
            idx_b = (bk_sq * NUM_SQ * NUM_PIECE_TYPES) + ((pt * 2 + color_idx) * NUM_SQ) + square
            # Set feature active:
            if idx_w < feat_dim:
                features[0, idx_w] = 1.0
            if idx_b < feat_dim:
                features[1, idx_b] = 1.0

        # Return a 2 x feat_dim tensor; later we'll flatten into a single vector.
        return features


##############################################
# 2. NNUE-Inspired Model Definition
##############################################

class NNUEModel(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=32):
        """
        input_dim: total dimension of flattened feature vector.
                   By default, two halves of dimension (64*64*11) each.
        hidden_dim: size of hidden layers.
        """
        super(NNUEModel, self).__init__()
        # Define input dimensions based on NNUE feature dimensions:
        # For each half, dimension = 64 (king positions) * 64 (squares) * 11 (piece types)
        self.half_dim = 64 * 64 * 11
        if input_dim is None:
            input_dim = 2 * self.half_dim
        self.input_dim = input_dim

        # Feature transformer: reduce each half to 256 dimensions.
        self.feature_linear = nn.Linear(self.half_dim, 256)
        # Final accumulator will be 512 (concatenated transformed halves).
        # Hidden layers:
        self.fc1 = nn.Linear(512, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: batch of flattened features, shape (batch, input_dim)
        We split x into two halves.
        The side-to-move flag is embedded in the last element of the first half.
        """
        batch_size = x.size(0)
        # Reshape to (batch, 2, half_dim)
        x = x.view(batch_size, 2, self.half_dim)
        white_feats = x[:, 0, :]  # white perspective half
        black_feats = x[:, 1, :]  # black perspective half

        # Pass each half through feature transformer
        w = self.feature_linear(white_feats)
        b = self.feature_linear(black_feats)

        # Retrieve side-to-move flag from the last element of white_feats (assumed stored there)
        stm_flag = white_feats[:, -1].unsqueeze(1)  # shape (batch, 1), 1 if white to move, else 0

        # Combine halves according to side-to-move:
        # If white to move (flag=1): accumulator = [w || b]
        # If black to move (flag=0): accumulator = [b || w]
        accum_white = torch.cat([w, b], dim=1)
        accum_black = torch.cat([b, w], dim=1)
        accum = stm_flag * accum_white + (1 - stm_flag) * accum_black

        # Apply activations (clipped ReLU as an approximation)
        x = torch.clamp(accum, 0.0, 1.0)
        x = torch.clamp(self.fc1(x), 0.0, 1.0)
        x = torch.clamp(self.fc2(x), 0.0, 1.0)
        out = self.output(x)
        return out


##############################################
# 3. Training Setup
##############################################

def train(model, dataloader, optimizer, criterion, device, scaler, epoch, writer):
    model.train()
    running_loss = 0.0
    for batch_idx, (features, targets) in enumerate(dataloader):
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(features)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * features.size(0)
        # if batch_idx % 100 == 0:
        #     print(f"Epoch {epoch} Batch {batch_idx}: Loss {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader.dataset)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    print(f"Epoch {epoch} Training Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * features.size(0)
    avg_loss = running_loss / len(dataloader.dataset)
    writer.add_scalar("Loss/val", avg_loss, epoch)
    print(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}")
    return avg_loss


##############################################
# 4. Main Training Routine
##############################################

def main():
    # Settings
    data_file_train = "data/chessData.csv"  # Path to your Kaggle CSV file
    data_file_val = "data/choppedTest.csv"  # Optionally, a separate validation set
    num_epochs = 10
    batch_size = 512
    lr = 1e-3
    hidden_dim = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Prepare datasets and dataloaders
    train_dataset = ChessEvalDataset(data_file_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    if os.path.exists(data_file_val):
        val_dataset = ChessEvalDataset(data_file_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        val_loader = None

    # Initialize model
    model = NNUEModel(hidden_dim=hidden_dim).to(device)

    # Initialize Ranger optimizer
    optimizer = Ranger(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Mixed precision scaler
    scaler = GradScaler()

    # TensorBoard writer for logging
    writer = SummaryWriter(log_dir="logs/nnue_train")

    # Directory for saving checkpoints
    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = math.inf
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device, scaler, epoch, writer)
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device, epoch, writer)
        else:
            val_loss = None

        # Save checkpoint
        checkpoint_path = os.path.join("checkpoints", f"nnue_epoch{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Optionally save the best model
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join("checkpoints", "nnue_best.pt"))
            print("Best model updated.")

    writer.close()


if __name__ == "__main__":
    main()
