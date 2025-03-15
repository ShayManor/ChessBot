import os
import time
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from preproces_data import PrecomputedChessDataset

# --- Helper functions for processing FEN ---
piece_to_idx = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}


def get_advantage(fen):
    piece_values = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0
    }
    board_part = fen.split()[0]
    advantage = 0
    for rank in board_part.split('/'):
        for char in rank:
            if char.isdigit():
                continue
            advantage += piece_values.get(char, 0)
    return advantage


def fen_to_tensor(fen):
    """
    Converts FEN into a 774-dimensional tensor:
      - First 768: board pieces (12 channels × 64 squares)
      - Last 6: extra features (turn, castling rights, material advantage)
    """
    board, active, castling, en_passant, halfmove, fullmove = fen.split(" ")
    tensor = torch.zeros(64 * 12 + 6)
    rows = board.split("/")
    for idx, row in enumerate(rows):
        file = 0
        for char in row:
            if char.isdigit():
                file += int(char)
            else:
                square = idx * 8 + file
                feat = square * 12 + piece_to_idx[char]
                tensor[feat] = 1
                file += 1
    # Extra features:
    tensor[768] = 1 if active == 'w' else -1
    options = "KQkq"
    for i in range(769, 773):
        tensor[i] = 1 if options[i - 769] in castling else -1
    tensor[773] = get_advantage(fen)
    return tensor


def process_fen(fen):
    """
    Splits the full tensor into:
      - board_tensor: shape [12, 8, 8]
      - extra_tensor: shape [6]
    """
    full_tensor = fen_to_tensor(fen)
    board_tensor = full_tensor[:768].view(12, 8, 8)
    extra_tensor = full_tensor[768:]
    return board_tensor, extra_tensor


# --- Dataset Class with Improved Normalization ---

class ChessDataset(Dataset):
    def __init__(self, csv_file, start_idx=None, end_idx=None, normalize=True, norm_params=None):
        self.data = pd.read_csv(csv_file)
        if start_idx is not None or end_idx is not None:
            self.data = self.data.iloc[start_idx:end_idx].reset_index(drop=True)
        self.normalize = normalize
        # Convert Evaluation to float (remove '#' if present)
        self.data['EvalFloat'] = self.data['Evaluation'].apply(lambda x: float(str(x).replace('#', '')))
        if self.normalize:
            if norm_params is None:
                # Compute normalization parameters from this dataset (ideally training set)
                self.mean_eval = self.data['EvalFloat'].mean()
                self.std_eval = self.data['EvalFloat'].std()
            else:
                self.mean_eval, self.std_eval = norm_params
            print(f"Normalization parameters: mean={self.mean_eval:.4f}, std={self.std_eval:.4f}")
        else:
            self.mean_eval, self.std_eval = 0, 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = row['FEN']
        board, extra = process_fen(fen)
        eval_value = float(str(row['Evaluation']).replace('#', ''))
        if self.normalize:
            norm_eval = (eval_value - self.mean_eval) / self.std_eval
            target = torch.tensor([norm_eval], dtype=torch.float)
        else:
            target = torch.tensor([eval_value], dtype=torch.float)
        # Return tuple: ((board tensor, extra features), target)
        return (board, extra), target


# --- New Model Architecture with Programmable Layers, Dropout, and Batch Normalization ---

class ChessCNN(nn.Module):
    def __init__(self, conv_channels=64, dropout_rate=0.5, fc_hidden_dim=512,
                 num_conv_layers: int = 3, num_fc_layers: int = 2):
        """
        conv_channels: Number of filters for all convolutional layers.
        dropout_rate: Dropout probability.
        fc_hidden_dim: Hidden dimension for fully connected layers.
        num_conv_layers: Programmatically select the number of convolutional layers.
        num_fc_layers: Programmatically select the number of fully connected layers.
                       (Minimum 1; if 1 then direct mapping from input to output.)
        """
        super(ChessCNN, self).__init__()
        conv_layers = []
        # First convolution: input channels 12 -> conv_channels
        conv_layers.append(nn.Conv2d(12, conv_channels, kernel_size=3, padding=1))
        conv_layers.append(nn.BatchNorm2d(conv_channels))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Dropout2d(dropout_rate))
        # Additional convolutional layers:
        for _ in range(1, int(num_conv_layers)):
            conv_layers.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.BatchNorm2d(conv_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout2d(dropout_rate))
        self.conv = nn.Sequential(*conv_layers)
        # Calculate conv output dimension (conv_channels x 8 x 8)
        conv_output_dim = conv_channels * 8 * 8
        # Input dimension for fully connected layers (add 6 extra features)
        fc_input_dim = conv_output_dim + 6
        fc_layers = []
        if int(num_fc_layers) >= 2:
            fc_layers.append(nn.Linear(fc_input_dim, fc_hidden_dim))
            fc_layers.append(nn.BatchNorm1d(fc_hidden_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            # Additional fc layers if specified
            for _ in range(num_fc_layers - 2):
                fc_layers.append(nn.Linear(fc_hidden_dim, fc_hidden_dim))
                fc_layers.append(nn.BatchNorm1d(fc_hidden_dim))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout_rate))
            fc_layers.append(nn.Linear(fc_hidden_dim, 1))
        else:
            # Only one FC layer: direct mapping.
            fc_layers.append(nn.Linear(fc_input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, board, extra):
        # board shape: [batch_size, 12, 8, 8]
        x = self.conv(board)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, extra), dim=1)
        out = self.fc(x)
        return out


# --- Training, Evaluation, and Saving Functions ---

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    scaler = torch.amp.GradScaler('cuda')  # Initialize the GradScaler at the beginning of training
    for (board, extra), target in dataloader:
        board = board.to(device)
        extra = extra.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            output = model(board, extra)
            output = output.squeeze(1)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for (board, extra), target in dataloader:
            board = board.to(device)
            extra = extra.to(device)
            target = target.to(device)
            output = model(board, extra)
            loss = criterion(output, target)
            total_loss += loss.item() * board.size(0)
            total_samples += board.size(0)
    avg_loss = total_loss / total_samples if total_samples else float('inf')
    print(f"Average Evaluation Loss: {avg_loss:.4f}")
    return avg_loss


def save_model_weights(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model weights saved to {file_path}")


# --- Main Training Script with Programmable Hyperparameters ---

def main(num_conv_layers=3, num_fc_layers=2, conv_channels=64, fc_hidden_dim=512,
         dropout_rate=0.25, num_epochs=25, learning_rate=1e-3):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training dataset and compute normalization parameters
    train_dataset = PrecomputedChessDataset('data/precomputedData.pt', normalize=True)
    norm_params = (train_dataset.mean_eval, train_dataset.std_eval)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

    # Use last 10% of data as test set with training normalization parameters
    total_rows = len(pd.read_csv('data/choppedData.csv'))
    test_start = int(0.9 * total_rows)
    test_dataset = ChessDataset('data/choppedData.csv', start_idx=test_start, end_idx=total_rows,
                                normalize=True, norm_params=norm_params)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model with programmable layers
    model = ChessCNN(conv_channels=conv_channels, dropout_rate=dropout_rate,
                     fc_hidden_dim=fc_hidden_dim, num_conv_layers=num_conv_layers,
                     num_fc_layers=num_fc_layers)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        scheduler.step()

    # evaluate(model, test_loader, criterion, device)

    # Save the model weights with a unique filename.
    i = 1
    while os.path.exists(f'test{i}_model_weights.pth'):
        i += 1
    save_model_weights(model, f'test{i}_model_weights.pth')


if __name__ == '__main__':
    # You can experiment with these parameters as needed.
    main(num_conv_layers=3, num_fc_layers=2, conv_channels=64, fc_hidden_dim=512,
         dropout_rate=0.25, num_epochs=25, learning_rate=1e-3)
