import os
import time
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from preproces_data import PrecomputedChessDataset

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
    full_tensor = fen_to_tensor(fen)
    board_tensor = full_tensor[:768].view(12, 8, 8)
    extra_tensor = full_tensor[768:]
    return board_tensor, extra_tensor


class ChessDataset(Dataset):
    def __init__(self, csv_file, start_idx=None, end_idx=None, normalize=True, norm_params=None, num_bins=100):
        self.data = pd.read_csv(csv_file)
        if start_idx is not None or end_idx is not None:
            self.data = self.data.iloc[start_idx:end_idx].reset_index(drop=True)
        self.normalize = normalize
        self.data['EvalFloat'] = self.data['Evaluation'].apply(lambda x: float(str(x).replace('#', '')))

        evals = self.data['EvalFloat'].values
        hist, bin_edges = np.histogram(evals, bins=num_bins, density=True)
        bin_indices = np.digitize(evals, bins=bin_edges, right=True)
        bin_freq = np.array([hist[i - 1] if i > 0 and i - 1 < len(hist) else 1.0 for i in bin_indices])
        weights = 1.0 / (bin_freq + 1e-6)
        weights = weights / np.mean(weights)
        self.data['weight'] = weights

        if self.normalize:
            if norm_params is None:
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
        weight = torch.tensor([row['weight']], dtype=torch.float)
        return (board, extra), target, weight


class ImprovedChessCNN(nn.Module):
    def __init__(self, conv_channels=64, dropout_rate=0.2, fc_hidden_dim=512,
                 num_conv_layers=3, num_fc_layers=2):
        """
        Improved model adjustments:
          - Uses LeakyReLU instead of ReLU for better gradient flow.
          - Reduces dropout in convolutional layers (or removes it) and applies dropout only in fully connected layers.
          - Lowers weight decay (handled in the optimizer) outside the model.
        """
        super(ImprovedChessCNN, self).__init__()
        conv_layers = []
        # First convolutional layer
        conv_layers.append(nn.Conv2d(12, conv_channels, kernel_size=3, padding=1))
        conv_layers.append(nn.BatchNorm2d(conv_channels))
        conv_layers.append(nn.LeakyReLU(0.1))
        # (Removed dropout in conv layers to improve gradient flow)
        # Additional convolutional layers:
        for _ in range(1, num_conv_layers):
            conv_layers.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.BatchNorm2d(conv_channels))
            conv_layers.append(nn.LeakyReLU(0.1))
        self.conv = nn.Sequential(*conv_layers)
        conv_output_dim = conv_channels * 8 * 8
        fc_input_dim = conv_output_dim + 6
        fc_layers = []
        if num_fc_layers >= 2:
            fc_layers.append(nn.Linear(fc_input_dim, fc_hidden_dim))
            fc_layers.append(nn.BatchNorm1d(fc_hidden_dim))
            fc_layers.append(nn.LeakyReLU(0.1))
            fc_layers.append(nn.Dropout(dropout_rate))  # Dropout applied in FC layers only
            for _ in range(num_fc_layers - 2):
                fc_layers.append(nn.Linear(fc_hidden_dim, fc_hidden_dim))
                fc_layers.append(nn.BatchNorm1d(fc_hidden_dim))
                fc_layers.append(nn.LeakyReLU(0.1))
                fc_layers.append(nn.Dropout(dropout_rate))
            fc_layers.append(nn.Linear(fc_hidden_dim, 1))
        else:
            fc_layers.append(nn.Linear(fc_input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, board, extra):
        x = self.conv(board)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, extra), dim=1)
        out = self.fc(x)
        return out


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def improved_train_model(hidden_dim, num_conv_layers, num_fc_layers):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PrecomputedChessDataset('data/precomputedData.pt', normalize=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

    model = ImprovedChessCNN(conv_channels=64, dropout_rate=0.2, fc_hidden_dim=hidden_dim,
                             num_conv_layers=num_conv_layers, num_fc_layers=num_fc_layers)
    model.to(device)

    # Lower weight decay helps improve gradient flow
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.MSELoss()

    warmup_epochs = 5
    total_epochs = 60

    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        for i, ((board, extra), target, weight) in enumerate(dataloader):
            board = board.to(device)
            extra = extra.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(board, extra)
            loss = criterion(output, target)
            loss.backward()

            grad_norm = model.fc[0].weight.grad.norm().item()
            if i % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Grad Norm: {grad_norm:.4f}")

            optimizer.step()
            running_loss += loss.item() * board.size(0)

        if epoch < warmup_epochs:
            warmup_lr = 1e-3 * float(epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{total_epochs} - Loss: {epoch_loss:.4f} - LR: {get_lr(optimizer):.6f}")

    # After training, test on a few FEN strings
    test_fens = [
        "4Rrk1/1p6/p2p2pp/P1pq2Nr/3b3P/6P1/1P2QP2/3R2K1 w - - 4 32",
        "4Rrk1/1p6/p2p2pp/P1pq2Nr/3R3P/6P1/1P2QP2/6K1 b - - 0 32",
        "4Rrk1/1p2Q3/p2p2pp/P2q2Nr/3p3P/6P1/1P3P2/6K1 w - - 0 34"
    ]
    print("\nPredictions on test FENs:")
    for fen in test_fens:
        board, extra = process_fen(fen)
        board = board.unsqueeze(0).to(device)
        extra = extra.unsqueeze(0).to(device)
        output = model(board, extra)
        prediction = output.item() * dataset.std_eval + dataset.mean_eval
        print(f"FEN: {fen}\n Prediction: {prediction:.4f}\n")


if __name__ == '__main__':
    improved_train_model()