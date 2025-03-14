import os
import time

import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

num_layers = 6
epocs = 25
hidden_dims = 6

# Mapping for pieces: white pieces use uppercase, black pieces use lowercase.
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
    ranks = board_part.split('/')
    for rank in ranks:
        for char in rank:
            if char.isdigit():
                continue
            else:
                advantage += piece_values.get(char, 0)

    return advantage


def fen_to_tensor(fen):
    """
    Converts board into a 774 dimensional tensor (64 squares * 12 pieces + 5 info)
    """
    board, active, castling, en_passant, halfmove, fullmove = fen.split(" ")
    # board, castling king white, queen white, king black, queen black, turn
    tensor = torch.zeros(64 * 12 + 6)
    rows = board.split("/")

    # FEN starts at rank 8 and ends at 1
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
    tensor[768] = 1 if active == 'w' else -1
    options = "KQkq"
    for i in range(769, 773):
        tensor[i] = 1 if options[i - 769] in castling else -1
    tensor[773] = get_advantage(fen)
    return tensor


class ChessDataset(Dataset):
    """
    Pytorch Dataset that loads FEN and evaluations from csv
    """

    def __init__(self, csv_file, start_idx=None, end_idx=None, normalize=True):
        self.data = pd.read_csv(csv_file)
        if start_idx is not None or end_idx is not None:
            self.data = self.data.iloc[start_idx:end_idx].reset_index(drop=True)
        self.normalize = normalize
        if normalize:
            # Convert Evaluation to float (removing any '#' if present)
            self.data['EvalFloat'] = self.data['Evaluation'].apply(
                lambda x: float(str(x).replace('#', ''))
            )
            self.mean_eval = self.data['EvalFloat'].mean()
            self.std_eval = self.data['EvalFloat'].std()
            print(f"Target normalization: mean={self.mean_eval:.4f}, std={self.std_eval:.4f}")
        else:
            self.mean_eval, self.std_eval = 0, 1  # dummy values


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = row['FEN']
        try:
            features = fen_to_tensor(fen)
        except ValueError as e:
            print(f"Error processing FEN at index {idx}: {e}")
            raise
        eval_value = float(row['Evaluation'].replace('#', ''))
        if self.normalize:
            # Normalize target: (target - mean) / std
            norm_eval = (eval_value - self.mean_eval) / self.std_eval
            target = torch.tensor([norm_eval], dtype=torch.float)
        else:
            target = torch.tensor([eval_value], dtype=torch.float)
        return features, target


class NNUE(nn.Module):
    """
    Simple NNUE with 5 fully connected layers
    """

    def __init__(self, input_dim=774, hidden_dim=1024, num_layers=6):
        super(NNUE, self).__init__()
        self.fc = nn.ModuleList()
        self.relu = nn.ModuleList()
        self.fc.append(nn.Linear(input_dim, hidden_dim))
        self.relu.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.fc.append(nn.Linear(hidden_dim, hidden_dim))
            self.relu.append(nn.ReLU())
        self.fc.append(nn.Linear(hidden_dim, 1))

    def forward(self, x):
        for idx in range(len(self.relu)):
            x = self.fc[idx](x)
            x = self.relu[idx](x)
        x = self.fc[-1](x)
        return x


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for features, target in dataloader:
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def test_net(model, fen, device):
    """
    Tests the network on a given FEN string.
    Prints out the network's evaluation.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Convert FEN to tensor and move to device; unsqueeze to add batch dimension.
        features = fen_to_tensor(fen).to(device).unsqueeze(0)
        evaluation = model(features)
        print(f"Evaluation for FEN '{fen}': {evaluation.item():.4f}")


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the data provided by the dataloader using the given loss criterion.
    Returns the average loss over the dataset.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, target in dataloader:
            features, target = features.to(device), target.to(device)
            output = model(features)
            loss = criterion(output, target)
            # Multiply by the number of samples in the batch for weighted sum
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

    average_loss = total_loss / total_samples if total_samples else float('inf')
    print(f"Average Evaluation Loss: {average_loss:.4f}")
    return average_loss


def save_model_weights(model, file_path="test2_model_weights.pth"):
    """
    Saves the model's state dictionary (its weights) to the specified file.
    """
    torch.save(model.state_dict(), file_path)
    print(f"Model weights saved to {file_path}")


def create_test_dataset(csv_file):
    # Read the full CSV to get the total number of rows
    data = pd.read_csv(csv_file)
    total_rows = len(data)
    test_start = int(0.9 * total_rows)  # Last 10% of rows
    print(f"Total rows: {total_rows}. Using rows {test_start} to {total_rows} for testing.")

    # Create and return a ChessDataset for the test set
    return ChessDataset(csv_file, start_idx=test_start, end_idx=total_rows)


def main(layers, epocs, hidden_dims):
    # Set up device: use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ChessDataset('data/choppedData.csv')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = NNUE(774, hidden_dims, layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epocs):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{epocs}, Loss: {loss:.4f}")
    test_dataset = create_test_dataset("data/choppedData.csv")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    evaluate(model, test_loader, criterion, device)
    i = 1
    while os.path.exists(f'test{i}_model_weights.pth'):
        i += 1
    save_model_weights(model, f'test{i}_model_weights.pth')
    # Example: normalize target evaluations
    # eval_values = df['Evaluation'].apply(lambda x: float(str(x).replace('#', '')))
    # mean_eval = eval_values.mean()
    # std_eval = eval_values.std()
    # df['Normalized Evaluation'] = eval_values.sub(mean_eval).div(std_eval)


if __name__ == '__main__':
    main(num_layers, epocs, hidden_dims)
