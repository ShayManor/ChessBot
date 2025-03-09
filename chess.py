import time

import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Mapping for pieces: white pieces use uppercase, black pieces use lowercase.
piece_to_idx = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}


def fen_to_tensor(fen):
    """
    Converts board into a 768 dimensional tensor (64 squares * 12 pieces)
    """
    board, active, castling, en_passant, halfmove, fullmove = fen.split(" ")
    tensor = torch.zeros(64 * 12)
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
    return tensor


class ChessDataset(Dataset):
    """
    Pytorch Dataset that loads FEN and evaluations from csv
    """

    def __init__(self, csv):
        self.data = pandas.read_csv(csv)

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
        eval_value = float(row['Evaluation'].replace('#',''))
        return features, torch.tensor([eval_value], dtype=torch.float)


class NNUE(nn.Module):
    """
    Simple NNUE with 2 fully connected layers
    """

    def __init__(self, input_dim=768, hidden_dim=256):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # output score

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
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


def main():
    # Set up device: use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ChessDataset('data/chessData.csv')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = NNUE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
    start_time = time.time()
    test_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    test_net(model, test_fen, device)
    print(f"Time taken: {time.time() - start_time}")


if __name__ == '__main__':
    main()
