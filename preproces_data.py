import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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


def precompute_data(csv_file, output_file, num_bins=100):
    df = pd.read_csv(csv_file)
    boards = []
    extras = []
    evals = []
    for _, row in df.iterrows():
        fen = row['FEN']
        board, extra = process_fen(fen)
        boards.append(board.unsqueeze(0))
        extras.append(extra.unsqueeze(0))
        eval_val = float(str(row['Evaluation']).replace('#', ''))
        evals.append(torch.tensor([eval_val], dtype=torch.float))
    boards = torch.cat(boards, dim=0)  # Shape: [N, 12, 8, 8]
    extras = torch.cat(extras, dim=0)  # Shape: [N, 6]
    evals = torch.cat(evals, dim=0)  # Shape: [N, 1]
    evals_tensor = torch.tensor(evals, dtype=torch.float).unsqueeze(1)  # Shape: [N, 1]
    # Compute weights using a histogram on the raw evals.
    evals_np = np.array(evals)
    hist, bin_edges = np.histogram(evals_np, bins=num_bins, density=True)
    bin_indices = np.digitize(evals_np, bins=bin_edges, right=True)
    bin_freq = np.array([hist[i-1] if i > 0 and i-1 < len(hist) else 1.0 for i in bin_indices])
    weights = 1.0 / (bin_freq + 1e-6)
    # Normalize weights so that the average weight is 1
    weights = weights / np.mean(weights)
    weights_tensor = torch.tensor(weights, dtype=torch.float).unsqueeze(1)  # Shape: [N, 1]
    torch.save({'boards': boards, 'extras': extras, 'evals': evals_tensor, 'weights': weights_tensor}, output_file)
    print(f"Precomputed tensors saved to {output_file}")


class PrecomputedChessDataset(Dataset):
    def __init__(self, tensor_file, normalize=True, norm_params=None):
        data = torch.load(tensor_file)
        self.boards = data['boards']  # Tensor of shape [N, 12, 8, 8]
        self.extras = data['extras']  # Tensor of shape [N, 6]
        self.evals = data['evals']  # Tensor of shape [N, 1]
        self.normalize = normalize
        if self.normalize:
            if norm_params is None:
                self.mean_eval = self.evals.mean().item()
                self.std_eval = self.evals.std().item()
            else:
                self.mean_eval, self.std_eval = norm_params
            # Normalize evaluations
            self.evals = (self.evals - self.mean_eval) / self.std_eval
        else:
            self.mean_eval, self.std_eval = 0, 1

    def __len__(self):
        return self.evals.size(0)

    def __getitem__(self, idx):
        return (self.boards[idx], self.extras[idx]), self.evals[idx]


if __name__ == '__main__':
    precompute_data('data/choppedData.csv', 'data/precomputedData.pt')
