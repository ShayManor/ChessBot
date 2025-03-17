import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def parse_eval(eval_str, mate_constant=1000):
    eval_str = eval_str.strip()
    if '#' in eval_str:
        # Expect formats like "#+2" or "#-3"
        sign = 1 if '+' in eval_str else -1
        moves = int(eval_str.replace('#', '').replace('+', '').replace('-', ''))
        return sign * (mate_constant - moves)
    else:
        return float(eval_str)


piece_to_idx = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}


def compute_king_safety(board_str):
    """
    Computes a simple king safety score by counting the number of friendly pieces
    in the 8 surrounding squares of each king.
    """
    board = []
    for row in board_str.split('/'):
        board_row = []
        for ch in row:
            if ch.isdigit():
                board_row.extend(['.'] * int(ch))
            else:
                board_row.append(ch)
        board.append(board_row)
    # Find white king and black king positions.
    white_king_pos = None
    black_king_pos = None
    for i in range(8):
        for j in range(8):
            if board[i][j] == 'K':
                white_king_pos = (i, j)
            elif board[i][j] == 'k':
                black_king_pos = (i, j)

    def count_adjacent(king_pos, friendly):
        if king_pos is None:
            return 0
        count = 0
        (i, j) = king_pos
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < 8 and 0 <= nj < 8:
                    if board[ni][nj] in friendly:
                        count += 1
        return count

    # For white, consider all uppercase pieces; for black, all lowercase.
    white_safety = count_adjacent(white_king_pos, 'PNBRQK')
    black_safety = count_adjacent(black_king_pos, 'pnbrqk')
    return white_safety, black_safety


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
    Converts FEN into a 776-dimensional tensor:
      - First 768: board pieces (12 channels × 64 squares)
      - Last 8: extra features (turn, castling rights, material advantage, king safety)
    """
    board, active, castling, en_passant, halfmove, fullmove = fen.split(" ")
    tensor = torch.zeros(64 * 12 + 8)
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
    tensor[774], tensor[775] = compute_king_safety(fen)
    return tensor


def process_fen(fen):
    """
    Splits the full tensor into:
      - board_tensor: shape [12, 8, 8]
      - extra_tensor: shape [8]
    """
    full_tensor = fen_to_tensor(fen)
    board_tensor = full_tensor[:768].view(12, 8, 8)
    extra_tensor = full_tensor[768:]
    return board_tensor, extra_tensor


def precompute_data(csv_file, output_file, num_bins=100, chunksize=10000):
    all_boards = []
    all_extras = []
    all_evals = []
    total_rows = 0

    # Process the CSV in chunks and accumulate data in lists.
    for chunk in pd.read_csv(csv_file, chunksize=chunksize):
        for _, row in chunk.iterrows():
            total_rows += 1
            fen = row['FEN']
            board, extra = process_fen(fen)
            all_boards.append(board.unsqueeze(0))
            all_extras.append(extra.unsqueeze(0))
            eval_val = parse_eval(str(row['Evaluation']))
            all_evals.append(eval_val)
        print(f"Processed {total_rows} rows")

    # Concatenate lists into tensors.
    boards = torch.cat(all_boards, dim=0)  # Shape: [N, 12, 8, 8]
    extras = torch.cat(all_extras, dim=0)  # Shape: [N, 8]
    evals = torch.tensor(all_evals, dtype=torch.float).unsqueeze(1)  # Shape: [N, 1]

    # Compute weights using a histogram on the raw eval values.
    evals_np = evals.squeeze(1).cpu().numpy()
    hist, bin_edges = np.histogram(evals_np, bins=num_bins, density=True)
    bin_indices = np.digitize(evals_np, bins=bin_edges, right=True)
    # Avoid division by zero; use a default frequency of 1.0 if needed.
    bin_freq = np.array([hist[i - 1] if i > 0 and (i - 1) < len(hist) else 1.0 for i in bin_indices])
    weights = 1.0 / (bin_freq + 1e-6)
    weights = weights / np.mean(weights)
    weights = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

    # Create a dictionary with all the tensors.
    data = {
        'boards': boards,
        'extras': extras,
        'evals': evals,
        'weights': weights,
    }

    # Save the dictionary using torch.save.
    torch.save(data, output_file)
    print(f"Precomputed tensors saved to {output_file}")

class PrecomputedChessDataset(Dataset):
    def __init__(self, tensor_file, normalize=True, norm_params=None):
        data = torch.load(tensor_file)
        self.boards = data['boards']  # Tensor of shape [N, 12, 8, 8]
        self.extras = data['extras']  # Tensor of shape [N, 6]
        self.evals = data['evals']  # Tensor of shape [N, 1]
        self.weights = data['weights']
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
        return (self.boards[idx], self.extras[idx]), self.evals[idx], self.weights[idx]


if __name__ == '__main__':
    precompute_data('data/tactic_evals.csv', 'data/tactic_precomputedData.pt')
    precompute_data('data/chessData.csv', 'data/precomputedData.pt')
