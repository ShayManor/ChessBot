import random

import chess
import time
from typing import Tuple, Optional, Dict, List


class ChessEngine:
    def __init__(self, max_depth: int = 4, time_limit: float = 0.8):
        self.max_depth = max_depth  # Maximum search depth
        self.time_limit = time_limit  # Time limit for move calculation in seconds
        self.transposition_table: Dict[str, Tuple[float, int, chess.Move]] = {}  # Position cache
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in
                                                               range(100)]  # Killer move heuristic
        self.history_table: Dict[Tuple[chess.Square, chess.Square], int] = {}  # History heuristic

    def evaluate(self, fen: str) -> float:
        # Placeholder, will call evaluation function
        return 0.0

    def get_best_move(self, board: chess.Board) -> chess.Move:
        """Find the best move using iterative deepening."""
        best_move = None
        start_time = time.time()

        self.history_table.clear()
        for from_square in range(64):
            for to_square in range(64):
                self.history_table[(from_square, to_square)] = 0

        for depth in range(1, self.max_depth + 1):
            if time.time() - start_time > self.time_limit:
                break

            self.killer_moves = [[None, None] for _ in range(100)]

            value, move = self.negamax(board, depth, float('-inf'), float('inf'), 1, 0)

            if move is not None:
                best_move = move

            if value > 9000:
                break

        return best_move if best_move else self._get_random_move(board)

    def _get_random_move(self, board: chess.Board) -> chess.Move:
        """Return a random legal move if no best move is found."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        return legal_moves[random.randint(0, len(list(board.legal_moves)))]

    def _order_moves(self, board: chess.Board, depth: int) -> List[chess.Move]:
        """Order moves to improve alpha-beta pruning efficiency."""
        moves = list(board.legal_moves)
        scored_moves = []

        for move in moves:
            score = 0

            board_hash = board.fen()
            if board_hash in self.transposition_table:
                _, _, tt_move = self.transposition_table[board_hash]
                if tt_move == move:
                    score += 10000

            # Killer move heuristic
            if self.killer_moves[depth][0] == move:
                score += 900
            elif self.killer_moves[depth][1] == move:
                score += 800

            # History heuristic
            from_square = move.from_square
            to_square = move.to_square
            score += self.history_table.get((from_square, to_square), 0)

            # Captures
            if board.is_capture(move):
                # MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
                victim = board.piece_type_at(to_square)
                aggressor = board.piece_type_at(from_square)
                if victim and aggressor:
                    score += 10 * victim - aggressor

            if move.promotion:
                score += 500 + move.promotion

            # Check extensions
            board.push(move)
            if board.is_check():
                score += 100
            board.pop()

            scored_moves.append((score, move))

        # Sort moves by score in descending order
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored_moves]

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float,
                color: int, ply: int) -> Tuple[float, Optional[chess.Move]]:
        """
        Negamax algorithm with alpha-beta pruning.

        Args:
            board: Current board state
            depth: Current search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            color: 1 for maximizing player, -1 for minimizing player
            ply: Current ply (half-move) in search tree

        Returns:
            Tuple of (best score, best move)
        """
        alpha_orig = alpha
        board_hash = board.fen()

        # Check transposition table
        if board_hash in self.transposition_table:
            tt_value, tt_depth, tt_move = self.transposition_table[board_hash]
            if tt_depth >= depth:
                return tt_value, tt_move

        # Check for game over
        if board.is_game_over():
            if board.is_checkmate():
                return -10000 + ply, None  # Prefer shorter paths to checkmate
            return 0, None  # Draw

        # Evaluate leaf nodes
        if depth == 0:
            return color * self.evaluate(board.fen()), None

        best_move = None
        best_value = float('-inf')

        # Get ordered moves
        moves = self._order_moves(board, ply)

        # Search all moves
        for move in moves:
            board.push(move)
            value, _ = self.negamax(board, depth - 1, -beta, -alpha, -color, ply + 1)
            value = -value
            board.pop()

            if value > best_value:
                best_value = value
                best_move = move

            if best_value > alpha:
                alpha = best_value

            # Beta cutoff
            if alpha >= beta:
                # Store killer move
                if not board.is_capture(move):
                    if self.killer_moves[ply][0] != move:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move

                # Update history table
                if not board.is_capture(move):
                    from_square = move.from_square
                    to_square = move.to_square
                    self.history_table[(from_square, to_square)] += depth * depth

                break

        # Store in transposition table
        self.transposition_table[board_hash] = (best_value, depth, best_move)

        return best_value, best_move

    def update_opponent_move(self, board: chess.Board, move: chess.Move):
        """Update the board with the opponent's move."""
        if move in board.legal_moves:
            board.push(move)
            return True
        return False