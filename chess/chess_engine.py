import random
import chess
import time
import threading
from typing import Tuple, Optional, Dict, List, Callable


class ChessEngine:
    def __init__(self, max_depth: int = 4, time_limit: float = 0.8, evaluation_function: Optional[Callable] = None):
        self.max_depth = max_depth
        self.current_depth = 0
        self.time_limit = time_limit
        self.transposition_table: Dict[str, Tuple[float, int, chess.Move]] = {}  # Position cache
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in
                                                               range(100)]  # Killer move heuristic
        self.history_table: Dict[Tuple[chess.Square, chess.Square], int] = {}  # History heuristic
        self.background_thinking = False
        self.thinking_thread = None
        self.background_best_move = None
        self.background_board_fen = None
        self.stop_thinking = False

        self.evaluation_function = evaluation_function or self.default_evaluate

        # Opening moves (hardcoded)
        self.opening_book = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": [  # Starting position
                chess.Move.from_uci("e2e4"),  # e4
                chess.Move.from_uci("d2d4"),  # d4
                chess.Move.from_uci("c2c4"),  # c4
                chess.Move.from_uci("g1f3"),  # Nf3
            ],
            # Response to e4
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": [
                chess.Move.from_uci("e7e5"),  # e5
                chess.Move.from_uci("c7c5"),  # c5 (Sicilian)
                chess.Move.from_uci("e7e6"),  # e6 (French)
            ],
            # Response to d4
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1": [
                chess.Move.from_uci("d7d5"),  # d5
                chess.Move.from_uci("g8f6"),  # Nf6 (Indian defenses)
                chess.Move.from_uci("f7f5"),  # f5 (Dutch)
            ],
            # Response to c4
            "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1": [
                chess.Move.from_uci("e7e5"),  # e5
                chess.Move.from_uci("c7c5"),  # c5
                chess.Move.from_uci("g8f6"),  # Nf6
            ],
            # Response to Nf3
            "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 0 1": [
                chess.Move.from_uci("d7d5"),  # d5
                chess.Move.from_uci("g8f6"),  # Nf6
                chess.Move.from_uci("c7c5"),  # c5
            ],
        }
    # Simple default evaluation
    def default_evaluate(self, board: chess.Board) -> float:
        """Simpl default evaluation function - material count."""
        if board.is_checkmate():
            return -10000 if board.turn else 10000

        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }

        material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                material += value if piece.color == chess.WHITE else -value

        # Bonus for controlling center
        center_control = 0
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for square in center_squares:
            if board.piece_at(square):
                piece = board.piece_at(square)
                center_control += 10 if piece.color == chess.WHITE else -10

        # Bonus for mobility
        mobility = len(list(board.legal_moves))
        board.push(chess.Move.null())
        opponent_mobility = len(list(board.legal_moves))
        board.pop()
        mobility_score = (mobility - opponent_mobility) * 5

        # Total score
        score = material + center_control + mobility_score

        return score if board.turn == chess.WHITE else -score

    def evaluate(self, board: chess.Board) -> float:
        """Evaluate the position using the configured evaluation function."""
        return self.evaluation_function(board)

    def get_best_move(self, board: chess.Board) -> chess.Move:
        """Find the best move using iterative deepening."""
        # Check if we have a book move
        board_fen = board.fen()
        if board_fen in self.opening_book:
            book_moves = self.opening_book[board_fen]
            valid_moves = [move for move in book_moves if move in board.legal_moves]
            if valid_moves:
                return random.choice(valid_moves)

        # Check if we already calculated this position in the background
        if self.background_best_move and self.background_board_fen == board_fen:
            return self.background_best_move

        # If background thinking is running, stop it
        self.stop_background_thinking()

        best_move = None
        start_time = time.time()

        self.history_table.clear()
        for from_square in range(64):
            for to_square in range(64):
                self.history_table[(from_square, to_square)] = 0

        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            if time.time() - start_time > self.time_limit:
                break
            self.current_depth += 1

            # Clear killer moves for this iteration
            self.killer_moves = [[None, None] for _ in range(100)]

            value, move = self.negamax(board, depth, float('-inf'), float('inf'), 1, 0)

            if move is not None:
                best_move = move

            # If we found a winning move, no need to search deeper
            if value > 9000:
                break

        return best_move if best_move else self._get_random_move(board)

    def start_background_thinking(self, board: chess.Board):
        """Start thinking in background for the current position."""
        if self.thinking_thread and self.thinking_thread.is_alive():
            self.stop_thinking = True
            self.thinking_thread.join()

        self.stop_thinking = False
        self.background_board_fen = board.fen()
        self.background_best_move = None
        self.thinking_thread = threading.Thread(target=self._background_search, args=(board.copy(),))
        self.thinking_thread.daemon = True
        self.thinking_thread.start()
        self.background_thinking = True

    def stop_background_thinking(self):
        """Stop background thinking."""
        if self.thinking_thread and self.thinking_thread.is_alive():
            self.stop_thinking = True
            self.thinking_thread.join()
        self.background_thinking = False

    def _background_search(self, board: chess.Board):
        """Search for the best move in the background."""
        best_move = None
        self.history_table.clear()

        for from_square in range(64):
            for to_square in range(64):
                self.history_table[(from_square, to_square)] = 0

        # Iterative deepening with no time limit
        for depth in range(1, self.max_depth + 5):  # Can go deeper than normal search
            if self.stop_thinking:
                return

            # Clear killer moves for this iteration
            self.killer_moves = [[None, None] for _ in range(100)]

            value, move = self.negamax(board, depth, float('-inf'), float('inf'), 1, 0, is_background=True)

            if move is not None:
                best_move = move
                self.background_best_move = move

            # If we found a mate, done
            if value > 9000:
                break

    def _get_random_move(self, board: chess.Board) -> chess.Move:
        """Return a random legal move if no best move is found."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        return random.choice(legal_moves)

    def _order_moves(self, board: chess.Board, depth: int) -> List[chess.Move]:
        """Order moves to improve alpha-beta pruning efficiency."""
        moves = list(board.legal_moves)
        scored_moves = []

        for move in moves:
            score = 0

            # Transposition table, not sure what this does
            board_hash = board.fen()
            if board_hash in self.transposition_table:
                _, _, tt_move = self.transposition_table[board_hash]
                if tt_move == move:
                    score += 10000

            # Killer move heuristic
            if self.killer_moves[depth][0] == move:
                score += 900  # update this value later
            elif self.killer_moves[depth][1] == move:
                score += 800  # less because opponent could avoid

            # History heuristic
            from_square = move.from_square
            to_square = move.to_square
            score += self.history_table.get((from_square, to_square), 0)

            # Captures
            if board.is_capture(move):
                # MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
                victim = board.piece_type_at(move.to_square)
                aggressor = board.piece_type_at(move.from_square)
                if victim and aggressor:
                    score += 10 * victim - aggressor

            # Promotions
            if move.promotion:
                score += 500 + move.promotion

            # Check extensions
            board.push(move)
            if board.is_check():
                score += 100
            board.pop()

            scored_moves.append((score, move))

        # Sort moves by score desc
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored_moves]

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float,
                color: int, ply: int, is_background: bool = False) -> Tuple[float, Optional[chess.Move]]:
        """
        Negamax algorithm with alpha-beta pruning.

        Args:
            board: Current board state
            depth: Current search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            color: 1 for maximizing player, -1 for minimizing player
            ply: Current ply (half-move) in search tree
            is_background: Whether this is a background search

        Returns:
            Tuple of (best score, best move)
        """
        if is_background and self.stop_thinking:
            return 0, None

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
            return color * self.evaluate(board), None

        best_move = None
        best_value = float('-inf')

        moves = self._order_moves(board, ply)

        for move in moves:
            # Check if background search should stop
            if is_background and self.stop_thinking:
                return 0, None

            board.push(move)
            # Figure out how this works
            value, _ = self.negamax(board, depth - 1, -beta, -alpha, -color, ply + 1, is_background)
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

                # Update history
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


# Other evaluations (later use different models)
def positional_eval(board: chess.Board) -> float:
    """Positional evaluation that emphasizes development, center control, and king safety."""
    if board.is_checkmate():
        return -10000 if board.turn else 10000

    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }

    # Calculate material balance
    material = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            material += value if piece.color == chess.WHITE else -value

    # Piece-square tables for positional evaluation
    pawn_table = [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ]

    knight_table = [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    ]

    bishop_table = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 5, 5, 5, 5, -10,
        -10, 0, 5, 0, 0, 5, 0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ]

    rook_table = [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0
    ]

    queen_table = [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    ]

    king_middle_table = [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20
    ]

    king_end_table = [
        -50, -40, -30, -20, -20, -30, -40, -50,
        -30, -20, -10, 0, 0, -10, -20, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -30, 0, 0, 0, 0, -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50
    ]

    # Calculate piece-square table bonus
    position_bonus = 0

    # Count material to determine game phase (middlegame or endgame)
    total_material = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            total_material += piece_values[piece.piece_type]

    is_endgame = total_material < 3000  # Arbitrary threshold

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue

        # Get piece position value based on piece type
        piece_type = piece.piece_type
        sq = square if piece.color == chess.WHITE else chess.square_mirror(square)

        if piece_type == chess.PAWN:
            position_bonus += pawn_table[sq] if piece.color == chess.WHITE else -pawn_table[sq]
        elif piece_type == chess.KNIGHT:
            position_bonus += knight_table[sq] if piece.color == chess.WHITE else -knight_table[sq]
        elif piece_type == chess.BISHOP:
            position_bonus += bishop_table[sq] if piece.color == chess.WHITE else -bishop_table[sq]
        elif piece_type == chess.ROOK:
            position_bonus += rook_table[sq] if piece.color == chess.WHITE else -rook_table[sq]
        elif piece_type == chess.QUEEN:
            position_bonus += queen_table[sq] if piece.color == chess.WHITE else -queen_table[sq]
        elif piece_type == chess.KING:
            if is_endgame:
                position_bonus += king_end_table[sq] if piece.color == chess.WHITE else -king_end_table[sq]
            else:
                position_bonus += king_middle_table[sq] if piece.color == chess.WHITE else -king_middle_table[sq]

    # Bonus for piece development
    development = 0
    if not board.piece_at(chess.B1) and board.turn == chess.WHITE:
        development += 10  # Knight developed
    if not board.piece_at(chess.G1) and board.turn == chess.WHITE:
        development += 10  # Knight developed
    if not board.piece_at(chess.C1) and board.turn == chess.WHITE:
        development += 10  # Bishop developed
    if not board.piece_at(chess.F1) and board.turn == chess.WHITE:
        development += 10  # Bishop developed

    if not board.piece_at(chess.B8) and board.turn == chess.BLACK:
        development -= 10  # Knight developed
    if not board.piece_at(chess.G8) and board.turn == chess.BLACK:
        development -= 10  # Knight developed
    if not board.piece_at(chess.C8) and board.turn == chess.BLACK:
        development -= 10  # Bishop developed
    if not board.piece_at(chess.F8) and board.turn == chess.BLACK:
        development -= 10  # Bishop developed

    # Penalty for undeveloped pieces in middlegame
    if not is_endgame:
        if board.piece_at(chess.D1) and board.piece_at(
                chess.D1).piece_type == chess.QUEEN and board.turn == chess.WHITE:
            development -= 5  # Queen not developed
        if board.piece_at(chess.D8) and board.piece_at(
                chess.D8).piece_type == chess.QUEEN and board.turn == chess.BLACK:
            development += 5  # Queen not developed

    # Mobility
    mobility = len(list(board.legal_moves))
    board.push(chess.Move.null())
    opponent_mobility = len(list(board.legal_moves))
    board.pop()
    mobility_score = (mobility - opponent_mobility) * 5

    # Total score
    score = material + position_bonus + development + mobility_score

    # Return score from white's perspective
    return score if board.turn == chess.WHITE else -score


def aggressive_eval(board: chess.Board) -> float:
    """Aggressive evaluation that emphasizes attacks and piece activity."""
    if board.is_checkmate():
        return -10000 if board.turn else 10000

    # Material values with higher value for attacking pieces
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 350,  # Knights valued more for their tactical ability
        chess.BISHOP: 350,  # Bishops valued more for their range
        chess.ROOK: 525,
        chess.QUEEN: 1000,  # Queens valued more for attack power
        chess.KING: 0
    }

    # Calculate material balance
    material = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            material += value if piece.color == chess.WHITE else -value

    # Bonus for checks and attacks
    attack_bonus = 0
    if board.is_check():
        attack_bonus = 50  # Bonus for giving check

    # Count attacked squares
    attacked_squares = 0
    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            attacked_squares += 1
        if board.is_attacked_by(chess.BLACK, square):
            attacked_squares -= 1

    attack_bonus += attacked_squares * 5

    # Bonus for forward piece positioning
    forward_bonus = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue

        rank = chess.square_rank(square)
        if piece.color == chess.WHITE:
            # White pieces get bonus for being on opponent's half
            if rank >= 4:  # Ranks 5-8
                forward_bonus += 5 * (rank - 3)  # More bonus for deeper penetration
        else:
            # Black pieces get bonus for being on opponent's half
            if rank <= 3:  # Ranks 1-4
                forward_bonus += 5 * (4 - rank)

    # Pawn storms - pawns advancing toward enemy king
    pawn_storm = 0
    white_king_file = chess.square_file(board.king(chess.WHITE))
    black_king_file = chess.square_file(board.king(chess.BLACK))

    for file in range(8):
        # Bonus for pawns close to opponent's king file
        file_distance_to_white_king = abs(file - white_king_file)
        file_distance_to_black_king = abs(file - black_king_file)

        # Check white pawns
        for rank in range(1, 8):  # Skip the first rank
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE and file_distance_to_black_king <= 2:
                    pawn_storm += 5 * rank  # More bonus for advanced pawns
                elif piece.color == chess.BLACK and file_distance_to_white_king <= 2:
                    pawn_storm -= 5 * (7 - rank)

    # Bonus for piece coordination
    coordination = 0
    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            # Count how many white pieces attack this square
            attacker_count = 0
            for attacker_square in chess.SQUARES:
                attacker = board.piece_at(attacker_square)
                if attacker and attacker.color == chess.WHITE:
                    if board.is_legal(chess.Move(attacker_square, square)):
                        attacker_count += 1

            if attacker_count > 1:
                coordination += 5 * attacker_count

        if board.is_attacked_by(chess.BLACK, square):
            # Count how many black pieces attack this square
            attacker_count = 0
            for attacker_square in chess.SQUARES:
                attacker = board.piece_at(attacker_square)
                if attacker and attacker.color == chess.BLACK:
                    if board.is_legal(chess.Move(attacker_square, square)):
                        attacker_count += 1

            if attacker_count > 1:
                coordination -= 5 * attacker_count

    # Mobility
    mobility = len(list(board.legal_moves))
    board.push(chess.Move.null())
    opponent_mobility = len(list(board.legal_moves))
    board.pop()
    mobility_score = (mobility - opponent_mobility) * 8  # Higher weight on mobility

    # Total score
    score = material + attack_bonus + forward_bonus + pawn_storm + coordination + mobility_score

    # Return score from white's perspective
    return score if board.turn == chess.WHITE else -score