from idlelib.run import flush_stdout

import pygame
import chess
import sys
import time
from pygame.locals import *
from typing import Tuple, Optional, List

# Import our chess engine
from chess_engine import ChessEngine, positional_eval, aggressive_eval

# Initialize pygame
pygame.init()

# Constants
BOARD_SIZE = 480
SQUARE_SIZE = BOARD_SIZE // 8
INFO_PANEL_WIDTH = 300
WINDOW_WIDTH = BOARD_SIZE + INFO_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_SIZE
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (124, 252, 0, 128)  # Semi-transparent green
SELECTED = (255, 255, 0, 128)  # Semi-transparent yellow
RED = (255, 0, 0)
GREEN = (0, 128, 0)

# Game modes
MODE_PLAYER_VS_AI = 0
MODE_AI_VS_AI = 1
MODE_AI_VS_PLAYER = 2

# Timer constants
BULLET_TIME = 60  # 60 seconds per player for bullet chess


# Load piece images
def load_piece_images() -> dict:
    pieces = {}
    pieces_chars = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']

    for piece in pieces_chars:
        # In a real application, you would have actual images
        # Here we'll create placeholder images with text
        surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))  # Transparent background
        font = pygame.font.Font(None, 48)
        text = font.render(piece, True, BLACK if piece.isupper() else WHITE)
        text_rect = text.get_rect(center=(SQUARE_SIZE // 2, SQUARE_SIZE // 2))
        pygame.draw.circle(surf, WHITE if piece.isupper() else BLACK,
                           (SQUARE_SIZE // 2, SQUARE_SIZE // 2), SQUARE_SIZE // 2 - 5)
        surf.blit(text, text_rect)
        pieces[piece] = surf

    return pieces


class ChessGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess AI")
        self.clock = pygame.time.Clock()
        self.board = chess.Board()

        # Create two different engines
        self.default_engine = ChessEngine(max_depth=4, time_limit=0.5)
        self.positional_engine = ChessEngine(max_depth=4, time_limit=0.5, evaluation_function=positional_eval)
        self.aggressive_engine = ChessEngine(max_depth=4, time_limit=0.5, evaluation_function=aggressive_eval)

        # Choose which engines to use
        self.white_engine = self.positional_engine
        self.black_engine = self.aggressive_engine

        self.piece_images = load_piece_images()
        self.selected_square = None
        self.highlighted_squares = []

        # Game mode (default: human vs AI)
        self.game_mode = MODE_PLAYER_VS_AI
        self.player_color = chess.WHITE

        # Game state
        self.is_game_over = False
        self.result_message = ""

        # Bullet chess timers (in seconds)
        self.white_time = BULLET_TIME
        self.black_time = BULLET_TIME
        self.last_move_time = time.time()
        self.clock_active = False

        # Game status
        self.status_message = "Game ready to start"
        self.move_list = []

        # Background thinking flag
        self.background_thinking_active = False

    def draw_board(self):
        """Draw the chess board."""
        for row in range(8):
            for col in range(8):
                x, y = col * SQUARE_SIZE, row * SQUARE_SIZE
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

                # Highlight selected square
                if self.selected_square is not None:
                    selected_col, selected_row = self.selected_square
                    if row == selected_row and col == selected_col:
                        highlight_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                        highlight_surf.fill(SELECTED)
                        self.screen.blit(highlight_surf, (x, y))

                # Highlight possible moves
                if (col, row) in self.highlighted_squares:
                    highlight_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    highlight_surf.fill(HIGHLIGHT)
                    self.screen.blit(highlight_surf, (x, y))

    def draw_pieces(self):
        """Draw the chess pieces on the board."""
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)  # Convert to chess.Square
                piece = self.board.piece_at(square)
                if piece:
                    piece_char = piece.symbol()
                    if piece_char in self.piece_images:
                        self.screen.blit(self.piece_images[piece_char],
                                         (col * SQUARE_SIZE, row * SQUARE_SIZE))

    def draw_info_panel(self):
        """Draw the information panel."""
        # Draw panel background
        pygame.draw.rect(self.screen, LIGHT_GRAY,
                         (BOARD_SIZE, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))

        # Draw timers
        font = pygame.font.Font(None, 36)

        # White timer
        white_timer_text = font.render(
            f"White: {int(self.white_time // 60)}:{int(self.white_time % 60):02d}",
            True, BLACK)
        self.screen.blit(white_timer_text, (BOARD_SIZE + 20, 20))

        # Black timer
        black_timer_text = font.render(
            f"Black: {int(self.black_time // 60)}:{int(self.black_time % 60):02d}",
            True, BLACK)
        self.screen.blit(black_timer_text, (BOARD_SIZE + 20, 60))

        # Current player
        current_player = "White" if self.board.turn == chess.WHITE else "Black"
        turn_text = font.render(f"Turn: {current_player}", True, BLACK)
        self.screen.blit(turn_text, (BOARD_SIZE + 20, 100))

        # Game mode
        mode_text = ""
        if self.game_mode == MODE_PLAYER_VS_AI:
            mode_text = "Player vs AI"
        elif self.game_mode == MODE_AI_VS_PLAYER:
            mode_text = "AI vs Player"
        else:
            mode_text = "AI vs AI"
        mode_label = font.render(f"Mode: {mode_text}", True, BLACK)
        self.screen.blit(mode_label, (BOARD_SIZE + 20, 140))

        # Engine information
        small_font = pygame.font.Font(None, 24)
        # --- CHANGED HERE: show each engine's current depth ---
        white_depth_text = f"White: Positional AI (Depth: {self.white_engine.current_depth})"
        white_engine = small_font.render(white_depth_text, True, BLACK)
        self.screen.blit(white_engine, (BOARD_SIZE + 20, 180))

        black_depth_text = f"Black: Aggressive AI (Depth: {self.black_engine.current_depth})"
        black_engine = small_font.render(black_depth_text, True, BLACK)
        self.screen.blit(black_engine, (BOARD_SIZE + 20, 210))
        # ------------------------------------------------------

        # Status message
        status_font = pygame.font.Font(None, 24)
        status_text = status_font.render(self.status_message, True, BLACK)
        self.screen.blit(status_text, (BOARD_SIZE + 20, WINDOW_HEIGHT - 60))

        # Game result if game is over
        if self.is_game_over:
            result_font = pygame.font.Font(None, 36)
            result_text = result_font.render(self.result_message, True, RED)
            self.screen.blit(result_text, (BOARD_SIZE + 20, WINDOW_HEIGHT - 100))

        # Background thinking indicator
        if self.background_thinking_active:
            thinking_text = small_font.render("Background analysis active", True, GREEN)
            self.screen.blit(thinking_text, (BOARD_SIZE + 20, WINDOW_HEIGHT - 30))

    def screen_to_board_coords(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert screen coordinates to board coordinates."""
        x, y = pos
        # Check if click is within the board area
        if x >= BOARD_SIZE or y >= BOARD_SIZE:
            return None

        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        return col, row

    def board_coords_to_square(self, coords: Tuple[int, int]) -> chess.Square:
        """Convert board coordinates to chess square."""
        col, row = coords
        return chess.square(col, 7 - row)  # Convert to chess.Square

    def get_possible_moves(self, square: chess.Square) -> List[Tuple[int, int]]:
        """Get all possible moves for a piece on the given square."""
        moves = []
        for move in self.board.legal_moves:
            if move.from_square == square:
                to_col = chess.square_file(move.to_square)
                to_row = 7 - chess.square_rank(move.to_square)  # Invert rank for display
                moves.append((to_col, to_row))
        return moves

    def make_move(self, from_square: chess.Square, to_square: chess.Square) -> bool:
        """Make a move on the board if it's legal."""
        # Check for promotion
        is_promotion = False
        piece = self.board.piece_at(from_square)

        if piece and piece.piece_type == chess.PAWN:
            if (chess.square_rank(to_square) == 0 and self.board.turn == chess.WHITE) or \
                    (chess.square_rank(to_square) == 7 and self.board.turn == chess.BLACK):
                is_promotion = True

        # Create the move
        if is_promotion:
            move = chess.Move(from_square, to_square, promotion=chess.QUEEN)  # Always promote to queen
        else:
            move = chess.Move(from_square, to_square)

        # Check if move is legal
        if move in self.board.legal_moves:
            # If this is a player's first move, start the clock
            if not self.clock_active:
                self.clock_active = True
                self.last_move_time = time.time()

            # Update timers for the player who just moved
            self.update_timers()

            # Make the move
            print(move)
            flush_stdout()
            # Record move in move list
            self.move_list.append(self.board.san(move))
            self.board.push(move)

            # Update last move time
            self.last_move_time = time.time()

            # Start background thinking for the relevant engine
            if self.board.turn == chess.WHITE:
                if self.game_mode == MODE_PLAYER_VS_AI and self.player_color == chess.BLACK:
                    self.white_engine.start_background_thinking(self.board)
                    self.background_thinking_active = True
                elif self.game_mode == MODE_AI_VS_AI:
                    self.white_engine.start_background_thinking(self.board)
                    self.background_thinking_active = True
            else:
                if self.game_mode == MODE_PLAYER_VS_AI and self.player_color == chess.WHITE:
                    self.black_engine.start_background_thinking(self.board)
                    self.background_thinking_active = True
                elif self.game_mode == MODE_AI_VS_AI:
                    self.black_engine.start_background_thinking(self.board)
                    self.background_thinking_active = True

            return True
        return False

    def ai_move(self, is_white: bool):
        """Make a move for the AI."""
        self.status_message = f"{'White' if is_white else 'Black'} AI is thinking..."

        if self.background_thinking_active:
            if is_white:
                self.white_engine.stop_background_thinking()
            else:
                self.black_engine.stop_background_thinking()
            self.background_thinking_active = False

        # Get the move from the appropriate engine
        engine = self.white_engine if is_white else self.black_engine
        move = engine.get_best_move(self.board)

        if move:
            self.update_timers()

            # Start the clock if this is the first move
            if not self.clock_active:
                self.clock_active = True

            # Make move
            self.move_list.append(self.board.san(move))
            self.board.push(move)

            self.last_move_time = time.time()

            # Start background thinking for the next player's engine
            if self.board.turn == chess.WHITE:
                if self.game_mode == MODE_PLAYER_VS_AI and self.player_color == chess.BLACK:
                    self.white_engine.start_background_thinking(self.board)
                    self.background_thinking_active = True
                elif self.game_mode == MODE_AI_VS_AI:
                    self.white_engine.start_background_thinking(self.board)
                    self.background_thinking_active = True
            else:
                if self.game_mode == MODE_PLAYER_VS_AI and self.player_color == chess.WHITE:
                    self.black_engine.start_background_thinking(self.board)
                    self.background_thinking_active = True
                elif self.game_mode == MODE_AI_VS_AI:
                    self.black_engine.start_background_thinking(self.board)
                    self.background_thinking_active = True

            self.status_message = f"{'White' if is_white else 'Black'} AI moved: {move}"

    def update_timers(self):
        """Update the chess clock."""
        if self.clock_active:
            current_time = time.time()
            elapsed = current_time - self.last_move_time

            if self.board.turn == chess.WHITE:
                # Black just moved, update black's timer
                self.black_time -= elapsed
                if self.black_time <= 0:
                    self.black_time = 0
                    self.is_game_over = True
                    self.result_message = "White wins on time!"
            else:
                # White just moved, update white's timer
                self.white_time -= elapsed
                if self.white_time <= 0:
                    self.white_time = 0
                    self.is_game_over = True
                    self.result_message = "Black wins on time!"

    def check_game_over(self):
        """Check if the game is over."""
        if self.board.is_game_over():
            self.is_game_over = True
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn == chess.WHITE else "White"
                self.result_message = f"{winner} wins by checkmate!"
            elif self.board.is_stalemate():
                self.result_message = "Draw by stalemate"
            elif self.board.is_insufficient_material():
                self.result_message = "Draw by insufficient material"
            elif self.board.is_fifty_moves():
                self.result_message = "Draw by fifty-move rule"
            elif self.board.is_repetition():
                self.result_message = "Draw by repetition"
            else:
                self.result_message = "Game over"

            # Stop background thinking
            if self.background_thinking_active:
                if self.board.turn == chess.WHITE:
                    self.white_engine.stop_background_thinking()
                else:
                    self.black_engine.stop_background_thinking()
                self.background_thinking_active = False

            return True
        return False

    def toggle_game_mode(self):
        """Toggle between game modes."""
        if self.game_mode == MODE_PLAYER_VS_AI:
            self.game_mode = MODE_AI_VS_AI
            self.status_message = "Mode: AI vs AI"
        elif self.game_mode == MODE_AI_VS_AI:
            self.game_mode = MODE_AI_VS_PLAYER
            self.player_color = chess.BLACK
            self.status_message = "Mode: AI vs Player"
        else:
            self.game_mode = MODE_PLAYER_VS_AI
            self.player_color = chess.WHITE
            self.status_message = "Mode: Player vs AI"

    def restart_game(self):
        """Restart the game."""
        # Stop background thinking if active
        if self.background_thinking_active:
            if self.board.turn == chess.WHITE:
                self.white_engine.stop_background_thinking()
            else:
                self.black_engine.stop_background_thinking()
            self.background_thinking_active = False

        # Reset game state
        self.board = chess.Board()
        self.is_game_over = False
        self.result_message = ""
        self.selected_square = None
        self.highlighted_squares = []
        self.white_time = BULLET_TIME
        self.black_time = BULLET_TIME
        self.last_move_time = time.time()
        self.clock_active = False
        self.move_list = []
        self.status_message = "Game restarted"

        # Make first move if AI is playing white
        if (self.game_mode == MODE_AI_VS_AI) or \
                (self.game_mode == MODE_AI_VS_PLAYER and self.player_color == chess.BLACK):
            self.ai_move(True)  # White AI's move

    def run(self):
        """Main game loop."""
        running = True

        # If AI plays white in appropriate modes, make first move
        if (self.game_mode == MODE_AI_VS_AI) or \
                (self.game_mode == MODE_AI_VS_PLAYER and self.player_color == chess.BLACK):
            self.ai_move(True)  # White AI's move

        # Start background thinking for the player
        if self.game_mode == MODE_PLAYER_VS_AI and self.player_color == chess.WHITE:
            self.black_engine.start_background_thinking(self.board)
            self.background_thinking_active = True

        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

                if event.type == KEYDOWN:
                    if event.key == K_m:  # 'M' key to toggle game mode
                        self.toggle_game_mode()
                    elif event.key == K_r:  # 'R' key to restart
                        self.restart_game()

                if event.type == MOUSEBUTTONDOWN and event.button == 1:  # Left click
                    pos = pygame.mouse.get_pos()
                    board_coords = self.screen_to_board_coords(pos)

                    # If click is outside the board, ignore
                    if not board_coords:
                        continue

                    # Don't process clicks if game is over
                    if self.is_game_over:
                        continue

                    # Don't process clicks in AI vs AI mode
                    if self.game_mode == MODE_AI_VS_AI:
                        continue

                    # Only process clicks when it's the player's turn
                    is_player_turn = (self.board.turn == self.player_color)
                    if not is_player_turn:
                        continue

                    col, row = board_coords

                    if self.selected_square is None:
                        # Select a square with a piece of the player's color
                        square = self.board_coords_to_square((col, row))
                        piece = self.board.piece_at(square)
                        if piece and piece.color == self.player_color:
                            self.selected_square = (col, row)
                            self.highlighted_squares = self.get_possible_moves(square)
                    else:
                        # Try to move the selected piece
                        from_square = self.board_coords_to_square(self.selected_square)
                        to_square = self.board_coords_to_square((col, row))

                        # If clicking on a highlighted square, make the move
                        if (col, row) in self.highlighted_squares:
                            # Stop background thinking
                            if self.background_thinking_active:
                                if self.board.turn == chess.WHITE:
                                    self.black_engine.stop_background_thinking()
                                else:
                                    self.white_engine.stop_background_thinking()
                                self.background_thinking_active = False

                            if self.make_move(from_square, to_square):
                                # Check if game is over after player's move
                                if not self.check_game_over():
                                    # If game is not over and it's AI's turn, make AI move
                                    if self.game_mode == MODE_PLAYER_VS_AI:
                                        self.ai_move(not self.player_color)
                                        # Check if game is over after AI's move
                                        self.check_game_over()

                        # Clear selection
                        self.selected_square = None
                        self.highlighted_squares = []

            # If in AI vs AI mode and game not over, let AIs play
            if self.game_mode == MODE_AI_VS_AI and not self.is_game_over:
                # Only make move after a short delay to visualize
                current_time = time.time()
                if current_time - self.last_move_time > 1.0:  # 1 second delay
                    self.ai_move(self.board.turn == chess.WHITE)
                    self.check_game_over()

            # Update timing
            if self.clock_active and not self.is_game_over:
                current_time = time.time()
                elapsed = current_time - self.last_move_time

                if self.board.turn == chess.WHITE:
                    display_time = max(0, self.white_time - elapsed)
                    if display_time <= 0 and self.white_time > 0:
                        self.white_time = 0
                        self.is_game_over = True
                        self.result_message = "Black wins on time!"
                else:
                    display_time = max(0, self.black_time - elapsed)
                    if display_time <= 0 and self.black_time > 0:
                        self.black_time = 0
                        self.is_game_over = True
                        self.result_message = "White wins on time!"

            # Draw everything
            self.draw_board()
            self.draw_pieces()
            self.draw_info_panel()
            pygame.display.flip()
            self.clock.tick(FPS)

        # Cleanup when exiting
        if self.background_thinking_active:
            if self.board.turn == chess.WHITE:
                self.white_engine.stop_background_thinking()
            else:
                self.black_engine.stop_background_thinking()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = ChessGame()
    game.run()
