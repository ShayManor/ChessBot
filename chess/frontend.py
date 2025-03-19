import pygame
import chess
import sys
import os
from pygame.locals import *
from typing import Tuple, Optional, List

# Import our chess engine
from chess_engine import ChessEngine

pygame.init()

BOARD_SIZE = 480
SQUARE_SIZE = BOARD_SIZE // 8
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (124, 252, 0, 128)  # Semi-transparent green
SELECTED = (255, 255, 0, 128)  # Semi-transparent yellow


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
        self.screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
        pygame.display.set_caption("Chess AI")
        self.clock = pygame.time.Clock()
        self.board = chess.Board()
        self.engine = ChessEngine(max_depth=4, time_limit=0.5)  # Adjust parameters as needed
        self.piece_images = load_piece_images()
        self.selected_square = None
        self.highlighted_squares = []
        self.player_color = chess.WHITE  # Human plays white by default

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

    def screen_to_board_coords(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert screen coordinates to board coordinates."""
        x, y = pos
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
        move = chess.Move(from_square, to_square)

        # Check for promotion
        if (self.board.piece_at(from_square).piece_type == chess.PAWN and
                ((to_square // 8 == 0 and self.board.turn == chess.WHITE) or
                 (to_square // 8 == 7 and self.board.turn == chess.BLACK))):
            move.promotion = chess.QUEEN  # Always promote to queen for simplicity

        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False

    def ai_move(self):
        """Make a move for the AI."""
        move = self.engine.get_best_move(self.board)
        if move:
            self.board.push(move)

    def run(self):
        """Main game loop."""
        running = True

        # AI is black
        if self.player_color == chess.BLACK:
            self.ai_move()

        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

                if event.type == MOUSEBUTTONDOWN and event.button == 1:
                    pos = pygame.mouse.get_pos()
                    col, row = self.screen_to_board_coords(pos)

                    if self.selected_square is None:
                        square = self.board_coords_to_square((col, row))
                        piece = self.board.piece_at(square)
                        if piece and piece.color == self.player_color:
                            self.selected_square = (col, row)
                            self.highlighted_squares = self.get_possible_moves(square)
                    else:
                        from_square = self.board_coords_to_square(self.selected_square)
                        to_square = self.board_coords_to_square((col, row))

                        if (col, row) in self.highlighted_squares:
                            if self.make_move(from_square, to_square):
                                if not self.board.is_game_over():
                                    self.ai_move()

                        self.selected_square = None
                        self.highlighted_squares = []

            self.draw_board()
            self.draw_pieces()
            pygame.display.flip()
            self.clock.tick(FPS)

            if self.board.is_game_over():
                result = "1-0" if self.board.is_checkmate() and not self.board.turn else "0-1" if self.board.is_checkmate() else "1/2-1/2"
                print(f"Game over. Result: {result}")
                pygame.time.wait(5000)
                running = False

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = ChessGame()
    game.run()
