# Othello AI Engine

This project is a complete Othello (Reversi) game featuring a custom-built GUI and a formidable AI opponent. It supports both single-player (vs. AI) and two-player modes. The application is written in Python, leveraging the Tkinter library for the user interface.

## Key Features

- **Interactive GUI**: A clean, responsive game board that provides several real-time visual aids for an insightful gameplay experience.
- **Legal Move Highlighting**: Visually indicates all valid moves for the current player and displays the number of pieces each move would flip.
- **Real-time Evaluation Bar**: A vertical bar, inspired by modern chess engines, provides a visual summary of the current positional advantage.
- **AI Transparency**: In single-player mode, a list of the AI's top-rated moves and their heuristic scores is displayed, offering a direct window into its decision-making process.

## The AI's Strategy

The AI's intelligence is driven by a **Negamax search algorithm** (a variant of Minimax) with **alpha-beta pruning**. This allows it to efficiently search the tree of possible future moves to find the optimal move. The final board positions are scored using a **positional evaluation heuristic** that assigns high value to strategic squares like corners and edges, a strategy refined through AI vs AI tournaments.

## Core Technologies

- **Python 3**
- **Tkinter**: For the graphical user interface.
- **NumPy**: For efficient, high-performance representation and manipulation of the game board.
- **Numba**: For just-in-time (JIT) compilation of performance-critical functions (move generation, evaluation), enabling the AI to run efficiently at higher depths.

## Installation & Usage

1.  Clone this repository to your local machine.
2.  Ensure you have the required libraries installed: `pip install numpy numba`
3.  Run the main script from your terminal:
    ```bash
    python main_othello.py
    ```
    *(Note: Please change `main_othello.py` to your actual main script filename if different.)*