import argparse
import sys
from pathlib import Path
import copy
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents import AlphaZeroAgent
from game import PylosGame
from mcts import play
from models import LinearNetwork


def parse_coords(text):
    parts = text.split(',')
    if len(parts) != 3:
        raise ValueError
    return tuple(int(p) for p in parts)


def find_ai_raises(game):
    moves = []
    for sl, layer in enumerate(game.board[:-1]):
        for sr in range(layer.shape[0]):
            for sc in range(layer.shape[1]):
                if layer[sr, sc] == game.turn and not game.piece_has_top(sl, sr, sc):
                    for dl in range(sl + 1, len(game.board)):
                        size = game.board[dl].shape[0]
                        for dr in range(size):
                            for dc in range(size):
                                if game.board[dl][dr, dc] == 0 and game.is_supported(dl, dr, dc):
                                    moves.append(((sl, sr, sc), (dl, dr, dc)))
    return moves


def evaluate_raise(game, move):
    temp = copy.deepcopy(game)
    (sl, sr, sc), (dl, dr, dc) = move
    if not temp.raise_piece(sl, sr, sc, dl, dr, dc):
        return -1
    if temp.top_filled():
        return 2
    if temp.check_for_removal():
        return 1
    return 0


def ai_remove(game):
    removed = 0
    for lvl, layer in enumerate(game.board):
        for r in range(layer.shape[0]):
            for c in range(layer.shape[1]):
                if layer[r, c] == game.turn and not game.piece_has_top(lvl, r, c):
                    if game.remove(lvl, r, c):
                        print(f"AI removes: {lvl},{r},{c}")
                        removed += 1
                        if removed == 2 or not game.check_for_removal():
                            return


def main():
    parser = argparse.ArgumentParser(description="Play Pylos against a trained AI")
    parser.add_argument(
        "--model",
        type=str,
        default="pylos/out/model.pth",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--search",
        type=int,
        default=32,
        help="Number of MCTS simulations for the AI",
    )
    parser.add_argument(
        "--first",
        action="store_true",
        help="Play as white and make the first move",
    )
    args = parser.parse_args()

    game = PylosGame()

    model = LinearNetwork(game.observation_shape, game.action_space)
    model.load_state_dict(torch.load(args.model, map_location=model.device))
    agent = AlphaZeroAgent(model)

    human_turn = 1 if args.first else -1

    while True:
        print(game)
        if game.top_filled():
            winner = "White" if game.board[-1][0, 0] == 1 else "Black"
            print(f"{winner} wins by completing the pyramid!")
            break
        if not game.has_move():
            winner = "White" if game.turn == -1 else "Black"
            print(f"{winner} wins. Opponent has no moves left.")
            break
        player = "White" if game.turn == 1 else "Black"
        if game.turn == human_turn:
            print(f"{player}'s turn. Reserve pieces: {game.reserves[game.turn]}")
            cmd = input("Enter command (place l,r,c | raise sl,sr,sc dl,dr,dc): ").strip()
            if not cmd:
                continue
            if cmd.lower() == "quit":
                break
            tokens = cmd.replace("->", " ").replace("|", " ").split()
            try:
                if tokens[0] == "place" and len(tokens) == 2:
                    lvl, r, c = parse_coords(tokens[1])
                    if not game.place(lvl, r, c):
                        print("Invalid placement")
                        continue
                elif tokens[0] == "raise" and len(tokens) == 3:
                    sl, sr, sc = parse_coords(tokens[1])
                    dl, dr, dc = parse_coords(tokens[2])
                    if not game.raise_piece(sl, sr, sc, dl, dr, dc):
                        print("Invalid raise move")
                        continue
                else:
                    print("Unrecognized command")
                    continue
            except Exception:
                print("Invalid input format")
                continue
        else:
            raise_moves = find_ai_raises(game)
            best_move = None
            best_score = -1
            for mv in raise_moves:
                score = evaluate_raise(game, mv)
                if score > best_score:
                    best_score = score
                    best_move = mv
            if best_move and best_score > 0:
                (sl, sr, sc), (dl, dr, dc) = best_move
                print(f"AI raises: {sl},{sr},{sc} -> {dl},{dr},{dc}")
                game.raise_piece(sl, sr, sc, dl, dr, dc)
            else:
                action = play(game, agent, args.search, c_puct=1.5)
                lvl, r, c = game.index_to_coords[action]
                print(f"AI plays: {lvl},{r},{c}")
                game.place(lvl, r, c)

        if game.check_for_removal():
            if game.turn == human_turn:
                for i in range(2):
                    rem = input(
                        f"You may remove piece {i + 1} (level,row,col) or press Enter to skip: "
                    ).strip()
                    if not rem:
                        break
                    try:
                        rl, rr, rc = parse_coords(rem)
                        if not game.remove(rl, rr, rc):
                            print("Cannot remove that piece")
                            continue
                    except Exception:
                        print("Invalid coordinates")
                        continue
            else:
                ai_remove(game)

        game.turn *= -1


if __name__ == "__main__":
    main()
