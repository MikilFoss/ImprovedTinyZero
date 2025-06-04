import argparse
import os
import sys
import torch

sys.path.append(os.getcwd())

from agents import AlphaZeroAgent
from game import PylosGame
from mcts import play
from models import LinearNetwork


def parse_coords(text):
    parts = text.split(',')
    if len(parts) != 3:
        raise ValueError
    return tuple(int(p) for p in parts)


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
                # AI skips removals
                pass

        game.turn *= -1


if __name__ == "__main__":
    main()
