from game import Connect4
import torch
from train import OUT_DIR, SEARCH_ITERATIONS
from tqdm import tqdm
from enum import Enum
import os
import sys

sys.path.append(os.getcwd())
from models import LinearNetwork  # noqa: E402
from agents import AlphaZeroAgent, ClassicMCTSAgent  # noqa: E402
from mcts import pit  # noqa: E402
from mcts import play # noqa: E402


class Mode(Enum):
    TEST = "TEST"
    HUMAN = "HUMAN"
    EVAL = "EVAL"

EVAL_GAMES = 100
MODE = Mode.TEST

if __name__ == "__main__":
    game = Connect4()

    model = LinearNetwork(game.observation_shape, game.action_space)
    model.load_state_dict(torch.load(f"{OUT_DIR}/model.pth"))

    agent = AlphaZeroAgent(model)
    agent_play_kwargs = {"search_iterations": SEARCH_ITERATIONS * 2, "c_puct": 1.0, "dirichlet_alpha": None}

    if MODE == Mode.HUMAN:
        print("Playing against human")
        while True:
            print(game)
            legal_actions = game.get_legal_actions()
            print(f"Legal actions: {legal_actions}")
            action = int(input("Enter action: "))
            game.step(action)
            if game.get_result() is not None:
                print(game)
                print("Human wins")
                break
            print(game)
            action = play(game, agent, **agent_play_kwargs)
            print(f"Agent action: {action}")
            game.step(action)
            if game.get_result() is not None:
                print(game)
                print("Agent wins")
                break
    elif MODE == Mode.EVAL:

        print(f"Playing {EVAL_GAMES} games against itself")

        results = {0: 0, 1: 0, -1: 0}
        for _ in tqdm(range(EVAL_GAMES)):
            game.reset()
            result = pit(
            game,
            agent,
            agent,
            agent_play_kwargs,
            agent_play_kwargs,
            )
            results[result] += 1

        print("Results:")
        print(f"First player wins: {results[1]}")
        print(f"Second player wins: {results[-1]}")
        print(f"Draws: {results[0]}")

        classic_mcts_agent = ClassicMCTSAgent
        classic_mcts_agent_play_kwargs = {"search_iterations": 250, "c_puct": 1.0, "dirichlet_alpha": None}

        print(f"Playing {EVAL_GAMES} games against classic MCTS agent (starting first)")

        results = {0: 0, 1: 0, -1: 0}
        for _ in tqdm(range(EVAL_GAMES)):
            game.reset()
            result = pit(
            game,
            agent,
            classic_mcts_agent,
            agent_play_kwargs,
            classic_mcts_agent_play_kwargs,
            )
            results[result] += 1

        print("Results:")
        print(f"AlphaZero agent wins: {results[1]}")
        print(f"Classic MCTS agent wins: {results[-1]}")
        print(f"Draws: {results[0]}")

        print(f"Playing {EVAL_GAMES} games against classic MCTS agent (starting second)")

        results = {0: 0, 1: 0, -1: 0}
        for _ in tqdm(range(EVAL_GAMES)):
            game.reset()
            result = pit(
            game,
            classic_mcts_agent,
            agent,
            classic_mcts_agent_play_kwargs,
            agent_play_kwargs,
            )
            results[result] += 1

        print("Results:")
        print(f"Classic MCTS agent wins: {results[1]}")
        print(f"AlphaZero agent wins: {results[-1]}")
        print(f"Draws: {results[0]}")
    elif MODE == Mode.TEST:
        print(f"Playing {EVAL_GAMES} games between MCTS agents")
        classic_mcts_agent = ClassicMCTSAgent
        classic_mcts_agent_play_kwargs = {"search_iterations": 2000, "c_puct": 1.0, "dirichlet_alpha": None}
        results = {0: 0, 1: 0, -1: 0}
        with tqdm(total=EVAL_GAMES) as pbar:
            for _ in range(EVAL_GAMES):
                game.reset()
                result = pit(
                    game,
                    classic_mcts_agent,
                    classic_mcts_agent,
                    classic_mcts_agent_play_kwargs,
                    classic_mcts_agent_play_kwargs,
                )
                results[result] += 1

                # Update the progress bar description with the current win counts
                pbar.set_description(f"First player wins: {results[1]}, Second player wins: {results[-1]}, Draws: {results[0]}")
                pbar.update()

        print("Results:")
        print(f"First player wins: {results[1]}")
        print(f"Second player wins: {results[-1]}")
        print(f"Draws: {results[0]}")        

