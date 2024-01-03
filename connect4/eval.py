from game import Connect4
import torch
from train import OUT_DIR, SEARCH_ITERATIONS
from tqdm import tqdm
from enum import Enum
import os
import sys
import pygame

sys.path.append(os.getcwd())
from models import LinearNetwork  # noqa: E402
from agents import AlphaZeroAgent, ClassicMCTSAgent  # noqa: E402
from mcts import pit  # noqa: E402
from mcts import play # noqa: E402


class Mode(Enum):
    TEST = "TEST"
    HUMAN = "HUMAN"
    EVAL = "EVAL"

EVAL_GAMES = 10
MODE = Mode.HUMAN

pygame.init()
SCREEN_HEIGHT = 1000
SCREEN_WIDTH = 1200
screen = pygame.display.set_mode((700, 600))
CELL_SIZE = 100

def draw_board(board):
    for row in range(-1,6):
        for col in range(7):
            pygame.draw.rect(screen, (0, 0, 100), (col*CELL_SIZE, row*CELL_SIZE+CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.circle(screen, (0,0,0), (int(col*CELL_SIZE+CELL_SIZE/2), int(row*CELL_SIZE+CELL_SIZE+CELL_SIZE/2)), CELL_SIZE/2-5)
    for row in range(6):
        for col in range(7):
            if board[col+row*7] == 1:
                pygame.draw.circle(screen, (255,0,0), (int(col*CELL_SIZE+CELL_SIZE/2), SCREEN_HEIGHT-int(row*CELL_SIZE+CELL_SIZE/2)), CELL_SIZE/2-5)
            elif board[col+row*7] == -1: 
                pygame.draw.circle(screen, (255,255,0), (int(col*CELL_SIZE+CELL_SIZE/2), SCREEN_HEIGHT-int(row*CELL_SIZE+CELL_SIZE/2)), CELL_SIZE/2-5)
    pygame.display.update()

if __name__ == "__main__":
    game = Connect4()

    model = LinearNetwork(game.observation_shape, game.action_space, 8000, 8000)
    model.load_state_dict(torch.load(f"{OUT_DIR}/model.pth"))

    agent = AlphaZeroAgent(model)
    agent_play_kwargs = {"search_iterations": SEARCH_ITERATIONS * 2, "c_puct": 1.0, "dirichlet_alpha": None}

    if MODE == Mode.HUMAN:
        print("Playing against human")
        agent_play_kwargs = {"search_iterations": 1000, "c_puct": 1.0, "dirichlet_alpha": None}
        draw_board(game.state)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x = event.pos[0] // CELL_SIZE
                    game.step(x)
                    draw_board(game.state)

                    if game.get_result() is not None:
                        print("Human wins")
                        pygame.time.wait(3000)
                        sys.exit()

                    action = play(game, agent, **agent_play_kwargs)
                    game.step(action)
                    draw_board(game.state)

                    if game.get_result() is not None:
                        print("Agent wins")
                        pygame.time.wait(3000)
                        sys.exit()
    elif MODE == Mode.EVAL:

        classic_mcts_agent = ClassicMCTSAgent
        classic_mcts_agent_play_kwargs = {"search_iterations": 1000, "c_puct": 1.0, "dirichlet_alpha": None}

        print(f"Playing {EVAL_GAMES} games against classic MCTS agent (starting first)")

        results = {0: 0, 1: 0, -1: 0}
        with tqdm(total=EVAL_GAMES) as pbar:
            for _ in range(EVAL_GAMES):
                game.reset()
                result = pit(
                game,
                agent,
                classic_mcts_agent,
                agent_play_kwargs,
                classic_mcts_agent_play_kwargs,
                )
                results[result] += 1
                pbar.set_description(f"First player wins: {results[1]}, Second player wins: {results[-1]}, Draws: {results[0]}")
                pbar.update()

            

        print("Results:")
        print(f"AlphaZero agent wins: {results[1]}")
        print(f"Classic MCTS agent wins: {results[-1]}")
        print(f"Draws: {results[0]}")

        print(f"Playing {EVAL_GAMES} games against classic MCTS agent (starting second)")

        results = {0: 0, 1: 0, -1: 0}
        with tqdm(total=EVAL_GAMES) as pbar:
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
                pbar.set_description(f"First player wins: {results[1]}, Second player wins: {results[-1]}, Draws: {results[0]}")
                pbar.update()

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

