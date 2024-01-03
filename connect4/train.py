from game import Connect4
from datetime import datetime
import torch
import wandb
from tqdm import tqdm
import os
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.getcwd())
from models import Connect4Network  # noqa: E402
from agents import AlphaZeroAgent  # noqa: E402

OUT_DIR = "connect4/out"
INIT_FROM_CHECKPOINT = False
SELFPLAY_GAMES = 400
SELFPLAY_GAMES_PER_SAVE = SELFPLAY_GAMES // 8
BATCH_SIZE = 128
SEARCH_ITERATIONS = 64
MAX_REPLAY_BUFFER_SIZE = BATCH_SIZE * 4
TRAINING_EPOCHS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3  # set to None to disable
WANDB_LOG = True
WANDB_PROJECT_NAME = "tinyalphazero-connect4"
WANDB_RUN_NAME = "run" + datetime.now().strftime("%Y%m%d-%H%M%S")
RUNNING_SWEEP = False



sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'combined_loss',
      'goal': 'minimize'
    },
    'parameters': {
        'LEARNING_RATE': {
            'values': [1e-2, 1e-3, 1e-4, 1e-5]
        },
        'first_layer': {
            'values': [512, 1024, 2048, 4096, 8192]
        },
        'second_layer': {
            'values': [256, 512, 1024, 2048, 4096]
        },
        'sched_factor': {
            'values': [0.1, 0.5, 0.9, 0.99]
        },
        'sched_patience': {
            'values': [10, 50, 100]
        }
    }
}

def train( LEARNING_RATE=None, sched_factor=None, sched_patience=None, BATCH_SIZE=BATCH_SIZE):
    game = Connect4()
    if RUNNING_SWEEP:
        wandb_run = wandb.init()
        first_layer = wandb.config.first_layer
        second_layer = wandb.config.second_layer
        LEARNING_RATE = wandb.config.LEARNING_RATE
        sched_factor = wandb.config.sched_factor
        sched_patience = wandb.config.sched_patience
    print(game.observation_shape)
    model = Connect4Network(game.observation_shape, game.action_space)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=sched_factor, patience=sched_patience, verbose=(not RUNNING_SWEEP))
    agent = AlphaZeroAgent(model, optimizer, MAX_REPLAY_BUFFER_SIZE, scheduler)

    if INIT_FROM_CHECKPOINT:
      agent.load_training_state(f"{OUT_DIR}/model.pth", f"{OUT_DIR}/optimizer.pth", f"{OUT_DIR}/lr_scheduler.pth")

    if WANDB_LOG and not RUNNING_SWEEP:
        wandb_run = wandb.init(project=WANDB_PROJECT_NAME, name=WANDB_RUN_NAME)


    os.makedirs(OUT_DIR, exist_ok=True)
    print("Starting training")
    best_loss = float('inf')
    avg_loss = 10
    for i in tqdm(range(SELFPLAY_GAMES)):
        game.reset()

        values_losses, policies_losses = agent.train_step(
        game, SEARCH_ITERATIONS, BATCH_SIZE, TRAINING_EPOCHS, c_puct=C_PUCT, dirichlet_alpha=DIRICHLET_ALPHA
        )
        
        if len(values_losses) > 0 and len(policies_losses) > 0:
            avg_values_loss = sum(values_losses) / len(values_losses)
            avg_policies_loss = sum(policies_losses) / len(policies_losses)
            avg_loss = avg_values_loss + avg_policies_loss
            if WANDB_LOG:
                wandb.log({"values_loss": avg_values_loss, "policies_loss": avg_policies_loss, "combined_loss": avg_loss})
         
        if (i > 0 and i % SELFPLAY_GAMES_PER_SAVE == 0) and not RUNNING_SWEEP:
            print("Attemping to save model")
            if (avg_loss < best_loss):
                best_loss = avg_loss
                print(f"New best loss: {best_loss}, saving model")
                agent.save_training_state(f"{OUT_DIR}/model.pth", f"{OUT_DIR}/optimizer.pth", f"{OUT_DIR}/lr_scheduler.pth")

    if WANDB_LOG:
      wandb_run.finish()
    if not RUNNING_SWEEP:
        print("Training complete")

        print("Saving final training state")
        agent.save_training_state(f"{OUT_DIR}/model.pth", f"{OUT_DIR}/optimizer.pth", f"{OUT_DIR}/lr_scheduler.pth")

if __name__ == "__main__":
    if not RUNNING_SWEEP:
        train(LEARNING_RATE, 0.1, 50)
    else:
        sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT_NAME)
        wandb.agent(sweep_id, function=train, count=60)