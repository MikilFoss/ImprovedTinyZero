import numpy as np


class PylosGame:
    """Simple playable version of the board game Pylos."""

    LEVEL_SIZES = [4, 3, 2, 1]

    def __init__(self):
        # mapping from action index to (level, row, col)
        self.index_to_coords = []
        for lvl, size in enumerate(self.LEVEL_SIZES):
            for r in range(size):
                for c in range(size):
                    self.index_to_coords.append((lvl, r, c))
        self.action_space = len(self.index_to_coords)
        self.observation_shape = (self.action_space,)

        self.reset()

    def reset(self):
        self.board = [np.zeros((s, s), dtype=int) for s in self.LEVEL_SIZES]
        self.turn = 1  # 1 -> White, -1 -> Black
        self.reserves = {1: 15, -1: 15}
        self.last_move = None
        self.actions_stack = []

    # -----------------------------------------------------------
    def __str__(self):
        lines = []
        for lvl, layer in enumerate(self.board):
            lines.append(f"Level {lvl}:")
            for r in range(layer.shape[0]):
                row = []
                for c in range(layer.shape[1]):
                    v = layer[r, c]
                    if v == 1:
                        row.append("W")
                    elif v == -1:
                        row.append("B")
                    else:
                        row.append(".")
                lines.append(" ".join(row))
            lines.append("")
        return "\n".join(lines)

    # -----------------------------------------------------------
    def piece_has_top(self, level, r, c):
        if level >= len(self.board) - 1:
            return False
        for dr in (0, -1):
            for dc in (0, -1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < self.board[level + 1].shape[0] and 0 <= nc < self.board[level + 1].shape[1]:
                    if self.board[level + 1][nr, nc] != 0:
                        return True
        return False

    def is_supported(self, level, r, c):
        if level == 0:
            return True
        b = self.board[level - 1]
        return (
            b[r, c] != 0
            and b[r + 1, c] != 0
            and b[r, c + 1] != 0
            and b[r + 1, c + 1] != 0
        )

    # -----------------------------------------------------------
    def place(self, level, r, c):
        if self.reserves[self.turn] <= 0:
            return False
        if self.board[level][r, c] != 0:
            return False
        if not self.is_supported(level, r, c):
            return False
        self.board[level][r, c] = self.turn
        self.reserves[self.turn] -= 1
        self.last_move = (level, r, c)
        return True

    def raise_piece(self, sl, sr, sc, dl, dr, dc):
        if self.board[sl][sr, sc] != self.turn:
            return False
        if self.piece_has_top(sl, sr, sc):
            return False
        if self.board[dl][dr, dc] != 0:
            return False
        if dl <= sl:
            return False
        if not self.is_supported(dl, dr, dc):
            return False
        # move
        self.board[sl][sr, sc] = 0
        self.board[dl][dr, dc] = self.turn
        self.last_move = (dl, dr, dc)
        return True

    # -----------------------------------------------------------
    def check_square(self, level, r, c):
        layer = self.board[level]
        player = layer[r, c]
        if player == 0:
            return False
        for dr in (0, -1):
            for dc in (0, -1):
                rr = r + dr
                cc = c + dc
                if rr < 0 or cc < 0 or rr + 1 >= layer.shape[0] or cc + 1 >= layer.shape[1]:
                    continue
                sq = layer[rr : rr + 2, cc : cc + 2]
                if np.all(sq == player):
                    return True
        return False

    def check_line(self, level, r, c):
        player = self.board[level][r, c]
        if player == 0:
            return False
        if level == 0:
            if np.all(self.board[level][r, :] == player):
                return True
            if np.all(self.board[level][:, c] == player):
                return True
        if level == 1:
            if np.all(self.board[level][r, :] == player):
                return True
            if np.all(self.board[level][:, c] == player):
                return True
        return False

    def check_for_removal(self):
        if not self.last_move:
            return False
        lvl, r, c = self.last_move
        if self.check_square(lvl, r, c):
            return True
        if self.check_line(lvl, r, c):
            return True
        return False

    def remove(self, level, r, c):
        if self.board[level][r, c] != self.turn:
            return False
        if self.piece_has_top(level, r, c):
            return False
        self.board[level][r, c] = 0
        self.reserves[self.turn] += 1
        return True

    def top_filled(self):
        return self.board[-1][0, 0] != 0

    # -----------------------------------------------------------
    def has_move(self):
        if self.reserves[self.turn] > 0:
            for lvl, layer in enumerate(self.board):
                for r in range(layer.shape[0]):
                    for c in range(layer.shape[1]):
                        if layer[r, c] == 0 and self.is_supported(lvl, r, c):
                            return True
        # check for possible raises
        for sl, layer in enumerate(self.board[:-1]):
            for r in range(layer.shape[0]):
                for c in range(layer.shape[1]):
                    if layer[r, c] == self.turn and not self.piece_has_top(sl, r, c):
                        for dl in range(sl + 1, len(self.board)):
                            size = self.board[dl].shape[0]
                            for dr in range(size):
                                for dc in range(size):
                                    if self.board[dl][dr, dc] == 0 and self.is_supported(dl, dr, dc):
                                        if dl - 1 >= sl:
                                            return True
        return False

    def get_legal_raises(self):
        """Return a list of valid raise moves for the current player.

        Each element is a tuple (sl, sr, sc, dl, dr, dc).
        """
        raises = []
        for sl, layer in enumerate(self.board[:-1]):
            for sr in range(layer.shape[0]):
                for sc in range(layer.shape[1]):
                    if layer[sr, sc] == self.turn and not self.piece_has_top(sl, sr, sc):
                        for dl in range(sl + 1, len(self.board)):
                            size = self.board[dl].shape[0]
                            for dr in range(size):
                                for dc in range(size):
                                    if self.board[dl][dr, dc] == 0 and self.is_supported(dl, dr, dc):
                                        if dl - 1 >= sl:
                                            raises.append((sl, sr, sc, dl, dr, dc))
        return raises

    # -----------------------------------------------------------
    # AlphaZero training helpers
    def get_legal_actions(self):
        if self.reserves[self.turn] <= 0:
            return []
        actions = []
        for idx, (lvl, r, c) in enumerate(self.index_to_coords):
            if self.board[lvl][r, c] == 0 and self.is_supported(lvl, r, c):
                actions.append(idx)
        return actions

    def step(self, action):
        lvl, r, c = self.index_to_coords[action]
        if not self.place(lvl, r, c):
            raise ValueError(f"Illegal action {action}")
        self.actions_stack.append(action)
        self.turn *= -1

    def undo_last_action(self):
        self.turn *= -1
        action = self.actions_stack.pop()
        lvl, r, c = self.index_to_coords[action]
        self.board[lvl][r, c] = 0
        self.reserves[self.turn] += 1

    def get_result(self):
        if self.top_filled():
            return self.board[-1][0, 0]
        if len(self.get_legal_actions()) == 0:
            return -self.turn

    def get_first_person_result(self):
        result = self.get_result()
        if result is not None:
            return result * self.turn

    @staticmethod
    def swap_result(result):
        return -result

    def to_observation(self):
        obs = []
        for lvl, size in enumerate(self.LEVEL_SIZES):
            layer = self.board[lvl]
            for r in range(size):
                for c in range(size):
                    cell = layer[r, c]
                    if cell == self.turn:
                        obs.append(1.0)
                    elif cell == -self.turn:
                        obs.append(-1.0)
                    else:
                        obs.append(0.0)
        return np.array(obs, dtype=np.float32)

