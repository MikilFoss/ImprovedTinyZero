import numpy as np


class Connect4:
    STATE_LEN = 42
    def __init__(self):
        self.reset()
        self.observation_shape = self.to_observation().shape
        self.action_space = 7
  
    def reset(self):
        self.state = [0] * self.STATE_LEN
        self.actions_stack = []
        self.turn = 1
    def __str__(self):
        new_arr = []
        for i in range(6):
            new_arr.append(['|' + str(x).center(3) + '|' for x in self.state[i*7:(i+1)*7]])
        new_arr.reverse()
        new_arr.insert(0,[' ' * 5 for _ in range(7)])
        new_arr.insert(0,['|' + str(i).center(3) + '|' for i in range(7)])
        return '\n'.join([''.join(x) for x in new_arr])
    
    def to_observation(self):
        # The board is 6 rows by 7 columns
        rows, columns = 6, 7

        # Create an empty 3D array with 3 channels, 6 rows, and 7 columns
        obs = np.zeros((3, rows, columns), dtype=np.float32)

        for row in range(rows):
            for col in range(columns):
                index = row * 7 + col
                if self.state[index] == self.turn:
                    # Current player's pieces
                    obs[0, row, col] = 1
                elif self.state[index] == -self.turn:
                    # Opponent's pieces
                    obs[1, row, col] = 1
                else:
                    # Empty cells
                    obs[2, row, col] = 1

        return obs
    def get_legal_actions(self):
        legal_actions = []
        for col in range(7):
            if self.state[col + 35] == 0:  
                legal_actions.append(col)
        return legal_actions
    
    def step(self, action):
        if self.state[action + 35] != 0:
            raise ValueError(f"Action {action} is illegal")
        for row in range(6):
            if self.state[action + row*7] == 0:
                self.state[action + row*7] = self.turn
                self.actions_stack.append(action)
                self.turn *= -1
                break
    def undo_last_action(self):
        action = self.actions_stack.pop()
        for row in range(5, -1, -1):
            if self.state[action + row*7] != 0:
                self.state[action + row*7] = 0
                self.turn *= -1
                break
    def get_result(self):
        # Check horizontal locations for win
        for r in range(6):
            for c in range(4):
                if self.state[r*7 + c] == self.state[r*7 + c+1] == self.state[r*7 + c+2] == self.state[r*7 + c+3] != 0:
                    return self.state[r*7 + c]

        # Check vertical locations for win
        for c in range(7):
            for r in range(3):
                if self.state[r*7 + c] == self.state[(r+1)*7 + c] == self.state[(r+2)*7 + c] == self.state[(r+3)*7 + c] != 0:
                    return self.state[r*7 + c]

        # Check positively sloped diagonals
        for r in range(3):
            for c in range(4):
                if self.state[r*7 + c] == self.state[(r+1)*7 + c+1] == self.state[(r+2)*7 + c+2] == self.state[(r+3)*7 + c+3] != 0:
                    return self.state[r*7 + c]

        # Check negatively sloped diagonals
        for r in range(3, 6):
            for c in range(4):
                if self.state[r*7 + c] == self.state[(r-1)*7 + c+1] == self.state[(r-2)*7 + c+2] == self.state[(r-3)*7 + c+3] != 0:
                    return self.state[r*7 + c]

        if len(self.get_legal_actions()) == 0:
            return 0
    def get_first_person_result(self):
        result = self.get_result()
        if result is not None:
            return result * self.turn
    @staticmethod
    def swap_result(result):
        return -result