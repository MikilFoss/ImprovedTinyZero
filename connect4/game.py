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
            new_arr.append([str(x).ljust(10) for x in self.state[i*7:(i+1)*7]])
        new_arr.reverse()
        return '\n'.join([''.join(x) for x in new_arr])
    
    def to_observation(self):
        obs = np.zeros(self.STATE_LEN, dtype=np.float32)
        for i, x in enumerate(self.state):
            if x == self.turn:
                obs[i] = 1
            elif x == -self.turn:
                obs[i] = -1
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


