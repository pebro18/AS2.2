class State:
    def __init__(self,rewards = -1,is_terminal = False) -> None:
        self.position = None
        self.reward = rewards
        self.terminal = is_terminal
        self.value = 0