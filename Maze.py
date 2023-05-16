import Actions
from State import State


class Maze:
    def __init__(self, length=4, width=4):
        self.maze_rewards = [[-1 for i in range(length)] for j in range(width)]
        self.maze_states = None
        self.actions = Actions.Actions

    def set_reward(self, state, reward):
        self.maze_rewards[state[0]][state[1]] = reward

    def assign_maze_states(self):
        self.maze_states = [[State() for i in range(len(self.maze_rewards))]
                            for j in range(len(self.maze_rewards[0]))]
        for x_axis in range(len(self.maze_rewards)):
            for y_axis in range(len(self.maze_rewards[x_axis])):
                self.maze_states[x_axis][y_axis].position = (x_axis, y_axis)
                self.maze_states[x_axis][y_axis].reward = self.maze_rewards[x_axis][y_axis]  # noqa: E501

    def set_terminal(self, state, is_terminal=True):
        self.maze_states[state[0]][state[1]].terminal = is_terminal

    def step(self, action, state):

        if action == self.actions.LEFT:
            if 0 <= state.position[1] - 1 <= len(self.maze_states) - 1:
                return self.maze_states[state.position[0]][state.position[1] - 1]
        if action == self.actions.RIGHT:
            if 0 <= state.position[1] + 1 <= len(self.maze_states) - 1:
                return self.maze_states[state.position[0]][state.position[1] + 1]
        if action == self.actions.UP:
            if 0 <= state.position[0] - 1 <= len(self.maze_states[0]) - 1:
                return self.maze_states[state.position[0] - 1][state.position[1]]
        if action == self.actions.DOWN:
            if 0 <= state.position[0] + 1 <= len(self.maze_states[0]) - 1:
                return self.maze_states[state.position[0] + 1][state.position[1]]

        return self.maze_states[state.position[0]][state.position[1]]

    def print_maze_states(self) -> None:
        print("Maze states:")
        for x_axis in range(len(self.maze_states)):
            for y_axis in range(len(self.maze_states[x_axis])):
                print(f"{self.maze_states[x_axis][y_axis].reward}", end=" ")
            print()
        print()
