from Actions import Actions
import copy


class Agent:
    def __init__(self, maze, policy, start_state, discount_factor=0.9):
        self.maze = maze
        self.state = start_state
        self.total_reward = 0

        self.policy = policy
        self.threshold = 0.1
        self.discount_factor = discount_factor

    def act(self):
        action = self.policy.select_action(self.state)
        self.state = self.maze.step(action, self.state)
        self.total_reward += self.state.reward
        pass

    def value_function(self):

        count = 0
        while True:
            count += 1
            # print(f"Value Iteration: {count}")
            # self.print_maze_values()
            delta = 0

            copy_of_maze = copy.deepcopy(self.maze.maze_states)

            for x, row in enumerate(self.maze.maze_states):
                for y, column in enumerate(row):

                    possible_states = self.get_all_possible_neighbors(
                        x, y, column)

                    for s in possible_states:
                        v = column.value
                        if self.maze.maze_states[x][y].terminal:
                            copy_of_maze[x][y].value = self.maze.maze_states[x][y].reward
                        else:
                            action = self.calculate_value(
                                possible_states, self.maze.maze_states[x][y].reward)
                            copy_of_maze[x][y].value = action[1]
                        delta = max(delta, abs(v - copy_of_maze[x][y].value))

            self.maze.maze_states = copy_of_maze
            if delta < self.threshold:
                break

        for x, row in enumerate(self.maze.maze_states):
            for y, column in enumerate(row):
                self.policy.policy[x][y] = self.maze.maze_states[x][y].value

    def get_all_possible_neighbors(self, x, y, column):
        possible_states = []
        for action in Actions:
            possible_states.append(
                (self.maze.step(action, column), self.maze.maze_states[x][y].value, action))

        return possible_states

    def calculate_value(self, states, current_state_reward):
        for index, state in enumerate(states):
            calculation = (self.discount_factor *
                           state[0].value) + (current_state_reward)
            states[index] = tuple((state[0], calculation, state[2]))

        total_sum = sum(states[x][1] for x in range(len(states)))
        chosen_action = max(states, key=lambda x: x[1])
        return chosen_action + tuple((total_sum,))

    def print_position(self):
        print(f"Agent at: {self.state.position}")

    def print_maze_values(self):
        for row in self.maze.maze_states:
            for state in row:
                print(state.value, end=" ")
            print()
        print()
