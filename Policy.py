from Actions import Actions


class Policy:
    def __init__(self, lenght=4, width=4):
        self.policy = [[0 for i in range(lenght)] for j in range(width)]

    def select_action(self, state):
        all_possible_neighbors = self.get_neighbors(state)
        best_action = max(all_possible_neighbors, key=lambda x: x[0])
        return best_action[1]

    def get_neighbors(self, state):
        neighbors = []
        if 0 <= state.position[1] - 1 <= len(self.policy) - 1:
            neighbors.append(
                (self.policy[state.position[0]][state.position[1] - 1], Actions.LEFT))
        if 0 <= state.position[1] + 1 <= len(self.policy) - 1:
            neighbors.append(
                (self.policy[state.position[0]][state.position[1] + 1], Actions.RIGHT))
        if 0 <= state.position[0] - 1 <= len(self.policy[0]) - 1:
            neighbors.append(
                (self.policy[state.position[0] - 1][state.position[1]], Actions.UP))
        if 0 <= state.position[0] + 1 <= len(self.policy[0]) - 1:
            neighbors.append(
                (self.policy[state.position[0] + 1][state.position[1]], Actions.DOWN))
        return neighbors

    def print_policy(self):
        print("Policy:")
        for row in self.policy:
            print(row)
