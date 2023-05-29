import random
from Actions import Actions

class Evaluation:

    def __init__(self, learning_rate, discount_factor,epsilon) -> None:
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def temporal_difference_learning(self, policy, maze):

        # bootstrapping
        for x in maze.maze_states:
            for y in x:
                if y.terminal is False:
                    y.value = random.randint(1, 10)
                else:  
                    y.value = 0

        for it in range(10000):
            state = self.get_random_state(maze.maze_states)
            while True:

                if state.terminal is True:
                    break

                selected_action = policy.select_action(state)
                state_prime = maze.step(selected_action, state)
                reward = state.reward 

                state.value += self.learning_rate*(reward + self.discount_factor*
                                             state_prime.value - state.value)
                        
                state = state_prime

        for row in maze.maze_states:
            for col in row:
                print("%.2f" % col.value, end=" ")
            print()
        print()

    def SARSA(self, maze):
        
        # Q(S,A) intialization
        Q_map = []

        for x in maze.maze_states:
            row = []
            for y in x:
                row.append(self.make_Q_map_of_1_state(y))
            Q_map.append(row)

        for it in range(1000000):
            state = self.get_random_state(maze.maze_states)
            selected_action = self.select_action_from_Q(state, Q_map)
            while True:
                if state.terminal is True:
                    break

                state_prime = maze.step(selected_action, state)
                reward = state.reward
                action_prime = self.select_action_from_Q(state_prime, Q_map)


                # Q(S,A) = Q(S,A) + alpha*(R + gamma*Q(S',A') - Q(S,A))

                Q_map[state.position[0]][state.position[1]][selected_action.value][2] += \
                    self.learning_rate*(reward + self.discount_factor* \
                    Q_map[state_prime.position[0]][state_prime.position[1]][action_prime.value][2] - \
                    Q_map[state.position[0]][state.position[1]][selected_action.value][2])

                state = state_prime
                selected_action = action_prime


        # printing Q_map for visualization in console
        self.print_Q_map(Q_map)

    def Q_learning(self, maze):
        # Q(S,A) intialization
        Q_map = []

        for x in maze.maze_states:
            row = []
            for y in x:
                row.append(self.make_Q_map_of_1_state(y))
            Q_map.append(row)

        for it in range(10000):
            state = self.get_random_state(maze.maze_states)
            while True:
                if state.terminal is True:
                    break

                selected_action = self.select_action_from_Q(state, Q_map)
                state_prime = maze.step(selected_action, state)
                reward = state.reward

                # Q(S,A) = Q(S,A) + alpha*(R + gamma*max_a(Q(S',a)) - Q(S,A))
                
                Q_map[state.position[0]][state.position[1]][selected_action.value][2] += \
                    self.learning_rate*(reward + self.discount_factor* \
                    self.get_best_action(state_prime, Q_map) - \
                    Q_map[state.position[0]][state.position[1]][selected_action.value][2])
        
                state = state_prime

        self.print_Q_map(Q_map)

    def select_action_from_Q(self, state, Q_Map):
        if random.random() < self.epsilon:
            chosen_action = random.choice(list(Actions))
            return chosen_action
        else:
            return self.get_best_action(state, Q_Map)
    
    def get_best_action(self, state, Q_Map):
        max = -10000
        best_action = None
        for action in Actions:
            if Q_Map[state.position[0]][state.position[1]][action.value][2] > max:
                max = Q_Map[state.position[0]][state.position[1]][action.value][2]
                best_action = action
        return best_action


    def make_Q_map_of_1_state(self,state):
        Q_Map = []
        for action in Actions:
            if state.terminal is True:
                Q_Map.append([state, action, 0])
            else:
                Q_Map.append([state, action, random.randint(1, 10)])
        return Q_Map

    def get_random_state(self, states):

        random_x = random.randint(0, len(states) - 1)
        random_y = random.randint(0, len(states[0]) - 1)

        random_state = states[random_x][random_y]
        return random_state

    def print_Q_map(self, Q_map):
        for row in Q_map:
            for col in row:
                print(col[0][0].position, end=" ")
                for action in col:
                    print(action[1].name, end=" ")
                    print("%.2f" % action[2], end=" ")
                print()
            print()