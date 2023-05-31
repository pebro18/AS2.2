import random
from Actions import Actions

class Evaluation:

    def __init__(self, learning_rate, discount_factor,epsilon) -> None:
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def temporal_difference_learning(self, policy, maze):
        """Temporal difference learning algorithm
        Estimates the optimal value function from a given policy and maze
        Based on the pseudocode from the book
        Args:
            policy (Policy): policy object
            maze (Maze): maze object
        Returns:
            maze.maze_states (list): maze_states"""
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
                reward = state_prime.reward

                # V(S) = V(S) + alpha*(R + gamma*V(S') - V(S))
                state.value += self.learning_rate*(reward + self.discount_factor*
                                             state_prime.value - state.value)
                        
                state = state_prime

        for row in maze.maze_states:
            for col in row:
                print("%.2f" % col.value, end=" ")
            print()
        print()
        return maze.maze_states
    

    def SARSA(self, maze):
        """SARSA algorithm
        Estimates the optimal Q from a given maze
        Based on the pseudocode from the book
        Args:
            maze (Maze): maze object
        Returns:
            Q_map (list): Q_map"""
        # Q(S,A) intialization
        Q_map = []

        for x in maze.maze_states:
            row = []
            for y in x:
                row.append(self.make_Q_tuple_of_1_state(y))
            Q_map.append(row)

        for it in range(100000):
            state = self.get_random_state(maze.maze_states)
            selected_action = self.select_action_from_Q(state, Q_map)
            while True:
                if state.terminal is True:
                    break

                state_prime = maze.step(selected_action, state)
                reward = state_prime.reward
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
        return Q_map

    def Q_learning(self, maze):
        """Q_learning algorithm
        Generates the optimal policy given a maze
        Based on the pseudocode from the book
        Args:
            maze (Maze): maze object
        Returns:
            Q_map (list): Q_map"""
        # Q(S,A) intialization
        Q_map = []

        for x in maze.maze_states:
            row = []
            for y in x:
                row.append(self.make_Q_tuple_of_1_state(y))
            Q_map.append(row)

        for it in range(100000):
            state = self.get_random_state(maze.maze_states)
            while True:
                if state.terminal is True:
                    break

                selected_action = self.select_action_from_Q(state, Q_map)
                state_prime = maze.step(selected_action, state)
                reward = state_prime.reward

                # Q(S,A) = Q(S,A) + alpha*(R + gamma*max_a(Q(S',a)) - Q(S,A))
                
                Q_map[state.position[0]][state.position[1]][selected_action.value][2] += \
                    self.learning_rate*(reward + self.discount_factor* \
                    self.max_a(state_prime,Q_map) - \
                    Q_map[state.position[0]][state.position[1]][selected_action.value][2])
        
                state = state_prime

        self.print_Q_map(Q_map)
        return Q_map

    def select_action_from_Q(self, state, Q_Map):
        """Selects action from Q_Map with epsilon greedy policy
        Args:
            state (State): current state
            Q_Map (list): Q_Map
        Returns:
            chosen_action (Action): chosen action"""	
        if random.random() < self.epsilon:
            chosen_action = random.choice(list(Actions))
            return chosen_action
        else:
            return self.get_best_action(state, Q_Map)
    
    def get_best_action(self, state, Q_Map):
        """Returns best action from Q_Map
        Args:
            state (State): current
            Q_Map (list): Q_Map
        Returns:
            best_action (Action): best action"""
        max = -10000
        best_action = None
        for action in Actions:
            if Q_Map[state.position[0]][state.position[1]][action.value][2] > max:
                max = Q_Map[state.position[0]][state.position[1]][action.value][2]
                best_action = action
        return best_action


    def max_a(self, state, Q_Map):
        """Returns max_a(Q(S',a))
        emulates the arg max function from the pseudocode	
        Args:
            state (State): current state
            Q_Map (list): Q_Map
        Returns:
            max (float): max_a(Q(S',a))"""
        max = -10000
        for a in Actions:
            if Q_Map[state.position[0]][state.position[1]][a.value][2] > max:
                max = Q_Map[state.position[0]][state.position[1]][a.value][2]
        return max

    def make_Q_tuple_of_1_state(self,state):
        """Creates Q_Map for one state
        Helper function for Q_learning and SARSA
        is used to initialize Q_Map in their respective functions
        Args:
            state (State): current state
        Returns:
            Q_tuple (list): Q_tuple for one state"""
        Q_Map = []
        for action in Actions:
            if state.terminal is True:
                Q_Map.append([state, action, 0])
            else:
                Q_Map.append([state, action, random.randint(1, 10)])
        return Q_Map

    def get_random_state(self, states):
        """Returns random state from states
        Helper function for all algorithms in this class
        every algorithm needs a random state to start an iteration with
        Args:
            states (list): list of states
        Returns:
            random_state (State): random state from states"""
        random_x = random.randint(0, len(states) - 1)
        random_y = random.randint(0, len(states[0]) - 1)

        random_state = states[random_x][random_y]
        return random_state

    def print_Q_map(self, Q_map):
        """Prints Q_Map in console
        Args:
            Q_map (list): Q_Map"""
        
        for row in Q_map:
            for col in row:
                print(col[0][0].position, end=" ")
                for action in col:
                    print(action[1].name, end=" ")
                    print("%.2f" % action[2], end=" ")
                print()
            print()