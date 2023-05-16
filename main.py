import Agent
import Maze
import Policy


def main():
    maze = Maze.Maze()
    maze.set_reward((0, 3), 40)
    maze.set_reward((1, 2), -10)
    maze.set_reward((1, 3), -10)
    maze.set_reward((3, 0), 10)
    maze.set_reward((3, 1), -2)

    maze.assign_maze_states()
    maze.set_terminal((0, 3), True)
    maze.set_terminal((3, 0), True)

    policy = Policy.Policy(lenght=4, width=4)
    agent = Agent.Agent(maze, policy, maze.maze_states[3][2],1)

    print("Initial iteration:")
    maze.print_maze_states()
    agent.print_position()
    print()

    agent.value_function()
    agent.policy.print_policy()
    
    count = 0
    while True:
        agent.act()
        count += 1
        print(f"Agent Iteration: {count}")
        maze.print_maze_states()
        agent.print_position()
        print(f"total reward: {agent.total_reward}")
        print()
        if agent.state.terminal:
            break


if __name__ == "__main__":
    main()
