from settings import *
from agent import DDPG_Agent
from simulate import Simulation

if __name__ == '__main__':
    agent = DDPG_Agent(agent_settings)
    simulation = Simulation(MDP_settings)

    loss = simulation.rollout_train(agent)
    plt.title(f'training loss')
    plt.xlabel('episode')
    plt.ylabel('loss')
    plt.plot(loss)
    plt.savefig(f'loss.png')
    plt.clf()

    score = simulation.rollout_test(agent, True)
    plt.title(f'performance')
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.plot(score)
    plt.savefig(f'performance.png')
    plt.clf()
