import gym
from RECURSIVE_NODES import *
import ctypes

env = gym.make("CartPole-v1")

N_SPECIMENS = 500
N_TRIALS = 5
in_size = env.observation_space.shape[0]
# out_size = env.action_space.shape[0] // does not work for cartpole
out_size = 1

population = create_population(in_size, out_size, N_SPECIMENS)
set_evolution_parameters(population, .2, .05)
drawer = initialize_drawer(720, 480)

#  Use type compatible with c++ arrays
scores = (ctypes.c_float * N_SPECIMENS)()                      
observation = (ctypes.c_float * in_size)()                          
action = (ctypes.c_float * out_size)()      

inv_N_TRIALS = 1/N_TRIALS

step = 0
while True:
    maxScore = 0

    mutate_population(population)
    
    for i in range(N_SPECIMENS):
        network = get_network_handle(population, i)

        for j in range(N_TRIALS):
            prepare_network(network)
            state = env.reset()
            score = 0
            while True:
                # env.render()
                for k in range(in_size):
                    observation[k] = state[k]
                get_actions(network, observation, action)
                a = 1 if action[0] > 0 else 0
                state, reward, terminal, info = env.step(a)
                score += reward 
                if terminal:
                    break
            if score > maxScore: 
                maxScore = score
            scores[i] += score
        scores[i] *= inv_N_TRIALS

    compute_fitnesses(population, scores)
    create_offsprings(population)
    draw_network(drawer, get_fittest_network_handle(population))
    print(f"At iteration {step}, max score was {maxScore}")
    step += 1
    if maxScore >= 500 : 
        break


destroy_population(population)


