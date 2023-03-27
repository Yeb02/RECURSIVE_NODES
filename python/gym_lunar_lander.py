import gymnasium as gym
from RECURSIVE_NODES import *
import ctypes

CONTINUOUS = True

env = gym.make("LunarLander-v2", continuous = CONTINUOUS)


print("Visit https://gymnasium.farama.org/environments/box2d/ if the environment does not work.")

N_SPECIMENS = 500
N_TRIALS = 1
in_size = env.observation_space.shape[0]
print(env.observation_space.shape)

# out_size = env.action_space.shape[0] # does not work for lunar lander
# out_size = 2 in [-1, 1] in the continuous case, 4 discrete actions otherwise.
if CONTINUOUS:
    out_size = 2
else :
    out_size = 4  
print(in_size, out_size)

population = create_population(in_size, out_size, N_SPECIMENS)
set_evolution_parameters(population, .0, .2, .0)
drawer = initialize_drawer(1080, 480)

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
        prepare_network(network)
        # print(i)
        for j in range(N_TRIALS):
        
            state = env.reset()
            state = state[0]
            score = 0
            while True:
                # env.render()
                
                for k in range(in_size):
                    observation[k] = state[k]
                get_actions(network, observation, action)

                if CONTINUOUS:
                    state, reward, terminal, truncated, info = env.step(action)
                else: 
                    max = - 2
                    discreteAction = 0
                    for a in range(out_size):
                        if action[a] > max:
                            discreteAction = a
                            max = action[a]
                    state, reward, terminal, truncated, info = env.step(discreteAction)

                score += reward 
                if terminal:
                    break
            end_trial(network)
            if score > maxScore: 
                maxScore = score
            scores[i] += score
        scores[i] *= inv_N_TRIALS

    center_reduce(scores, N_SPECIMENS)
    compute_fitnesses(population, scores)
    create_offsprings(population)
    draw_network(drawer, get_fittest_network_handle(population), step)
    print(f"At iteration {step}, max score was {maxScore}")
    step += 1
    if maxScore >= 500 : 
        print("Max score reached.")
        break

env.close()
destroy_population(population)


