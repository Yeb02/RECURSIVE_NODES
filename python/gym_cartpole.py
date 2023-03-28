import gym
from RECURSIVE_NODES import *
import ctypes

print("If there is a syntax error, update your version of gym, the lib has changed a lot.\n")
env = gym.make("CartPole-v1")

N_SPECIMENS = 500
N_TRIALS = 5
in_size = env.observation_space.shape[0]
# out_size = env.action_space.shape[0] # does not work for cartpole
out_size = 1

population = create_population(in_size, out_size, N_SPECIMENS)
set_evolution_parameters(population, .5, .05, 0.0)
drawer = initialize_drawer(1080, 480)

#  Use type compatible with c++ arrays
scores = (ctypes.c_float * N_SPECIMENS)()                      
observation = (ctypes.c_float * in_size)()                          
action = (ctypes.c_float * out_size)()      

maxTrialSteps = 50000
evolutionStep = 0
while True:
    maxScore = 0

    mutate_population(population)
    
    for i in range(N_SPECIMENS):
        network = get_network_handle(population, i)
        prepare_network(network)
        for j in range(N_TRIALS):
        
            state = env.reset()[0]
            score = 0
            trialStep = 0
            while True:
                # env.render()
                for k in range(in_size):
                    observation[k] = state[k]
                get_actions(network, observation, action)
                a = 1 if action[0] > 0 else 0
                state, reward, terminal, truncated, info = env.step(a)
                score += reward 
                trialStep+=1
                if terminal or trialStep>maxTrialSteps:
                    break
            end_trial(network)
            if score > maxScore: 
                maxScore = score
            scores[i] += score
        

    compute_fitnesses(population, scores)
    create_offsprings(population)
    draw_network(drawer, get_fittest_network_handle(population), evolutionStep)
    print(f"At iteration {evolutionStep}, max score was {maxScore}")
    evolutionStep += 1
    if maxScore >= 500 : 
        print("Max score reached.")
        break

env.close()
destroy_population(population)


