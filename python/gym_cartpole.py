import gym
import os
import ctypes

dll_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + r"\\x64\\Release\\RECURSIVE_NODES.dll"
libc = ctypes.cdll.LoadLibrary(dll_path)

env = gym.make("CartPole-v1")

N_SPECIMENS = 100
N_TRIALS = 3
in_size = env.observation_space.shape[0]
# out_size = env.action_space
out_size = 1

population = libc.create_population(4, 1, 10)

FloatArrayNSpecimens = ctypes.c_float * N_SPECIMENS  # define new type, compatible with c++ arrays.
scores = FloatArrayNSpecimens()                      # instantiate this new type

FloatArrayObsSize = ctypes.c_float * in_size         
observation = FloatArrayObsSize()                      

FloatArrayActionSize = ctypes.c_float * out_size         
action = FloatArrayActionSize()  

inv_N_TRIALS = 1/N_TRIALS

while True:
    maxScore = 0

    libc.mutate_population(population)
    
    for i in range(N_SPECIMENS):
        network = libc.get_network_handle(population, i)

        for j in range(N_TRIALS):
            libc.prepare_network(network)
            state = env.reset()
            score = 0
            while True:
                # env.render()
                for k in range(in_size):
                    observation[k] = state[k]
                action = libc.get_actions(network, observation)
                state, reward, terminal, info = env.step(action)
                score += reward 
                if terminal:
                    break
            if score > maxScore: 
                maxScore = score
            scores[i] += score
        scores[i] *= inv_N_TRIALS

    libc.compute_fitnesses(population, scores)
    libc.create_offsprings(population)

    if maxScore > 500 : 
        break

libc.destroy_population(population)
