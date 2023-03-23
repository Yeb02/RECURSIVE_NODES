import os
from ctypes import *

MODE  = r"Release" # "Debug" or "Release"

try:
    dll_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + \
           r"\\x64\\" + MODE + r"\\RECURSIVE_NODES.dll"
    libc = cdll.LoadLibrary(dll_path)
except:
    try:
        dll_path = "RECURSIVE_NODES.dll"
        libc = cdll.LoadLibrary(dll_path)
    except:
        dll_path = os.getcwd() + r"\\x64\\" + MODE + r"\\RECURSIVE_NODES.dll"
        libc = cdll.LoadLibrary(dll_path)


create_population = libc.create_population
create_population.argtypes = [c_int, c_int, c_int]
create_population.restype = c_void_p

mutate_population = libc.mutate_population
mutate_population.argtypes = [c_void_p]
mutate_population.restype = None

get_network_handle = libc.get_network_handle
get_network_handle.argtypes = [c_void_p, c_int]
get_network_handle.restype = c_void_p

get_fittest_network_handle = libc.get_fittest_network_handle
get_fittest_network_handle.argtypes = [c_void_p]
get_fittest_network_handle.restype = c_void_p

set_evolution_parameters = libc.set_evolution_parameters
set_evolution_parameters.argtypes = [c_void_p, c_float, c_float, c_float]
set_evolution_parameters.restype = None

prepare_network = libc.prepare_network
prepare_network.argtypes = [c_void_p]
prepare_network.restype = None

end_trial = libc.end_trial
end_trial.argtypes = [c_void_p]
end_trial.restype = None

get_actions = libc.get_actions
get_actions.argtypes = [c_void_p, c_void_p, c_void_p]
get_actions.restype = None

compute_fitnesses = libc.compute_fitnesses
compute_fitnesses.argtypes = [c_void_p, c_void_p]
compute_fitnesses.restype = None

create_offsprings = libc.create_offsprings
create_offsprings.argtypes = [c_void_p]
create_offsprings.restype = None

destroy_population = libc.destroy_population
destroy_population.argtypes = [c_void_p]
destroy_population.restype = None

initialize_drawer = libc.initialize_drawer
initialize_drawer.argtypes = [c_int, c_int]
initialize_drawer.restype = c_void_p

draw_network = libc.draw_network
draw_network.argtypes = [c_void_p, c_void_p]
draw_network.restype = None