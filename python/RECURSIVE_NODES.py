import os
from ctypes import *

MODE  = r"Release" # "Debug" or "Release"
print(os.getcwd())
try:
    dll_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + \
           r"\\x64\\" + MODE + r"\\RECURSIVE_NODES.dll"
    libc = cdll.LoadLibrary(dll_path)
except:
    try:
        dll_path = "RECURSIVE_NODES.dll"
        libc = cdll.LoadLibrary(dll_path)
    except:
        try:
            dll_path = os.getcwd() + r"\\x64\\" + MODE + r"\\RECURSIVE_NODES.dll"
            libc = cdll.LoadLibrary(dll_path)
        except:
            try:
                dll_path = os.getcwd() + r"\\src\\RECURSIVE_NODES.dll"
                libc = cdll.LoadLibrary(dll_path)
            except:
                dll_path = r"src\\RECURSIVE_NODES.dll"
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

load_existing_network = libc.load_existing_network
load_existing_network.argtypes = [c_char_p]
load_existing_network.restype = c_void_p

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

is_drawing_enabled = libc.is_drawing_enabled
is_drawing_enabled.argtypes = []
is_drawing_enabled.restype = c_int

get_observations_size = libc.get_observations_size
get_observations_size.argtypes = [c_void_p]
get_observations_size.restype = int

get_actions_size = libc.get_actions_size
get_actions_size.argtypes = [c_void_p]
get_actions_size.restype = int

if is_drawing_enabled() != 0:
    initialize_drawer = libc.initialize_drawer
    initialize_drawer.argtypes = [c_int, c_int]
    initialize_drawer.restype = c_void_p

    draw_network = libc.draw_network
    draw_network.argtypes = [c_void_p, c_void_p, c_int]
    draw_network.restype = None


#utils:

# takes a c-float array by reference and its size, and linearly
#transforms it so that it has mean 0 and variance 1. 
def normalize(a, size):
    E = 0.0
    for i in range(size):
        E += a[i]

    E /= size
    var = 0.0
    for i in range(size):
        a[i] -= E
        var += a[i] * a[i]

    invStddev = 1.0/(var**.5)
    for i in range(size):
        a[i] *= invStddev

    return