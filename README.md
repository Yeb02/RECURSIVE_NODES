# RECURSIVE_NODES

A very modified NEAT algorithm.

Implements a 3 loops optimization algorithm to tackle reinforcement learning challenges and general learning tasks. A population of recursive networks is meta-optimized by a custom genetic algorithm, which evolves both topology and floating point parameters. Networks use evolved local learning rules, cascading neuromodulation and dedicated memory units with various available implementations.

&nbsp;

<p align="center">
  <img src="Capture.PNG">
</p>
<em>Visualisation of the genotype of a simple specimen.</em>

&nbsp;
##### KNOWN BUGS: Stable. 

##### IN PROGRESS: GPU-isation (CUDA), and progressive environments (LibTorch).

##### TODO: Heap defragmentation. GPU port only when sufficient gains.
&nbsp;

## USER GUIDE
&nbsp;
#### Several variations of the algorithm are implemented, change the active preprocessor directives in config.h to compile a custom version. More info in the file. The project can be compiled into an executable or a dll.
&nbsp;

With the executable, one can tinker with the whole algorithm directly. It can be tested on rocketSim and then in Rocket league with RLBot, or on the some simple c++ trials. As of now, 5 trials are implemented: XoR with memorization, gym's cartpole, T-Maze, N-Links pendulum, and Key-value memorization. More to come !

The dll has limited features, especially when it comes to the meta-optimization algorithm. It is meant to be used in python with ctypes, for benchmarking on openAI's gym's (now gymnasium) for instance. A demo can be found in python\gym_cartpole.py. 
It also allows models optimized with the exe to be tested in python.

#### Example run:

- Compile a Release dll (Project Properties -> Configurations Properties -> General -> Configuration Type).
- Open a command prompt and navigate to the RECURSIVE_NODES folder. (Not the "python" subfolder)
 - Run `python python\gym_cartpole.py`

&nbsp;

## Visual studio 2022 setup:
&nbsp;

ISO C++20 is required. To switch between dll and exe, go to Project Properties -> Configurations Properties -> General -> Configuration Type. 

The project requires SFML 2.5, if you wish to display the evolved topology. (It can be toggled on and off with the DRAWING  preprocessor directive in config.h before building.)

If SFML is used, the following DLLs must be placed in the same folder as the built file, whether compiling a .exe or a .dll :

&nbsp;

### In debug mode, in RECURSIVE_NODES\x64\Debug:

  sfml-graphics-d-2.dll     sfml-system-d-2.dll     sfml-window-d-2.dll
  
  
### In release mode, in RECURSIVE_NODES\x64\Release:

  sfml-graphics-2.dll      sfml-system-2.dll     sfml-window-2.dll
  
&nbsp;

##### To obtain those, download SFML 2.5.1.

&nbsp;

#### Details

- I recommend toggling adress Sanitizer on (if you have it installed) when debuging the .exe . It MUST be disabled when debuging the DLL because of a VS bug. Found in  Project Properties -> C/C++ -> General -> Enable Adress Sanitizer. It is also incompatible with rocketSim. 

- If you do not have a rocketSim install you will need to exclude rocketSim.h and rocketSim.cpp from the solution, same for files using torch if you dont have libtorch. 
