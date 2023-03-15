# RECURSIVE_NODES

Implements a 3 loops optimization algorithm to solve optimal control and memorization problems. The neural network is meta-optimized by a custom genetic algorithm, which evolves both topology and floating point parameters. The network itself uses 3 factor Hebbian learning rules. 
Several variants are implemented, change the active preprocessor directives in main.cpp or dllMain.h to compile a custom version.

##### KNOWN BUG: None, report any !

The code can be compiled into an executable, to experiment on the implemented c++ trials. As of now, 2 trials are implemented: XoR with memorization, and gym's cartpole. The algorithm solves both easily. More to come !

It can also be compiled into a dll, to use in python with ctypes. One can then evaluate the algorithm on any python trial, like on openAI's gym's for instance. A demo can be found in python\gym_cartpole.py (slower convergence than this repo's c++ implementation because gym offers no way to reset an env to its initial state, which introduces significant noise in the fitness function). It is still in developpement, so stick close to the demo's architecture if you dont want to run into undefined behaviours.

## Visual studio 2022 setup:

ISO C++20. To switch between dll and exe, go to Project Properties -> Configurations Properties -> General -> Configuration Type. 

Using SFML 2.5.1 for node topology display. It can be toggled on and off with the DRAWING preprocessor directive in dllMain.h or main.cpp before building. 

If SFML is used, the following DLLs must be placed in the same folder as the executable, whether compiling a .exe or a .dll :

### In debug mode, in RECURSIVE_NODES\x64\Debug:

  sfml-graphics-d-2.dll     sfml-system-d-2.dll     sfml-window-d-2.dll
  
  
### In release mode, in RECURSIVE_NODES\x64\Release:

  sfml-graphics-2.dll      sfml-system-2.dll     sfml-window-2.dll
  
 
##### To obtain those, download SFML 2.5.1.

#### Details

- I recommend toggling adress Sanitizer on (if you have it installed) when debuging the .exe . It MUST be disabled when debuging the DLL because of a VS bug. Found in  Project Properties -> C/C++ -> General -> Enable Adress Sanitizer  . 

- Sciplot's headers are in the repo but unused yet.
