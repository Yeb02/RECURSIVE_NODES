# RECURSIVE_NODES

Implements a 2 (aiming for 3) loops optimization algorithm to solve optimal control and memorization problems. The neural network is meta-optimized by a custom genetic algorithm, which evolves both topology and floating point parameters. The network itself uses 3 factor Hebbian learning rules. 

##### KNOWN BUG: rare infinite recursion in updateDepth (genotype.cpp), working on it.

The code can be compiled into an executable, to experiment on the implemented c++ trials. As of now, 2 trials are implemented: XoR with memorization, and gym's cartpole. The algorithm solves both easily. More to come !

It can also be compiled into a dll, to use in python with ctypes. One can then evaluate the algorithm on any python trial, like openAI's gym's for instance. Work in progress.

To switch between dll and exe, go to Project Properties -> Configurations Properties -> General -> Configuration Type. 

## Visual studio 2022 setup:

ISO C++20, requires AdressSanitizer to run in debug mode. 


IF the project is compiled with the DRAWING preprocessor directive, and IF it is compiled into an exe (not dll), the following DLLs must be placed in the same folder as the executable 
### In debug mode, in RECURSIVE_NODES\x64\Debug:

  sfml-graphics-d-2.dll     sfml-system-d-2.dll     sfml-window-d-2.dll
  
  
### In release mode, in RECURSIVE_NODES\x64\Release:

  sfml-graphics-2.dll      sfml-system-2.dll     sfml-window-2.dll
  
 
##### To obtain those, download SFML 2.5.1.
