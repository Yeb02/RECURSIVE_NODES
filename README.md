# RECURSIVE_NODES

Implements a 2 (aiming for 3) loops optimization algorithm to solve optimal control and memorization problems. The neural network is meta-optimized by a custom genetic algorithm, which evolves both topology and floating point parameters. The network itself uses 3 factor Hebbian learning rules. 

2 trials are implemented: XoR with memorization, and gym's cartpole. The algorithm solves both easily.

## Visual studio setup:

ISO C++20, requires ASan to run in debug mode. The following DLLs must be placed in the same folder as the executable if the project is compiled with the DRAWING flag.
### In debug mode:

  sfml-graphics-d-2.dll     sfml-system-d-2.dll     sfml-window-d-2.dll
  
  
### In release mode:

  sfml-graphics-2.dll      sfml-system-2.dll     sfml-window-2.dll
  
 
##### To obtain those, download SFML 2.5.1.
