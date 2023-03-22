#pragma once


////////////////////////////////////
///// USER COMPILATION CHOICES /////
////////////////////////////////////

// Comment or uncomment the preprocessor directives to compile versions of the code
// Or use the -D flag.



// Draws one of the fittest specimens at each step, using SFML. Requires the appropriate DLLs 
// alongside the generated executable, details on setup in readme.md .
#define DRAWING 


// Define the trials on which to evolve. One and only one must be defined if compiling the .exe (or tweak main()). 
// These do not affect the DLL.
#define CARTPOLE_T
//#define XOR_T
//#define TMAZE_T


// When defined, wLifetime updates take place during the trial and not at the end of it. The purpose is to
// allow for a very long term memory, in parallel with E and H but much slower. Better performance overall.
// Should be defined if there is just 1 trial, or equivalently no trials at all. Could be on even if there 
// are multiple trials. 
#define CONTINUOUS_LEARNING


// When defined, tries to deduce information on the desirable mutation direction from weights learned over lifetime.
// Still experimenting. On continuous cartpole, CONTINUOUS_LEARNING + GUIDED_MUTATIONS drastically improve convergence speed, 
// but makes it much more noisy. So adding a factor decreasing its amplitude over generations could be useful.
// At the opposite GUIDED_MUTATIONS, with CONTINUOUS_LEARNING disabled, impressively improves convergence stability,
// while being more or less as fast as without GUIDED_MUTATIONS. Very interesting behaviours to be investigated.
// Conducted with population.setEvolutionParameters(-1.0f, .1f, .0f, true), nTrials = 8.
#define GUIDED_MUTATIONS
