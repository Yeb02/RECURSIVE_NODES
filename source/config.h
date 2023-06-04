#pragma once


////////////////////////////////////
///// USER COMPILATION CHOICES /////
////////////////////////////////////


// Comment/uncomment or change the value of various preprocessor directives
// to compile different versions of the code. Or use the -D flag.





// Draws one of the fittest specimens at each step, using SFML. Requires the appropriate DLLs 
// alongside the generated executable, details on setup in readme.md .
#define DRAWING 


// Define the trials on which to evolve. One and only one must be defined if compiling the .exe (or tweak main()). 
// These do not affect the DLL.
#define CARTPOLE_T
//#define XOR_T
//#define TMAZE_T
//#define N_LINKS_PENDULUM_T
//#define MEMORY_T
//#define ROCKET_SIM_T // for now, path are hardcoded in the project properties (linker->input), in main.cpp and in RocketSim.h.


// When defined, wLifetime updates take place during the trial and not at the end of it. The purpose is to
// allow for a very long term memory, in parallel with E and H but much slower. Better performance overall.
// Should be defined if there is just 1 trial, or equivalently no trials at all. Could be on even if there 
// are multiple trials. 
#define CONTINUOUS_LEARNING


// IN DEVELOPPEMENT. Incompatible with RANDOM_W.
// When defined, tries to deduce information on the desirable mutation direction from weights learned over lifetime.
// Adding a factor decreasing the effect over generations could be interesting, TODO .
// TODO toggle zeroing of wLifetime at trial end. If off, the accumulation should only happen at lifetime's end.
//#define GUIDED_MUTATIONS


// IN DEVELOPPEMENT, DO NOT USE
// When defined, for each network, float sp = sum of a function F of each activation of the network, at each step.
// F is of the kind pow(activation, 2*k), so that its symmetric around 0 and decreasing in [-1,0]. (=> increasing in [0, -1])
// At the end of the lifetime, sp is divided by the numer of steps and the number of activations, as both may differ from
// one specimen to another. The vector of [sp/(nA*nS) for each specimen] is normalized (mean 0 var 1), and for each specimen
// the corresponding value in the vector is substracted to the score in parallel of size and amplitude regularization terms
// when computing fitness. The lower sum(F), the fitter.
//#define SATURATION_PENALIZING

// IN DEVELOPPEMENT
// Instead of a discrete, finite set of memorized vectors and an attention mechanism, memory is programmed with
// ordinary feedforward neural networks, that use backpropagation during lifetime to continuously learn. Learning 
// signals are provided by neuromodulation.
//#define DNN_MEMORY



#define MAX_COMPLEX_CHILDREN_PER_COMPLEX  10
#define MAX_MEMORY_CHILDREN_PER_COMPLEX  5
#define MAX_COMPLEX_INPUT_NODE_SIZE  10          // Does not apply to the top node
#define MAX_COMPLEX_OUTPUT_SIZE  10				 // Does not apply to the top node


// TODO : implement DERIVATOR, that outputs the difference between INPUT_NODE at this step and INPUT_NODE at the previous step.
// CENTERED_TANH(x) = tanhf(x) * expf(-x*x) * 1/.375261
// I dont really know what to expect from non-monotonous functions when it comes to applying 
// hebbian updates... It does not make much sense. But I plan to add cases where activations
// do not use hebbian rules.
#define N_ACTIVATIONS  2 // only activation functions < N_ACTIVATIONS are used
const enum ACTIVATION { TANH = 0, GAUSSIAN = 1, LOG2 = 2, EXP2 = 3, RELU = 4, SINE = 5, CENTERED_TANH = 6 };

	
// Usually, when there are several aspects to a task, an average over performances on each task (here, trial) is used to 
// compute the general performance. This option generalizes the idea, by computing the p-norm of the vector of score per
// sub task for each specimen. (The classic average is proportional to the 1-norm.) Happens after raw score transformation
// (ranking/normalization) and competition term addition, the values are made positive by clamping at 0. This has a significant
// influence on selection, TODO figure a better way to do it.
// The higher p, the more specialists are incentivized. And the lower p, the more generalists strive. 
//#define SPECIALIZATION_INCENTIVE .8f	// = p. Recommended range: [0.6, 10.0]


// Maximum number of generations since last common ancestor of two  (of the) specimens combined to form a new specimen. >= 2.
#define MAX_MATING_DEPTH 10

// parameters that have values in the range [0,1] and are stored in R are initialized with mean DECAY_PARAMETERS_STORAGE_BIAS
#define DECAY_PARAMETERS_STORAGE_BIAS -1.0f

// When defined, presynaptic activities of complexNodes (topNode excepted) are an exponential moving average. Each complex node
// has a parameter used by its parent
#define STDP

// The fixed weights w are not evolved anymore, but set randomly (uniform(-.1,.1)) at the beginning of each trial. This also means
// that it is now InternalConnexion_P and not InternalConnexion_G that handles the w matrix. Incompatible with GUIDED_MUTATIONS.
#ifndef GUIDED_MUTATIONS
#define RANDOM_W
#endif

// Adds Oja's rule to the ABCD rule. This requires the addition of the matrices delta and storage_delta to InternalConnexion_G, 
// deltas being in the [0, 1] range. The update of E is now :  E = (1-eta)E + eta(ABCD... - delta*yj*yj*w_eff),  where w_eff is the 
// effective weight, something like w_eff = w + alpha * H + wL.    Delta_storage are initialized centered on -1.
#define OJA