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
//#define N_LINKS_PENDULUM_T
//#define MEMORY_T


// When defined, wLifetime updates take place during the trial and not at the end of it. The purpose is to
// allow for a very long term memory, in parallel with E and H but much slower. Better performance overall.
// Should be defined if there is just 1 trial, or equivalently no trials at all. Could be on even if there 
// are multiple trials. 
#define CONTINUOUS_LEARNING


// IN DEVELOPPEMENT
// When defined, tries to deduce information on the desirable mutation direction from weights learned over lifetime.
// WIP. On continuous cartpole, CONTINUOUS_LEARNING + GUIDED_MUTATIONS drastically improve convergence speed. (You 
// may need to increase the value fed to accumulateW in Network::postTrialUpdate(), to 10 or 50., and accumulatorClipRange to
// 1.0 or 2.0 in ComplexNode_G::mutateFloats()). Accumulators are reset to 0 after a call to mutate floats.
// Adding a factor decreasing the effect over generations could be interesting, TODO .
// TODO toggle zeroing of wLifetime at trial end. If off, the accumulation should only happen at lifetime's end.
#define GUIDED_MUTATIONS


// IN DEVELOPPEMENT
// When defined, for each network, float sp = sum of a function F of each activation of the network, at each step.
// F is of the kind pow(activation, 2*k), so that its symmetric around 0 and decreasing in [-1,0]. (=> increasing in [0, -1])
// At the end of the lifetime, sp is divided by the numer of steps and the number of activations, as both may differ from
// one specimen to another. The vector of [sp/(nA*nS) for each specimen] is normalized (mean 0 var 1), and for each specimen
// the corresponding value in the vector is substracted to the score in parallel of size and amplitude regularization terms
// when computing fitness. The lower sum(F), the fitter.
#define SATURATION_PENALIZING

// IN DEVELOPPEMENT
// Instead of a discrete, finite set of memorized vectors and an attention mechanism, memory is programmed with
// ordinary feedforward neural networks, that use backpropagation during lifetime to continuously learn. Learning 
// signals are provided by neuromodulation.
//#define DNN_MEMORY


/*********************************************************************************************************
		Various minor options, that appear here to avoid jumping from one file to another:
**********************************************************************************************************/


	
// Usually, when there are several aspects to a task, an average over performances on each task (here, trial) is used to 
// compute the general performance. This option generalizes the idea, by computing the p-norm of the vector of score per
// sub task for each specimen. (The classic average is proportional to the 1-norm.) Happens after raw score transformation
// (ranking/normalization) and competition term addition, the values are made positive by clamping at 0. This has a significant
// influence on selection, TODO figure a better way to do it.
// The higher p, the more specialists are incentivized. And the lower p, the more generalists strive. 
//#define SPECIALIZATION_INCENTIVE .8f	// = p. Recommended range: [0.6, 10.0]


// Maximum number of generations since last common ancestor of two  (of the) specimens combined to form a new specimen. >= 2.
#define MAX_MATING_DEPTH 10

