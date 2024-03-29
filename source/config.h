#pragma once


////////////////////////////////////
///// USER COMPILATION CHOICES /////
////////////////////////////////////


// Comment/uncomment or change the value of various preprocessor directives
// to compile different versions of the code. Or use the -D flag.





// Draws one of the fittest specimens at each step, using SFML. Requires the appropriate DLLs 
// alongside the generated executable, details on setup in readme.md .
//#define DRAWING 


// Define the trials on which to evolve. One and only one must be defined if compiling the .exe (or tweak main()). 
// These do not affect the DLL.
//#define CARTPOLE_T
//#define XOR_T
//#define TMAZE_T
//#define N_LINKS_PENDULUM_T
//#define MEMORY_T
#define ROCKET_SIM_T 


// When defined, wLifetime updates take place during the trial and not at the end of it. The purpose is to
// allow for a very long term memory, in parallel with E and H but much slower.
// Should be defined if there is just 1 trial, or equivalently no trials at all. Could be on even if there 
// are multiple trials. 
#define CONTINUOUS_LEARNING

// Defined if and only if there should not be any lifelong (i.e. inter trial) learning, when CONTINUOUS_LEARNING is used.
#ifdef CONTINUOUS_LEARNING
#define ZERO_WL_BEFORE_TRIAL  
#endif

// When defined, tries to deduce information on the desirable mutation direction from weights learned over lifetime.
// Adding a factor decreasing the effect over generations could be interesting, TODO .
// TODO toggle zeroing of wLifetime at trial end. If off, the accumulation should only happen at lifetime's end.
// Note: this requires passing the "accumulator" matrices from the parent to the children, and through Network::combine,
// which is a tedious and error prone task.
//#define GUIDED_MUTATIONS


// When defined, for each network, float sp = sum of a function F of each activation of the network, at each step.
// F is of the kind pow(activation, 2*k), so that its symmetric around 0 and decreasing in [-1,0]. (=> increasing in [0, -1])
// At the end of the lifetime, sp is divided by the numer of steps and the number of activations, as both may differ from
// one specimen to another. The vector of [sp/(nA*nS) for each specimen] is normalized (mean 0 var 1), and for each specimen
// the corresponding value in the vector is substracted to the score in parallel of size and amplitude regularization terms
// when computing fitness. The lower sum(F), the fitter.
#define SATURATION_PENALIZING


// At each inference, a small random proportion of the lifetime quantities is reset to either 0 or a random value. These are,
// when available; wL, H, E, w.
#define DROPOUT



// Several mutually exclusive types of memory. One MUST be active.


// A hopfield/transformer - inspired QueryKeyValue memory with a discrete, increasing number of stored vectors.
//#define QKV_MEMORY

// Instead of a discrete, finite set of memorized vectors and an attention mechanism, memory is programmed with
// ordinary feedforward neural networks, that use online gradient descent during trials. Learning rate and other
// signals are provided by neuromodulation. Activation function is tanh, cost is euclidean distance.
// Pre trial reset does not restore the network to its genotypic state.
#if !defined QKV_MEMORY
#define DNN_MEMORY
#endif

// WARNING as of now, unstable in the long term and blows up to Nan. The paper that inspired this states that it 
// should not happen, but despite my best efforts I cant prevent it. https://arxiv.org/abs/2202.05780
// Sequences are probably too long for the current implementation.
#if !defined QKV_MEMORY && !defined DNN_MEMORY
#define SRWM
#endif

// TODO: a [Self Referencial Weight Matrix applied to a DeltaNet] SR-DeltaNet, (https://arxiv.org/pdf/2202.05780.pdf)
//  adapted to fit within the ReNo paradigm: modulation, evolution, and unsupervised learning.
#if !defined QKV_MEMORY && !defined DNN_MEMORY && !defined SRWM
#define SR_DELTANET
#endif



#ifdef QKV_MEMORY
#define MAX_KERNEL_DIMENSION 100	// can be changed
#define MEMORY_MODULATION_SIZE 3    // DO NOT CHANGE
#elif defined DNN_MEMORY
#define MEMORY_MODULATION_SIZE 3    // DO NOT CHANGE
#elif defined SRWM
#define MEMORY_MODULATION_SIZE 4    // DO NOT CHANGE
#else //SR_DELTANET	
#define MEMORY_MODULATION_SIZE 1    // DO NOT CHANGE
#endif


#define MAX_COMPLEX_CHILDREN_PER_COMPLEX  8
#define MAX_MEMORY_CHILDREN_PER_COMPLEX  3
#define MAX_COMPLEX_INPUT_NODE_SIZE  10          // Does not apply to the top node
#define MAX_COMPLEX_OUTPUT_SIZE  10				 // Does not apply to the top node

#define MAX_MEMORY_INPUT_SIZE  15          
#define MAX_MEMORY_OUTPUT_SIZE  15         

#define MODULATION_VECTOR_SIZE (2 + MEMORY_MODULATION_SIZE)      // DO NOT CHANGE


// TODO : implement DERIVATOR
// CENTERED_TANH(x) = tanhf(x) * expf(-x*x) * 1/.375261
// I dont really know what to expect from non-monotonous functions when it comes to applying 
// hebbian updates... It does not make much sense. But I plan to add cases where activations
// do not use hebbian rules.
#define N_ACTIVATIONS  1 // only activation functions < N_ACTIVATIONS are used
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

// When combining specimens, weights are a function of  (spefitness - f0), where f0 is either the primary parent's fitness
// or the primary parent's parent's fitness (when this directive is = true)
#define USE_PARENT_FITNESS_AS_ZERO false

// Positive integer value. Specimens whose phenotypic distance to the primary parent are below it are not used for combination.
#define CONSANGUINITY_DISTANCE 3


// parameters that have values in the range [0,1] are initialized with mean DECAY_PARAMETERS_BIAS
// These parameters (denoted �) are typically used in exponential moving average updates, i.e.
// X(t+1) = X(t) * (1-�)  +  � * ....
#define DECAY_PARAMETERS_INIT_BIAS .1f



// When defined, presynaptic activities of complexNodes (topNode excepted) are an exponential moving average. Each node 
// be it Modulation, complex, memory or output has an evolved parameter (STDP_decay) that parametrizes the average.
// WARNING only compatible with N_ACTIVATIONS = 1, I havent implemented all the derivatives in complexNode_P::forward yet
#define STDP

// The fixed weights w and biases b are not evolved anymore, but set randomly (w -> uniform(-.1,.1), b->NORMAL or 0) 
// at the beginning of each trial (or lifetime). These parameters are also set per phenotype instance, not per genotype instance.
// When used in conjunction with GUIDED_MUTATIONS, w are no longer accumulated. 
#define RANDOM_WB

// Adds Oja's rule to the ABCD rule. This requires the addition of the matrices delta and storage_delta to InternalConnexion_G, 
// deltas being in the [0, 1] range. The update of E is now :  E = (1-eta)E + eta(ABCD... - delta*yj*yj*w_eff),  where w_eff is the 
// effective weight, something like w_eff = w + alpha * H + wL. 
//#define OJA