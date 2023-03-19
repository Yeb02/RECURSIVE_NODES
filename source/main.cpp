#pragma once
#include <iostream>
#include "Population.h"
#include "Random.h"

////////////////////////////////////
///// USER COMPILATION CHOICES /////
////////////////////////////////////

// Comment or uncomment the preprocessor directives to compile versions of the code
// Or use the -D flag.

// Draws a specimen at each step, using SFML. Requires the appropriate DLLs 
// alongside the generated executable, details in readme.md .
#define DRAWING 


// Define the trials on which to evolve. One and only one must be defined: (or tweak main())
//#define CARTPOLE_T
//#define XOR_T
//#define TMAZE_T
#define POLYNOME_T

// When defined, wLifetime updates take place during the trial and not at the end of it. The purpose is to
// allow for a very long term memory, in parallel with E and H but much slower.
// Should be on if there is just 1 trial, or equivalently no trials at all. Could be on even if there 
// are multiple trials, but not recommended in this case. Worth a shot anyway.
// Define or undefine it in Genotype.h, it does not do anything here !!!!
#define CONTINUOUS_LEARNING


#define GUIDED_MUTATIONS

////////////////////////////////////
////////////////////////////////////


#ifdef DRAWING
#include "Drawer.h"
#endif 

#define LOGV(v) for (const auto e : v) {cout << e << " ";}; cout << "\n"
#define LOG(x) cout << x << endl;

using namespace std;


int main()
{
#ifdef DRAWING
    Drawer drawer(720, 480);
#endif

    int nThreads = std::thread::hardware_concurrency();
    LOG(nThreads << " concurrent threads are supported at hardware level.");
#ifdef _DEBUG
    nThreads = 1;
#endif
    int nSpecimens = nThreads * 64;
    int nDifferentTrials = 8;
    int nSteps = 1000;

    // ALL TRIALS IN THE VECTOR MUST HAVE SAME netInSize AND netOutSize. When this condition is met
    // different kinds of trials can be put in the vector.
    vector<unique_ptr<Trial>> trials;
    for (int i = 0; i < nDifferentTrials; i++) {
#ifdef CARTPOLE_T
        trials.emplace_back(new CartPoleTrial(true)); // Parameter corresponds to continuous control.
#elif defined XOR_T
        trials.emplace_back(new XorTrial(1,10));  
#elif defined TMAZE_T
        trials.emplace_back(new TMazeTrial(false));
#elif defined POLYNOME_T
        trials.emplace_back(new PolynomeTrial(20, 2));
#endif
    }

    Population population(trials[0]->netInSize, trials[0]->netOutSize, nSpecimens);
    population.setEvolutionParameters(.2f, .25f, .0f); 
    int nTrialsEvaluated = (int)trials.size() / 4;          // (int)trials.size(), or 4, or (int)trials.size() / 4, ...


    LOG("Using " << nThreads << ".");
    LOG("N_SPECIMEN = " << nSpecimens << " and N_TRIALS = " << nDifferentTrials);
    LOG("Evaluating on the last " << nTrialsEvaluated << " trials.");


    // Evolution loop :
    population.startThreads(nThreads);
    for (int i = 0; i < nSteps; i++) {
#ifdef DRAWING
        drawer.draw(population.getSpecimenPointer(population.fittestSpecimen));
#endif

        for (int j = 0; j < nDifferentTrials; j++) {
            trials[j]->reset(false);
        }


#ifdef TMAZE_T
        for (int j = 0; j < nDifferentTrials; j++) {
            int switchesSide = UNIFORM_01 > 0.5f;
            trials[j]->outerLoopUpdate(&switchesSide);
        }
#elif defined POLYNOME_T
        for (int j = 1; j < nDifferentTrials; j++) {
            trials[j]->copy(trials[0].get());
        }
#endif

        //population.setEvolutionParameters(sinf((float)i / 2.0f), .1f, 0.0f); // parameters can be changed at each step.
        population.step(trials, nTrialsEvaluated);

        if ((i + 1) % 100 == 0) { // Defragmentate. Not implemented yet.
            string fileName = population.save();
            population.load(fileName);
        }
    }
    population.stopThreads();


#ifdef CARTPOLE_T
    // If the Cartpole trial was used, copy the console output in the "data" array of 
    // RECURSIVE_NODES\python\CartPoleData.py    and run   RECURSIVE_NODES\python\CartPoleVisualizer.py
    // to observe the behaviour !
    Network* n = population.getSpecimenPointer(population.fittestSpecimen);
    trials[0]->reset();
    n->intertrialReset();
    cout << "\n";
    while (!trials[0]->isTrialOver) {
        n->step(trials[0]->observations);
        trials[0]->step(n->getOutput());
        cout << ", " << trials[0]->observations[0] << ", " << trials[0]->observations[2];
    }
    cout << "\n" << trials[0]->score;
#endif

    return 0;
}


// scipy tests
   /* Vec x = linspace(1, nSpecimens, nSpecimens);
    Plot2D plot;

    plot.xlabel("x");
    plot.ylabel("y");
    plot.xrange(0.0, nSpecimens);
    plot.yrange(-2.0, 2.0);

    plot.drawCurve(x, population.).label("sin(x)");

    Figure fig = { {plot} };
    Canvas canvas = { {fig} };
    canvas.show();*/
