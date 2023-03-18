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
#define CARTPOLE
#ifndef CARTPOLE
#define XOR
#endif 

// Should be on if there is just 1 trial, or no trials at all. Could be on even if there are multiple trials, 
// but it disables the intertrial update of wLifetime. Not recommended in his case.
// Enable and disable it in Genotype.h, it does not do anything here !
#define CONTINUOUS_LEARNING

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
    int N_SPECIMENS = nThreads * 64;
    int nDifferentTrials = 3;
    int nSteps = 10000;

    // ALL TRIALS MUST HAVE SAME netInSize AND netOutSize
    vector<unique_ptr<Trial>> trials;
    for (int i = 0; i < nDifferentTrials; i++) {
#ifdef CARTPOLE
        trials.emplace_back(new CartPoleTrial()); // Set nDifferentTrials to 5
#elif defined XOR 
        trials.emplace_back(new XorTrial(3,5));  // Set nDifferentTrials to vSize * vSize
#endif
    }

    Population population(trials[0]->netInSize, trials[0]->netOutSize, N_SPECIMENS);
    population.setEvolutionParameters(.0f, .15f, 1.0f); 
    //population.setEvolutionParameters(.0f, .15f, 0.0f);
    int nTrialsEvaluated = (int)trials.size();
    //int nTrialsEvaluated = 2;
    //int nTrialsEvaluated = (int)trials.size() / 4;

    LOG("Using " << nThreads << ".")
    LOG("N_SPECIMEN = " << N_SPECIMENS << " and N_TRIALS = " << nDifferentTrials);
    LOG("Evaluating on the last " << nTrialsEvaluated << " trials.");

    // Evolution loop :
    population.startThreads(nThreads);
    for (int i = 0; i < nSteps; i++) {
#ifdef DRAWING
        drawer.draw(population.getSpecimenPointer(population.fittestSpecimen));
#endif
        //float f0 = 1.0f * (1.0f + sinf((float)i / 2.0f));
        //population.setEvolutionParameters(f0, .2f, true);
        population.step(trials, nTrialsEvaluated);
        if ((i + 1) % 100 == 0) { // defragmentate.
            string fileName = population.save();
            population.load(fileName);
        }
    }
    population.stopThreads();


#ifdef CARTPOLE
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

    // scipy tests
   /* Vec x = linspace(1, N_SPECIMENS, N_SPECIMENS);
    Plot2D plot;

    plot.xlabel("x");
    plot.ylabel("y");
    plot.xrange(0.0, N_SPECIMENS);
    plot.yrange(-2.0, 2.0);

    plot.drawCurve(x, population.).label("sin(x)");

    Figure fig = { {plot} };
    Canvas canvas = { {fig} };
    canvas.show();*/
}
