#pragma once
#include <iostream>
#include "Population.h"
#include "Random.h"


#ifdef DRAWING
#include "Drawer.h"
#endif 

#define LOGV(v) for (const auto e : v) {cout << e << " ";}; cout << "\n"
#define LOG(x) cout << x << endl;

using namespace std;


int main()
{
#ifdef DRAWING
    Drawer drawer(1080, 480);
#endif

    int nThreads = std::thread::hardware_concurrency();
    LOG(nThreads << " concurrent threads are supported at hardware level.");
#ifdef _DEBUG
    nThreads = 1;
#endif
    int nSpecimens = nThreads * 64;
    int nDifferentTrials = 8;
    int nSteps = 10000;


    // ALL TRIALS IN THE VECTOR MUST HAVE SAME netInSize AND netOutSize. When this condition is met
    // different kinds of trials can be put in the vector.
    vector<unique_ptr<Trial>> trials;
    for (int i = 0; i < nDifferentTrials; i++) {
#ifdef CARTPOLE_T
        trials.emplace_back(new CartPoleTrial(true)); // Parameter corresponds to continuous control.
#elif defined XOR_T
        trials.emplace_back(new XorTrial(2,5));  
#elif defined TMAZE_T
        trials.emplace_back(new TMazeTrial(false));
#endif
    }

    Population population(trials[0]->netInSize, trials[0]->netOutSize, nSpecimens);
    population.setEvolutionParameters(-1.0f, .1f, .0f, true); 
    // Only the last _nTrialsEvaluated are used for fitness calculations. Others are only used for learning.
    int _nTrialsEvaluated = nDifferentTrials ;          // (int)trials.size(), or 4, or (int)trials.size() / 4, ...


    LOG("Using " << nThreads << ".");
    LOG("N_SPECIMEN = " << nSpecimens << " and N_TRIALS = " << nDifferentTrials);
    LOG("Evaluating on the last " << _nTrialsEvaluated << " trials.");


    // Evolution loop :
    population.startThreads(nThreads);
    for (int i = 0; i < nSteps; i++) {
#ifdef DRAWING
        drawer.draw(population.getSpecimenPointer(population.fittestSpecimen));
        if (drawer.paused) {
            i--;
            continue;
        }
#endif

        for (int j = 0; j < nDifferentTrials; j++) {
            trials[j]->reset(false);
        }


#ifdef TMAZE_T
        for (int j = 0; j < nDifferentTrials; j++) {
            int switchesSide = UNIFORM_01 > 0.5f;
            trials[j]->outerLoopUpdate(&switchesSide);
        }
#endif


        //population.setEvolutionParameters(sinf((float)i / 2.0f), .1f, 0.0f, false); // parameters can be changed at each step.
        population.step(trials, _nTrialsEvaluated);

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
    n->createPhenotype();
    n->preTrialReset();
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
