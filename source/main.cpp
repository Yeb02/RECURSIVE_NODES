#pragma once

#include <iostream>

#include "sciplot/sciplot.hpp"

#include "Population.h"
#include "Random.h"

#define LOGV(v) for (const auto e : v) {cout << e << " ";}; cout << "\n"
#define LOG(x) cout << x << endl;

/*
//#define RISI_NAJARRO_2020
//#define USING_NEUROMODULATION
are the 2 mutually exclusive running modes. Change in Genotype.h.
*/

#define DRAWING
//#define XOR
#ifndef XOR
#define CARTPOLE
#endif 


#ifdef DRAWING
#include "Drawer.h"
#endif 

using namespace std;
using namespace sciplot;

int main()
{

#ifdef DRAWING
    Drawer drawer(720, 480);
#endif

    int nThreads = std::thread::hardware_concurrency();
    LOG(nThreads << " concurrent threads are supported at hardware level.");
    int N_SPECIMENS = nThreads * 64;
    int nDifferentTrials = 10;
    int nSteps = 100;

    // ALL TRIALS MUST HAVE SAME netInSize AND netOutSize
    vector<unique_ptr<Trial>> trials;
    for (int i = 0; i < nDifferentTrials; i++) {
#ifdef CARTPOLE
        trials.emplace_back(new CartPoleTrial()); // Set nDifferentTrials to 3
#elif defined XOR 
        trials.emplace_back(new XorTrial(3));  // Set nDifferentTrials to vSize * vSize
#endif
    }

    Population population(trials[0]->netInSize, trials[0]->netOutSize, N_SPECIMENS);

    LOG("Using " << nThreads << ".")
    LOG("N_SPECIMEN = " << N_SPECIMENS << " and N_TRIALS = " << nDifferentTrials);



    // Evolution loop :

    population.startThreads(nThreads);
    for (int i = 0; i < nSteps; i++) {
#ifdef DRAWING
        drawer.draw(population.getSpecimenPointer(population.fittestSpecimen));
#endif

        population.step(trials);
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
