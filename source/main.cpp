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
//#define DYNAMIC_MUTATION_P make mutation rate mutable. Optional, and usually worsens results.
*/

#define DRAWING
#ifdef DRAWING
#include "Drawer.h"
#endif 

using namespace std;
using namespace sciplot;

int main()
{

#ifdef DRAWING
    sf::RenderWindow window(sf::VideoMode(720, 480), "Top Node");
    Drawer drawer(window);
#endif

    int nThreads = std::thread::hardware_concurrency();
    LOG(nThreads << " concurrent threads are supported at hardware level.");
    int N_SPECIMENS = nThreads * 64;
    int nDifferentTrials = 10;
    int nSteps = 5000;



    // ALL TRIALS MUST HAVE SAME netInSize AND netOutSize
    vector<unique_ptr<Trial>> trials;
    for (int i = 0; i < nDifferentTrials; i++) {
        //trials.emplace_back(new CartPoleTrial()); // Set nDifferentTrials to 3
        trials.emplace_back(new XorTrial(3));  // Set nDifferentTrials to vSize * vSize
    }

    Population population(trials[0]->netInSize, trials[0]->netOutSize, N_SPECIMENS);

    LOG("Using " << nThreads << ".")
    LOG("N_SPECIMEN = " << N_SPECIMENS << " and N_TRIALS = " << nDifferentTrials);

    // evolution loop
    population.startThreads(nThreads);
    for (int i = 0; i < nSteps; i++) {
#ifdef DRAWING
        window.clear();
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        drawer.draw(population.getFittestSpecimenPointer());
#endif

        population.step(trials);
        if ((i + 1) % 100 == 0) { // defragmentate.
            string fileName = population.save();
            population.load(fileName);
        }

#ifdef DRAWING
        window.display();
#endif
    }
    population.stopThreads();

    {
        // If the Cartpole trial was used, copy the console output in the "data" array of 
        // RECURSIVE_NODES\python\CartPoleData.py    and run   RECURSIVE_NODES\python\CartPoleVisualizer.py
        // to observe the behaviour !
        Network* n = population.getFittestSpecimenPointer();
        trials[0]->reset();
        n->intertrialReset();
        cout << "\n";
        while (!trials[0]->isTrialOver) {
            n->step(trials[0]->observations);
            trials[0]->step(n->getOutput());
            cout << ", " << trials[0]->observations[0] << ", " << trials[0]->observations[2];
        }
        cout << "\n" << trials[0]->score;
        delete n;
    }

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
