#pragma once
#include <iostream>
#include "Population.h"
#include "Random.h"


#ifdef DRAWING
#include "Drawer.h"
#endif 

#define LOGV(v) for (const auto e : v) {cout << std::setprecision(2)<< e << " ";}; cout << "\n"
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
    int nDifferentTrials = 4;
    int nSteps = 10000;

    // ALL TRIALS IN THE VECTOR MUST HAVE SAME netInSize AND netOutSize. When this condition is met
    // different kinds of trials can be put in the vector.
    vector<unique_ptr<Trial>> trials;
    for (int i = 0; i < nDifferentTrials; i++) {
#ifdef CARTPOLE_T
        trials.emplace_back(new CartPoleTrial(true)); // Parameter corresponds to continuous control.
#elif defined XOR_T
        trials.emplace_back(new XorTrial(1,5));  
#elif defined TMAZE_T
        trials.emplace_back(new TMazeTrial(false));
#elif defined N_LINKS_PENDULUM_T
        trials.emplace_back(new NLinksPendulumTrial(false, 2));
#elif defined MEMORY_T
        trials.emplace_back(new MemoryTrial(2, 4, 2, true));
#endif
    }

    // In visual studio, hover your cursor on the parameters name to see their description ! They are initialized 
    // by default to safe values, the initilaization below is just for demonstration purposes.
    PopulationEvolutionParameters params;
    params.selectionPressure = -0.0f;
    params.nichingNorm = 0.0f;
    params.useSameTrialInit = true;
    params.normalizedScoreGradients = false;
    params.rankingFitness = true;
    params.saturationFactor = 0.05f;
    params.regularizationFactor = 0.05f; // +2.0f * params.saturationFactor;
    params.targetNSpecimens = 0;

    Population population(trials[0]->netInSize, trials[0]->netOutSize, nSpecimens);
    population.setEvolutionParameters(params); 

    // Only the last _nTrialsEvaluated are used for fitness calculations. Others are only used for learning.
    int _nTrialsEvaluated = nDifferentTrials ;   // = (int)trials.size(), or 4, or (int)trials.size() / 4, ...


    LOG("Using " << nThreads << ".");
    LOG("N_SPECIMEN = " << nSpecimens << " and N_TRIALS = " << nDifferentTrials);
    LOG("Evaluating on the last " << _nTrialsEvaluated << " trials.");


    // Evolution loop :
    population.startThreads(nThreads);
    for (int i = 0; i < nSteps; i++) {
#ifdef DRAWING
        drawer.draw(population.getSpecimenPointer(population.fittestSpecimen), i);
        if (drawer.paused) {
            i--;
            continue;
        }
#endif

        for (int j = 0; j < nDifferentTrials; j++) {
            trials[j]->reset(false);
        }


#ifdef TMAZE_T
        bool switchesSide = false;
        for (int j = 0; j < nDifferentTrials; j++) {
            switchesSide = UNIFORM_01 < std::min((float)i / 1000.0f, .5f);
            trials[j]->outerLoopUpdate(&switchesSide);
        }
#endif

        // params.selectionPressure = sinf((float)i / 2.0f) - .5f;
        //population.setEvolutionParameters(params); // parameters can be changed at each step.
        population.step(trials, _nTrialsEvaluated);


        //if ((i + 1) % 10000 == 0) population.defragmentate(); // Defragmentate. Not implemented yet.
    }
    population.stopThreads();

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
