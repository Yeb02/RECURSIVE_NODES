#pragma once

#ifdef _DEBUG
#define _CRT_SECURE_NO_WARNINGS
#include <float.h>
unsigned int fp_control_state = _controlfp(_EM_INEXACT, _MCW_EM);
#endif

#include <iostream>
#include "Population.h"
#include "Random.h"

#ifdef ROCKET_SIM_T
#include "RocketSim.h"
#endif


#ifdef DRAWING
#include "Drawer.h"
#endif 

#define LOGV(v) for (const auto e : v) {cout << std::setprecision(2)<< e << " ";}; cout << "\n"
#define LOG(x) cout << x << endl;

using namespace std;



int main()
{
    LOG("Seed : " << seed);

#ifdef ROCKET_SIM_T
    RocketSim::Init((std::filesystem::path)"C:/Users/alpha/Bureau/RLRL/collisionDumper/x64/Release/collision_meshes");
#endif

#ifdef DRAWING
    Drawer drawer(1080, 480);
#endif

    int nThreads = std::thread::hardware_concurrency();
    LOG(nThreads << " concurrent threads are supported at hardware level.");
#ifdef _DEBUG
    nThreads = 1; // Because multi-threaded functions are difficult to step through line by line.
#endif
    int nSpecimens = nThreads * 256; //16 -> 512
    int nDifferentTrials = 4;
    int nSteps = 2000;

    // ALL TRIALS IN THE VECTOR MUST HAVE SAME netInSize AND netOutSize. When this condition is met
    // different kinds of trials can be put in the vector.
    vector<unique_ptr<Trial>> trials;
    for (int i = 0; i < nDifferentTrials; i++) {
#ifdef CARTPOLE_T
        trials.emplace_back(new CartPoleTrial(true)); // bool : continuous control.
#elif defined XOR_T
        trials.emplace_back(new XorTrial(2,15));  // int : vSize, int : delay
#elif defined TMAZE_T
        trials.emplace_back(new TMazeTrial(false));
#elif defined N_LINKS_PENDULUM_T
        trials.emplace_back(new NLinksPendulumTrial(false, 2));
#elif defined MEMORY_T
        trials.emplace_back(new MemoryTrial(1, 2, 2, true)); // int nMotifs, int motifSize, int responseSize, bool binary = true
#elif defined ROCKET_SIM_T
        trials.emplace_back(new RocketSimTrial());
#endif
    }

    // In visual studio, hover your cursor on the parameters name to read their description. They are initialized 
    // by default to safe values, the initialization below is just for demonstration purposes.
    PopulationEvolutionParameters params;
    params.selectionPressure = { -2.0f, .3f}; // first param < -1, second << 1.
    params.useSameTrialInit = true; 
    params.rankingFitness = true;
    params.saturationFactor = 0.0f;
    params.regularizationFactor = 0.05f; 
    params.competitionFactor = .0f; 
    params.scoreBatchTransformation = RANK;
    params.nParents = 20;

    Population population(trials[0]->netInSize, trials[0]->netOutSize, nSpecimens);
    population.setEvolutionParameters(params); 

    // Only the last _nTrialsEvaluated are used for fitness calculations. Others are only used for learning.
    // DO NOT USE A VALUE DIFFERENT OF nDifferentTrials, AS OF NOW IT WILL CAUSE A CRASH. TODO adapt parentData
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

        // params.selectionPressure.second = sinf((float)i / 2.0f) - .5f;
        //population.setEvolutionParameters(params); // parameters can be changed at each step.
        population.step(trials, _nTrialsEvaluated);

        if ((i + 1) % 30 == 0) {
            population.saveFittestSpecimen();
        }
    }

    population.stopThreads();

    // Tests.
    if (false) {
        std::ifstream is("topNet_1685814432215_3.renon", std::ios::binary);
        trials[0]->reset(true);
        Network* n = new Network(is);
        LOG("Loaded.")
            n->createPhenotype();
        n->preTrialReset();
        while (!trials[0]->isTrialOver) {
            n->step(trials[0]->observations);
            trials[0]->step(n->getOutput());
        }
        delete n;
        LOG("Reloaded best specimen's score on the same trial = " << trials[0]->score);
    }

    return 0;
}
