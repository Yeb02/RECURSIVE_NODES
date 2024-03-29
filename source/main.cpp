#pragma once

#ifdef _DEBUG
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/control87-controlfp-control87-2?view=msvc-170
// These are incompatible with RocketSim that has many float errors, and should be commented when rocketsim.h and 
// .cpp are included in the project (so exclude them temporarily to use this feature).
#define _CRT_SECURE_NO_WARNINGS
#include <float.h>
unsigned int fp_control_state = _controlfp(_EM_UNDERFLOW | _EM_INEXACT, _MCW_EM);

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
    // Path to where you dumped rocket league collision meshes.
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
    int nSpecimens = nThreads * 128; //16 -> 512 in most cases
    int nDifferentTrials = 12;
    int nSteps = 10000;

    // ALL TRIALS IN THE VECTOR MUST HAVE SAME netInSize AND netOutSize. When this condition is met
    // different kinds of trials can be put in the vector.
    vector<unique_ptr<Trial>> trials;
    for (int i = 0; i < nDifferentTrials; i++) {
#ifdef CARTPOLE_T
        trials.emplace_back(new CartPoleTrial(true)); // bool : continuous control.
#elif defined XOR_T
        trials.emplace_back(new XorTrial(4,5));  // int : vSize, int : delay
#elif defined TMAZE_T
        trials.emplace_back(new TMazeTrial(false));
#elif defined N_LINKS_PENDULUM_T
        trials.emplace_back(new NLinksPendulumTrial(false, 2));
#elif defined MEMORY_T
        trials.emplace_back(new MemoryTrial(2, 5, 5, true)); // int nMotifs, int motifSize, int responseSize, bool binary = true
#elif defined ROCKET_SIM_T
        trials.emplace_back(new RocketSimTrial());
#endif
    }
    
    // In visual studio, hover your cursor on the parameters name to read their description. They are initialized 
    // by default to safe values, the initialization below is just for demonstration purposes.
    PopulationEvolutionParameters params;
    params.selectionPressure = { -2.0f, .3f}; 
    params.useSameTrialInit = false; 
    params.rankingFitness = true;
    params.saturationFactor = .08f;
    params.regularizationFactor = .03f; 
    params.competitionFactor = .0f; 
    params.scoreBatchTransformation = NONE; // NONE recommended when useSameTrialInit = false
    params.nParents = 10;

    Population population(trials[0]->netInSize, trials[0]->netOutSize, nSpecimens);
    population.setEvolutionParameters(params); 

    // Only the last _nTrialsEvaluated are used for fitness calculations. Others are only used for learning.
    // DO NOT USE A VALUE DIFFERENT FROM nDifferentTrials, AS OF NOW IT WILL CAUSE A CRASH. TODO adapt parentData
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

#ifdef TMAZE_T
        bool switchesSide = false;
        for (int j = 0; j < nDifferentTrials; j++) {
            switchesSide = UNIFORM_01 < std::min((float)i / 1000.0f, .5f);
            trials[j]->outerLoopUpdate(&switchesSide);
        }
#endif

#ifdef ROCKET_SIM_T
        if (i < 10.0f) {
            float jbt[3] = { .002f, .002f, .002f };
            for (int j = 0; j < nDifferentTrials; j++) {
                trials[j]->outerLoopUpdate(&jbt);
            }
        }
#endif

        for (int j = 0; j < nDifferentTrials; j++) {
            trials[j]->reset(false);
        }


        float c = std::max(.0F, 1.0f - (float)i / 20.0f);
        params.selectionPressure.first = (1.0f - c) * (1.5f * sinf((float)i / 3.0f) - 1.5f) + c * -1.0f;
        LOG("Pressure was : " << params.selectionPressure.first);
        population.setEvolutionParameters(params); // parameters can be changed at each step.

        population.step(trials, _nTrialsEvaluated);

        if ((i + 1) % 30 == 0) {
            population.saveFittestSpecimen();
        }
    }

    population.stopThreads();

    // Tests.
    if (false) {
        std::ifstream is("models\\topNet_1685971637922_631.renon", std::ios::binary);
        LOG(is.is_open());
        Network* n = new Network(is);
        LOG("Loaded.");
        n->createPhenotype();
        n->preTrialReset();
        trials[0]->reset(true);
        float avg_thr = 0.0f;
        while (!trials[0]->isTrialOver) {
            n->step(trials[0]->observations);
            trials[0]->step(n->getOutput());
            LOG(n->getOutput()[0]);
        }
        delete n;
        LOG("Reloaded best specimen's score on the same trial = " << trials[0]->score);
    }

    return 0;
}
