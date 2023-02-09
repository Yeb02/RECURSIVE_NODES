#pragma once

#include <iostream>
#include "Population.h"
#include "Random.h"

#define LOGV(v) for (const auto e : v) {cout << e << " ";}; cout << "\n"
#define LOG(x) cout << x << " ";

using namespace std;

// TODO  !
/*
Connecter les enfants à la neuromodulation, et pas le parent !
    
Moins important, essayer d'inverser l'ordre de propagation du signal de neuromodulation.
Plutot que de passer du parent aux enfants, passer de l'enfant aux connexions 
le concernant. Mais comment faire pour les simples neurones ?
*/


int main()
{
    vector<float> trialInput = { 0.1f };
    XorTrial xorTrial(1);
    xorTrial.reset();
    while (!xorTrial.isTrialOver) xorTrial.step(trialInput);
    LOG(xorTrial.score);
    Network n(2, 2);
    vector<float> input = { 0, 1 };
    n.step(input);
    vector<float> output = n.getOutput();
    LOGV(output);
    return 0;
}
