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
    
    /*Network n(2, 2);
    vector<float> input = { 0, 1 };
    n.step(input);
    vector<float> output = n.getOutput();
    LOGV(output);*/
    {
        vector<Trial*> trials;
        for (int i = 0; i < 8; i++) trials.push_back(new XorTrial(2));
        Population population(2, 2, 101);
        for (int i = 0; i < 1000; i++) {
            population.step(trials);
        }
    }
    return 0;
}
