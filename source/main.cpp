#pragma once

#include <iostream>
#include "Population.h"

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
    /* Network est de taille variable (des attributs sont des vecteurs.)
    vector<Network>  vn(2); // tas ??
    Network vn[2]; // tas ??
    vector<Network*> vn2(2); // pile
    Network n(2, 2); // tas ?
    Network* n2 = new Network(2, 2); //pile 
    */
    Network n(2, 2);
    float input[2] = { 0, 1 };
    n.step(input);
    vector<float> output = n.getOutput();
    LOGV(output);
    return 0;
}
