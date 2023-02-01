#include <iostream>
#include "Network.h"

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
    Network n(2, 2);
    float input[2] = { 0, 1 };
    vector<float> output;
    output = n.step(input);
    LOGV(output);
    return 0;
}
