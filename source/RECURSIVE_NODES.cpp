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

int g_seed = 100000000;
inline int fastrand2() {
    g_seed = (214013 * g_seed + 2531011);
    int r = (g_seed >> 16) & 0x7FFF;
    LOG(r);
    return g_seed;
}


int main()
{
    Network n(2, 2);
    float input[2] = { 0, 1 };
    vector<float> output;
    output = n.step(input);
    LOGV(output);
    for (int i = 0; i < 100; i++) {
        fastrand2();
    }
    return 0;
}
