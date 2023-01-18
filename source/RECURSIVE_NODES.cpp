#include <iostream>
#include "Network.h"

#define LOGV(v) for (const auto e : v) {cout << e;}; cout << "\n";

using namespace std;

int main()
{
    Network n(2, 2);
    float input[2] = { 0, 1 };
    vector<float> output;
    output = n.step(input);
    LOGV(output);
    return 0;
}
