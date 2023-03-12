#pragma once

#include "dllMain.h"

using namespace std;


/*
//#define RISI_NAJARRO_2020
//#define USING_NEUROMODULATION
are the 2 mutually exclusive running modes. Change in Genotype.h.
//#define DYNAMIC_MUTATION_P make mutation rate mutable. Optional, and usually worsens results.
*/

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {

    switch (fdwReason)
    {
    case DLL_PROCESS_ATTACH:
        // Code to run when the DLL is loaded
        printf("Load working...\n");
        break;

    case DLL_PROCESS_DETACH:
        // Code to run when the DLL is freed
        printf("Unload working...\n");
        break;

    case DLL_THREAD_ATTACH:
        // Code to run when a thread is created during the DLL's lifetime
        printf("ThreadLoad working...\n");
        break;

    case DLL_THREAD_DETACH:
        // Code to run when a thread ends normally.
        printf("ThreadUnload working...\n");
        break;
    }

    return TRUE;
}