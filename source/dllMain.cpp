#pragma once

#include "dllMain.h"

using namespace std;


BOOL DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {

    switch (fdwReason)
    {
    case DLL_PROCESS_ATTACH:
        // Code to run when the DLL is loaded
        printf("Recursive nodes dll loaded.\n");
        break;

    case DLL_PROCESS_DETACH:
        // Code to run when the DLL is freed
        printf("Recursive nodes dll freed.\n");
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

