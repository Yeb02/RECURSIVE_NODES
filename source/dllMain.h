#pragma once

#include "Population.h"

#define NOMINMAX
#include <windows.h>

extern "C" {

	BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved);

}