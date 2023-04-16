#pragma once


// TODO : implement DERIVATOR, that outputs the difference between INPUT_NODE at this step and INPUT_NODE at the previous step.

// CENTERED_SINE(x) = tanhf(x) * expf(-x*x) * 1/.375261
// I dont really know what to expect from SINE and CENTERED_SINE when it comes to applying 
// hebbian updates... It does not make much sense. But I plan to add cases where activations
// do not use hebbian rules.
const enum ACTIVATION { TANH = 0, GAUSSIAN = 1, SINE = 2, CENTERED_SINE = 3};


struct SimpleNode_G {
	// the activation function this node uses.
	ACTIVATION activation;

	// util for Network. The position in the simpleGenome array.
	int position;

	SimpleNode_G(SimpleNode_G* n) {
		activation = n->activation;
		position = n->position;
	}

	SimpleNode_G(ACTIVATION a) {
		activation = a;
		position = -1;
	}
};