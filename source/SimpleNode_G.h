#pragma once


// TODO : implement DERIVATOR, that outputs the difference between INPUT_NODE at this step and INPUT_NODE at the previous step.

// I dont really know what to expect from SINE when it comes to applying hebbian updates...
const enum ACTIVATION { TANH = 0, GAUSSIAN = 1, SINE = 2 };


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