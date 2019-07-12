/*
    ABCD-GPU: Simulating Population Dynamics P systems on the GPU, by DCBA
    ABCD-GPU is a subproject of PMCGPU (Parallel simulators for Membrane
                                        Computing on the GPU)

    Copyright (c) 2015  Research Group on Natural Computing, Universidad de Sevilla
    					Dpto. Ciencias de la Computación e Inteligencia Artificial
    					Escuela Técnica Superior de Ingeniería Informática,
    					Avda. Reina Mercedes s/n, 41012 Sevilla (Spain)

    Author: Miguel Ángel Martínez-del-Amor

    This file is part of ABCD-GPU.

    ABCD-GPU is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ABCD-GPU is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ABCD-GPU.  If not, see <http://www.gnu.org/licenses/>. */


/*
 * PDP_Psystem_output_binary.h
 *
 *  Created on: 26/11/2014
 *      Author: miguel
 */

#ifndef PDP_PSYSTEM_OUTPUT_BINARY_H_
#define PDP_PSYSTEM_OUTPUT_BINARY_H_

#include "binbit.h"
#include "pdp_psystem_output.h"

class PDP_Psystem_output_binary: public PDP_Psystem_output {
private:
	/*Active class*/
	bool donotoutput;
	/* File stuff */
	ofstream outfile;
	char buffer [8];
	unsigned int * myword;
	unsigned short int *myhalfword, *myhalfword2;
	/* Precisions */
	int pobjects;

	/* Cache */
	unsigned int * object4;
	unsigned int * multip4;


	int get_precision(unsigned int number) {
		if (number<256) return 0;
		else if (number < 65536) return 1;
		else return 2;
	}

public:

	PDP_Psystem_output_binary(const char * file,Options opt);

	~PDP_Psystem_output_binary() {
		if (donotoutput) return;
		outfile.close();
		if (myword) delete [] myword;
		if (object4) delete [] object4;
		if (multip4) delete [] multip4;
	}

	bool write_configuration(unsigned int* config_multisets,char * config_charges, int sim, int step, char** objstrings=NULL);
	bool write_configuration_filtered(unsigned int* filtered_multisets,char * config_charges, int sim, int step, char** objstrings=NULL);
};



#endif /* PDP_PSYSTEM_OUTPUT_BINARY_H_ */
