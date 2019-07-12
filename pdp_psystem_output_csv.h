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
 * PDP_Psystem_output_csv.h
 *
 *  Created on: 30/11/2014
 *      Author: miguel
 */

#ifndef PDP_PSYSTEM_OUTPUT_CSV_H_
#define PDP_PSYSTEM_OUTPUT_CSV_H_

#include "pdp_psystem_output.h"
#include <iostream>
#include <fstream>

using namespace std;

class PDP_Psystem_output_csv: public PDP_Psystem_output {
private:
	/*Active class*/
	bool donotoutput;
	/* File stuff */
	ofstream outfile;

public:

	PDP_Psystem_output_csv(const char * file,Options opt);

	~PDP_Psystem_output_csv() {
		if (donotoutput) return;
		outfile.close();
	}

	bool write_configuration(unsigned int* config_multisets,char * config_charges, int sim, int step, char** objstrings=NULL);
};



#endif /* PDP_PSYSTEM_OUTPUT_CSV_H_ */
