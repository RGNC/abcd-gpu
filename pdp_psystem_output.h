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
 * File:   pdp_psystem_output.h
 * Author: miguel
 *
 * Created on 20 de enero de 2014, 17:52
 */

#ifndef PDP_PSYSTEM_OUTPUT_H
#define	PDP_PSYSTEM_OUTPUT_H 

#include "pdp_psystem.h"
#include <cstddef>

// TODO: Expand this general class with more common attributes and methods
// The class for PDP system outputs
class PDP_Psystem_output {

protected:
	Options options;
	//char * filename;
	
public:

	virtual bool write_configuration(unsigned int* config_multisets, char * config_charges, int sim, int step, char** objstrings)=0;
	virtual bool write_configuration_filtered(unsigned int* filtered_multisets, char * config_charges, int sim, int step, char** objstrings)=0;
};


#endif	/* PDP_PSYSTEM_OUTPUT_H */

