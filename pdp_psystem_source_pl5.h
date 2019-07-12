/*
    ABCD-GPU: Simulating Population Dynamics P systems on the GPU, by DCBA 
    ABCD-GPU is a subproject of PMCGPU (Parallel simulators for Membrane 
                                        Computing on the GPU)   
 
    Copyright (c) 2019  Research Group on Natural Computing, Universidad de Sevilla
    					Dpto. Ciencias de la Computación e Inteligencia Artificial
    					Escuela Técnica Superior de Ingeniería Informática,
    					Avda. Reina Mercedes s/n, 41012 Sevilla (Spain)

    Authors: Miguel Ángel Martínez-del-Amor
             Ignacio Pérez-Hurtado
    
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


#ifndef PDP_PSYSTEM_SOURCE_PL5_H_
#define PDP_PSYSTEM_SOURCE_PL5_H_

#include "pdp_psystem_source_binary.h"
#include <iostream>

/**********************/
/* Classes for source */
class PDP_Psystem_source_plingua5: public PDP_Psystem_source_binary {
    
public:
	PDP_Psystem_source_plingua5(const char* filename, Options options, std::string& filter_file);
	PDP_Psystem_source_plingua5(Options options);
	~PDP_Psystem_source_plingua5();
	
    /* Public methods inherited from pdp_psystem_source_binary being redefined */
	bool start();

	/* Specific for modular definition of P systems redefined */
	/*int number_of_modules();
	int* modules_start_step();
	int* modules_end_step();
	int* modules_pi_ruleblock_indexes();
	int* modules_env_ruleblock_indexes();*/
};

#endif /* PDP_PSYSTEM_SOURCE_PL5_H_ */
