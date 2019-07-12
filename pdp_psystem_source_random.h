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
 * PDP_psystem_source.h
 *
 *  Created on: 15/08/2011
 *      Author: miguel
 */

#ifndef PDP_PSYSTEM_SOURCE_RANDOM_H_
#define PDP_PSYSTEM_SOURCE_RANDOM_H_

#include "pdp_psystem_source.h"
#include <stdio.h>

/**********************/
/* Classes for source */
class PDP_Psystem_source_random: public PDP_Psystem_source {
public:
	PDP_Psystem_source_random(Options options);
	~PDP_Psystem_source_random();

private:
	/* Auxiliary structures */
	short int *lengthU;
	short int *lengthV;
	unsigned int *active_membrane;
	short int *rules;
	short int *lengthUp;
	short int *lengthVp;
	unsigned int* obj_lhs,* obj_rhs;
	float * prob;
	short int *env_lengthU;

	/* Iterators */
	unsigned int rule_block_it;
	unsigned int U_it;
	unsigned int V_it;
	unsigned int Up_it;
	unsigned int Vp_it;
	unsigned int rule_it;

	unsigned int block_env_it;
	unsigned int rule_env_it;

	unsigned int env_it;
	unsigned int memb_it;
	unsigned int obj_it;

public:
	bool start();
	unsigned int pi_loop_rule_blocks();

	bool pi_next_rule_block();

	char pi_lhs_charge();

	unsigned int pi_lhs_membrane();

	unsigned int pi_lhs_parent_membrane();

	char pi_rhs_charge();

	unsigned int pi_loop_lhs();

	unsigned int pi_lhs_loop_U();

	unsigned int pi_lhs_loop_V();

	bool pi_lhs_next_object(unsigned int & object, unsigned int & multiplicity);

	unsigned int pi_loop_rules();

	bool pi_next_rule();

	float* pi_rule_probabilities();

	unsigned int pi_loop_rhs();

	unsigned int pi_rhs_membrane();

	unsigned int pi_rhs_loop_U();

	unsigned int pi_rhs_loop_V();

	bool pi_rhs_next_object(unsigned int & object, unsigned int & multiplicity);

	unsigned int env_loop_rule_blocks();

	bool env_next_rule_block();

	unsigned int env_get_object_lhs();

	unsigned int env_get_environment();

	unsigned int env_loop_rules();

	bool env_next_rule();

	float env_get_probability();

	unsigned int env_loop_rhs();

	bool env_next_object(unsigned int & object, unsigned int & environment);

	bool conf_loop();

	unsigned int conf_loop_environments();

	bool conf_next_environment();

	unsigned int conf_loop_membranes();

	bool conf_next_membrane();

	char conf_charge_membrane();

	unsigned int conf_parent_membrane();

	unsigned int conf_loop_objects();

	bool conf_next_object(unsigned int & object, unsigned int & multiplicity);
	
	// In random systems, we don't generate string identificators
	char ** get_objects_ids() {return NULL;}
	
	char ** get_environments_ids() {return NULL;}
	
	char ** get_membranes_ids() {return NULL;}

};

#endif /* PDP_PSYSTEM_SOURCE_H_ */
