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


#ifndef PDP_PSYSTEM_SOURCE_H_
#define PDP_PSYSTEM_SOURCE_H_

/*
 * Options structure
 */

struct _options{
	int num_rule_blocks;
	int num_blocks_env;
	int num_simulations;
	int num_environments;
	int num_objects;
	int num_membranes;

	int max_num_rules;
	int max_lhs;
	int max_rhs;

	int num_parallel_simulations;
	int num_parallel_environments;

	int time;
	int cycles;
	int accuracy;

	int verbose;
	unsigned long int mem;
	int debug;
};

typedef struct _options * Options;


/*******************************/
/* Class representing a source */
/*******************************/

class PDP_Psystem_source {
public:
	static const int num_parts=4;

	enum info_part {GENERAL_INFO=0, BLOCKS_PI=1, BLOCKS_ENV=2, CONFIGURATION=3};

	enum info_part order[4];

	virtual bool start()=0; // Return: false if problem in the input

	/**************************************/
	/* Corresponding to rule blocks of PI */
	virtual unsigned int pi_loop_rule_blocks()=0; // Return: number of rule blocks of Pi

	virtual bool pi_next_rule_block()=0;  // Return: false if no more rules

	virtual char pi_lhs_charge()=0; // Return: alpha

	virtual unsigned int pi_lhs_membrane()=0; // Return: active membrane

	virtual unsigned int pi_lhs_parent_membrane()=0; // Return: parent of the active membrane (deprecated)

	virtual char pi_rhs_charge()=0; // Return: beta

	/* Corresponding to LHS */
	virtual unsigned int pi_loop_lhs()=0; // Return: number of total objects in the LHS

	virtual unsigned int pi_lhs_loop_U()=0; // Return: objects in the parent of the active membrane

	virtual unsigned int pi_lhs_loop_V()=0; // Return: objects in the active membrane

	virtual bool pi_lhs_next_object(unsigned int & object, unsigned int & multiplicity)=0; // Return: false if last object
	/* end about LHS */

	/* Corresponding to rules */
	virtual unsigned int pi_loop_rules()=0; // Return: number of rules in the rule block, Pi or Env state

	virtual bool pi_next_rule()=0; // Return: false if no more rules

	virtual float* pi_rule_probabilities()=0;

	/* Corresponding to RHS */
	virtual unsigned int pi_loop_rhs()=0; // Return: total number of objects in the RHS

	virtual unsigned int pi_rhs_membrane()=0; // Return: active membrane

	virtual unsigned int pi_rhs_loop_U()=0; // Return: objects in the parent of the active membrane

	virtual unsigned int pi_rhs_loop_V()=0; // Return: objects in the active membrane

	virtual bool pi_rhs_next_object(unsigned int & object, unsigned int & multiplicity)=0; // Return: false if last object
	/* end about RHS*/

	/* end about rules */

	/* end about rule blocks of Pi*/
	/******************************/

	/****************************************************/
	/* Corresponding to rule blocks of the environments */
	virtual unsigned int env_loop_rule_blocks()=0; // Return: number of rule blocks for communication rules

	virtual bool env_next_rule_block()=0; // Return: false if no more rule blocks

	virtual unsigned int env_get_object_lhs()=0; // Return: object corresponding in the LHs

	virtual unsigned int env_get_environment()=0; // Return: environment where the rule block is applicable

	/* Corresponding to rules in the rule block for environments */
	virtual unsigned int env_loop_rules()=0;  // Return: number of rules in the block

	virtual bool env_next_rule()=0; // Return: false if no more rules

	virtual float env_get_probability()=0; // Return: probability associated to the rule

	/* Corresponding to the RHS of a rule */
	virtual unsigned int env_loop_rhs()=0; // Return: total number of objects in the RHS

	virtual bool env_next_object(unsigned int & object, unsigned int & environment)=0; // Return: false if no more objects
	/* end about RHS */

	/* end about rules for environments */

	/* Use functions corresponding to Pi: next_rule_block, loop_rules, next_rule, loop_rhs */

	/* end about rule blocks of environments */
	/*****************************************/

	/*************************************/
	/* Corresponding to the configuration*/
	virtual bool conf_loop()=0;

	virtual unsigned int conf_loop_environments()=0;

	virtual bool conf_next_environment()=0;

	virtual unsigned int conf_loop_membranes()=0;

	virtual bool conf_next_membrane()=0;

	virtual char conf_charge_membrane()=0;

	virtual unsigned int conf_parent_membrane()=0;

	virtual unsigned int conf_loop_objects()=0;

	virtual bool conf_next_object(unsigned int & object, unsigned int & multiplicity)=0;
	/* end about configuration */
	/***************************/

	/*******************************************/
	/* Corresponding to simulation information */
	Options options;
	/* end about simulation information */
	/************************************/

	/**********************/
	/* Debugging purposes */
	virtual char ** get_objects_ids()=0;
	virtual char ** get_environments_ids()=0;
	virtual char ** get_membranes_ids()=0;
	/* End about debugging */
};

#endif /* PDP_PSYSTEM_SOURCE_H_ */
