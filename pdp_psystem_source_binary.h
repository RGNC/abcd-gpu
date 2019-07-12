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


#ifndef PDP_PSYSTEM_SOURCE_BIN_H_
#define PDP_PSYSTEM_SOURCE_BIN_H_

#include "binbit.h"
#include "pdp_psystem_source.h"

/**********************/
/* Classes for source */
class PDP_Psystem_source_binary: public PDP_Psystem_source {
    
public:
	PDP_Psystem_source_binary(const char* filename, Options options);
	PDP_Psystem_source_binary(Options options);
	~PDP_Psystem_source_binary();
	void write_test_binary();
	
private:
	/* File */
	ifstream is;
	
	/* Sizes */
	unsigned int number_objects;
	unsigned int number_environments;
	unsigned int number_membranes;
	unsigned int number_rule_blocks_pi;
	unsigned int number_rule_blocks_env;	
	unsigned int number_rules_pi;
	unsigned int number_rules_env;
	unsigned int number_rules;
	
	/* Auxiliary structures */
	struct precision {
		short int objects;
		short int environments;
		short int membranes;
		short int pi_rules;
		short int env_rules;
		short int multisets;
		short int ini_num_objects;
		short int ini_mult;
	} precision;
	
	struct id_strings {
		bool objects;
		bool environments;
		bool membranes;
	} id_strings;
	
	bool delete_id_objects,delete_id_environments,delete_id_membranes;
	char** id_objects;
	char** id_environments;
	char** id_membranes;
	
	unsigned int* membrane_hierarchy;
	
	short int *lengthU;
	short int *lengthV;
	unsigned int *active_membrane;
	unsigned int *rules;
	short int *lengthUp;
	short int *lengthVp;
	unsigned int* obj_lhs, *obj_lhs_m,* obj_rhs, *obj_rhs_m;
	unsigned int num_obj_lhs_blocks_pi;
	unsigned int num_obj_rhs_rules_pi;
	unsigned int prob_length;
	float * prob;
	char* block_precision, * charge;
	
//	unsigned int * env_rules;
	short int *env_lengthU;
	
	unsigned int ** ini_multiset;
	unsigned int * ini_info;
	char * ini_charge;

	/* Iterators */
	unsigned int rule_block_it;
	unsigned int lhs_it;
	unsigned int U_it;
	unsigned int V_it;
	unsigned int Up_it;
	unsigned int Vp_it;
	unsigned int rule_it;
	unsigned int rhs_it,rhs_it_internal;

	unsigned int block_env_it;
	unsigned int rule_env_it;

	unsigned int env_it;
	unsigned int memb_it;
	unsigned int obj_it;
	
	/* Auxiliary methods to read binary data */
	/* In order to split the procedure used in start() */
	void check_file_exception(bool condition, string message);
	bool read_header();
	bool read_global_sizes();
	bool read_inf_1();
	bool read_inf_2();
	bool read_inf_3();
	bool read_multisets();
	
	/* Class for errors and exceptions concerning the file */
	class FileException {
	    private:
	    string message;
	    
	    public:
	    FileException (string message) {
		this->message=message;
	    }
	    
	    string getMessage() {
		return message;
	    }
	};
	

public: /* Public methods inherited from pdp_psystem_source */
    
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
	
	char ** get_objects_ids();
	
	char ** get_environments_ids();
	
	char ** get_membranes_ids();

};

#endif /* PDP_PSYSTEM_SOURCE_BIN_H_ */
