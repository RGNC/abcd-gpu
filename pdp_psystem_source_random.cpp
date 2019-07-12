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


#include "pdp_psystem_source_random.h"
#include <stdlib.h>
#include <iostream>
#include <time.h>

using namespace std;

#define MAX_MULTIPLICITY 5
#define PI_OBJ 0.75
/* Random generator of a PDP P System: maravillosos procedures */
PDP_Psystem_source_random::PDP_Psystem_source_random(Options options) {
	this->options=options;

	//srandom(time(NULL));
	srandom(1);

	order[0]=GENERAL_INFO;
	order[1]=BLOCKS_PI;
	order[2]=BLOCKS_ENV;
	order[3]=CONFIGURATION;

	if (options->verbose>0)
		cout << "[1] STARTING THE RANDOM GENERATION OF THE PDP P SYSTEM" << endl;
}

PDP_Psystem_source_random::~PDP_Psystem_source_random() {
	delete []lengthU;
	delete []lengthV;
	delete []active_membrane;
	delete []rules;
	delete []prob;
	delete []obj_lhs;
	delete []obj_rhs;
	delete []env_lengthU;
	delete []lengthUp;
	delete []lengthVp;
	delete []env_env;
	delete []env_obj;
}

bool PDP_Psystem_source_random::start() {
	mutree=new unsigned int [options->num_membranes-1];
	//Membrane 1's parent is the environment
	mutree[0]=0;
	//Membrane (i+1)'s parent is random from 1 to i
	for (int i=1;i<options->num_membranes-1;i++){
		mutree[i]=random()%(i) +1;
		//cout<<"membrane "<< i+1<< " has parent:" <<mutree[i]<< endl;
	}

	//
	env_env= new uint[options->num_blocks_env];
	env_obj= new uint[options->num_blocks_env];

	lengthU= new short int[options->num_rule_blocks];
	for (int i=0;i<options->num_rule_blocks;i++)
		lengthU[i]=0;

	lengthV= new short int[options->num_rule_blocks];
	for (int i=0;i<options->num_rule_blocks;i++)
		lengthV[i]=0;

	active_membrane= new unsigned int[options->num_rule_blocks];
	for (int i=0;i<options->num_rule_blocks;i++)
		active_membrane[i]=0;

	rules= new short int[options->num_rule_blocks];
	for (int i=0;i<options->num_rule_blocks;i++)
		rules[i]=0;

	prob= new float[options->max_num_rules*options->num_environments];
	for (int i=0;i<options->max_num_rules*options->num_environments;i++)
		prob[i]=0.0;

	obj_lhs= new unsigned int[options->max_lhs];
	for (int i=0;i<options->max_lhs;i++)
		obj_lhs[i]=0;

	obj_rhs= new unsigned int[options->max_rhs];
		for (int i=0;i<options->max_rhs;i++)
			obj_rhs[i]=0;

	lengthUp= new short int[options->num_rule_blocks*options->max_num_rules];
	for (int i=0;i<options->max_num_rules*options->num_rule_blocks;i++)
		lengthUp[i]=0;

	lengthVp= new short int[options->num_rule_blocks*options->max_num_rules];
	for (int i=0;i<options->max_num_rules*options->num_rule_blocks;i++)
		lengthVp[i]=0;

	env_lengthU= new short int[options->num_blocks_env*options->max_num_rules];
	for (int i=0;i<options->num_blocks_env*options->max_num_rules;i++)
		env_lengthU[i]=0;
	return true;
}

/*********************/
/* Procedures for Pi */

unsigned int PDP_Psystem_source_random::pi_loop_rule_blocks(){
	rule_block_it=0;
	return options->num_rule_blocks;
}

bool PDP_Psystem_source_random::pi_next_rule_block() {
	if (rule_block_it < options->num_rule_blocks-1) {
		rule_block_it++;
		return true;
	}
	return false;
}

unsigned int PDP_Psystem_source_random::pi_loop_lhs() {
	if (lengthV[rule_block_it]+lengthU[rule_block_it]==0) {
		lengthV[rule_block_it]=random() % (options->max_lhs-1) + 1;
		lengthU[rule_block_it]=random() % (options->max_lhs-lengthV[rule_block_it]);
		active_membrane[rule_block_it]=random() % (options->num_membranes-1) +1;
	}
	return lengthV[rule_block_it]+lengthU[rule_block_it];
}

char PDP_Psystem_source_random::pi_lhs_charge() {
	return 0;//random()%3;
}

unsigned int PDP_Psystem_source_random::pi_lhs_membrane() {

	return active_membrane[rule_block_it];
}

unsigned int PDP_Psystem_source_random::pi_lhs_parent_membrane() {
	return mutree[active_membrane[rule_block_it]-1];
}

unsigned int PDP_Psystem_source_random::pi_lhs_loop_U() {
	U_it=0; V_it=lengthV[rule_block_it]+2;
	return lengthU[rule_block_it];
}

unsigned int PDP_Psystem_source_random::pi_lhs_loop_V() {
	V_it=0; U_it=lengthU[rule_block_it]+2;
	return lengthV[rule_block_it];
}

bool PDP_Psystem_source_random::pi_lhs_next_object(unsigned int & object, unsigned int & multiplicity) {
	//20% objects reserved for environment
	//LHS can be any object in v, but only the 80% in u
	bool pi_segment=U_it>lengthU[rule_block_it];

	int obj_segment=pi_segment?
			options->num_objects:
			(options->num_objects*PI_OBJ);
	int obj_offset=	0;


	int oidx=0;
	if (U_it<lengthU[rule_block_it]) {
		oidx=U_it++;
	}
	else if (V_it<lengthV[rule_block_it]) {
		oidx=lengthU[rule_block_it]+V_it++;
	}
	else {
		return false;
	}
	unsigned int obj=(random() % obj_segment) + obj_offset;
	bool rep=false;

	do {
		rep=false;
		for (int aux=0;aux<oidx;aux++) {
			if (obj_lhs[aux]==obj) {
				rep=true;
				obj=(random() % obj_segment) + obj_offset;
				break;
			}
		}
	} while (rep);

	obj_lhs[oidx]=obj;

	object = obj;//random() % options->num_objects;
	multiplicity = random() % MAX_MULTIPLICITY +1;

	return true;
}

unsigned int PDP_Psystem_source_random::pi_loop_rules() {
	rule_it=0;
	if (rules[rule_block_it]==0) {
		int nrules=random() % options->max_num_rules + 1;
		rules[rule_block_it]=nrules;
	}
	for (unsigned int env=0; env<options->num_environments; env++) {
		float baux=100.0;
		for (int i=0; i<rules[rule_block_it]; i++) {
			float pr = (i==(rules[rule_block_it]-1))?
					baux/100.0 :
					(random()%((int)100/rules[rule_block_it])+1)/100.0;
			baux-= pr*100;
			prob[i*options->num_environments+env]=pr;//rule_block_it*100 + env*10 + i;//pr;
		}
	}

	return rules[rule_block_it];
}

bool PDP_Psystem_source_random::pi_next_rule() {
	if (rule_it<rules[rule_block_it]-1) {
		rule_it++;
		return true;
	}
	return false;
}

float* PDP_Psystem_source_random::pi_rule_probabilities() {
	return &(prob[rule_it*options->num_environments]);
}

unsigned int PDP_Psystem_source_random::pi_loop_rhs() {
	if (lengthVp[rule_block_it*options->max_num_rules+rule_it]+lengthUp[rule_block_it*options->max_num_rules+rule_it]==0) {
		lengthVp[rule_block_it*options->max_num_rules+rule_it]=random() % (options->max_rhs-1) + 1;
		lengthUp[rule_block_it*options->max_num_rules+rule_it]=random() % (options->max_rhs-lengthVp[rule_block_it*options->max_num_rules+rule_it]);
	}

	return lengthVp[rule_block_it*options->max_num_rules+rule_it]+lengthUp[rule_block_it*options->max_num_rules+rule_it];
}

char PDP_Psystem_source_random::pi_rhs_charge() {
	return 0;//random()%3;
}

unsigned int PDP_Psystem_source_random::pi_rhs_membrane() {
	return active_membrane[rule_block_it];
}

unsigned int PDP_Psystem_source_random::pi_rhs_loop_U() {
	Up_it=0; Vp_it=lengthVp[rule_block_it*options->max_num_rules+rule_it]+2;
	return lengthUp[rule_block_it*options->max_num_rules+rule_it];
}

unsigned int PDP_Psystem_source_random::pi_rhs_loop_V() {
	Vp_it=0; Up_it=lengthUp[rule_block_it*options->max_num_rules+rule_it]+2;
	return lengthVp[rule_block_it*options->max_num_rules+rule_it];
}

bool PDP_Psystem_source_random::pi_rhs_next_object(unsigned int & object, unsigned int & multiplicity) {
	//RHS can be any object
	int obj_segment=options->num_objects;
	int obj_offset=0;

	int oidx=0;
	if (Up_it<lengthUp[rule_block_it*options->max_num_rules+rule_it]) {
		oidx=Up_it++;
	}
	else if (Vp_it<lengthVp[rule_block_it*options->max_num_rules+rule_it]) {
		oidx=lengthUp[rule_block_it*options->max_num_rules+rule_it]+Vp_it++;
	}
	else {
		return false;
	}

	unsigned int obj=(random() % obj_segment) + obj_offset;
	bool rep=false;

	do {
		rep=false;
		for (int aux=0;aux<oidx;aux++) {
			if (obj_rhs[aux]==obj) {
				rep=true;
				obj=(random() % obj_segment) + obj_offset;
				break;
			}
		}
	} while (rep);

	obj_rhs[oidx]=obj;

	object = obj;//random() % options->num_objects;
	multiplicity = random() % MAX_MULTIPLICITY +1;

	return true;
}


/****************************************************/
/* Corresponding to rule blocks of the environments */

unsigned int PDP_Psystem_source_random::env_loop_rule_blocks() {
	block_env_it=0;

	return options->num_blocks_env;
}

bool PDP_Psystem_source_random::env_next_rule_block() {
	if (block_env_it < options->num_blocks_env-1) {
		block_env_it++;
		return true;
	}
	return false;
}

unsigned int PDP_Psystem_source_random::env_get_object_lhs() {
	return env_obj[block_env_it];
}

unsigned int PDP_Psystem_source_random::env_get_environment() {
	//LHS env rules can only be 25%
	int obj_segment=(options->num_objects-(options->num_objects*PI_OBJ));
	int obj_offset=(options->num_objects*PI_OBJ);

	uint next_object=(random() % obj_segment)+obj_offset;
	uint next_env=random() % options->num_environments;
	bool rep=false;
	do{
		rep=false;
		for (int i=0;i<block_env_it;i++){
			if(env_obj[i]==next_object && env_env[i]==next_env) {
					rep=true;
					next_object=(random() % obj_segment)+obj_offset;
					next_env=random() % options->num_environments;
			}
		}
	}while(rep);

	env_obj[block_env_it]=next_object;
	env_env[block_env_it]=next_env;
	return next_env;
}

unsigned int PDP_Psystem_source_random::env_loop_rules() {
	rule_env_it=0;

	if (rules[block_env_it]==0) {
		int nrules=random() % options->max_num_rules + 1;
		rules[block_env_it]=nrules;
	}

	float baux=100.0;
	for (int i=0; i<rules[block_env_it]; i++) {
		float pr = (i==(rules[block_env_it]-1))?
				baux/100.0 :
				(random()%((int)100/rules[block_env_it])+1)/100.0;
		baux-= pr*100;
		prob[i]=pr;
	}

	return rules[block_env_it];
}

bool PDP_Psystem_source_random::env_next_rule() {
	if (rule_env_it < rules[block_env_it] -1) {
		rule_env_it++;
		return true;
	}
	return false;
}

float PDP_Psystem_source_random::env_get_probability() {
	return prob[rule_env_it];
}

unsigned int PDP_Psystem_source_random::env_loop_rhs() {
	Up_it=0;

	if (env_lengthU[rule_env_it]==0) {
		env_lengthU[rule_env_it]=random() % (options->max_rhs-1) +1;
	}

	return env_lengthU[rule_env_it];
}

bool PDP_Psystem_source_random::env_next_object(unsigned int & object, unsigned int & environment) {

	if (Up_it<env_lengthU[rule_env_it]) {
		//RHS for env rules can be any object
		int obj_segment=options->num_objects;
		int obj_offset=0;
		Up_it++;
		object=(random() % obj_segment)+obj_offset;
		environment=random() % options->num_environments;
		return true;
	}
	return false;
}

/*************************************/
/* Corresponding to the configuration*/

bool PDP_Psystem_source_random::conf_loop() {
	return true;
}

unsigned int PDP_Psystem_source_random::conf_loop_environments() {
	env_it=0;
	return options->num_environments;
}

bool PDP_Psystem_source_random::conf_next_environment() {
	if (env_it<options->num_environments-1) {
		env_it++;
		return true;
	}
	return false;
}

unsigned int PDP_Psystem_source_random::conf_loop_membranes() {
	memb_it=0;
	return options->num_membranes;
}

bool PDP_Psystem_source_random::conf_next_membrane() {
	if (memb_it<options->num_membranes-1) {
		memb_it++;
		return true;
	}
	return false;
}

unsigned int PDP_Psystem_source_random::conf_parent_membrane() {
	return 0;
}

char PDP_Psystem_source_random::conf_charge_membrane() {
	return 0;
}

unsigned int PDP_Psystem_source_random::conf_loop_objects() {
	obj_it=0;
	return options->num_objects;
}

bool PDP_Psystem_source_random::conf_next_object(unsigned int & object, unsigned int & multiplicity) {
	if (obj_it<options->num_objects) {
		object=obj_it;
		//If random number is greater than the value specified,
		//the object will not be in the initial configuration
		//E.g.: objects_init_config=0.8f, so 20% of objects won't be in the init config
		multiplicity=random()<(options->objects_init_config*RAND_MAX)?random()%100:0;
		obj_it++;
		return true;
	}
	return false;
}
