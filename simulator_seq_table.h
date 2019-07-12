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

#ifndef __SEQ_SIM__TABLE__
#define __SEQ_SIM__TABLE__

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "pdp_psystem_sab.h"
#include "simulator.h"

/* Warning: This simulator only operates with rules of Pi */
class Simulator_seq_table: public Simulator {

	friend class PDP_Psystem_SAB;

public:

	Simulator_seq_table(PDP_Psystem_SAB* PDPps) {
		this->structures=PDPps->structures;
		this->options=PDPps->options;
	}

	bool run();
	bool step(int k);
	
	/* Methods for querying data */
	unsigned int get_object_value(  int obj, 
					int membr, 
					int env,
					int sim) {
		return structures->configuration.multiset[env*options->num_objects*options->num_membranes+membr*options->num_objects+obj];
	}

	char get_membrane_charge(  int membr,
				   int env,
				   int sim) {
		return structures->configuration.membrane[env*options->num_membranes+membr];
	}

private:

	PDP_Psystem_SAB::Structures structures;
	Options options;

	/*****************************/
	/* Auxiliary data structures */
	/*****************************/

	float * table;
	bool *active_col;
	bool *active_row;
	bool *init_active_row;
	unsigned int colsize;
	unsigned int rowsize;
	float *addition;
	unsigned int *block_min;

	/* Only debuggin purposes */
	void print_configuration();
	void print_table();
	void print_activations(unsigned int environment);
	void print_block_applications();

	/* Initialization */
	float * init_table();
	void init_activations();
	void delete_table();
	void delete_activations();

	/* Micro phases */
	void selection_phase1();
	void selection_phase2();
	void selection_phase3();

	/* Main phases */
	unsigned int selection();
	unsigned int execution();
};


/* Simulation algorithm based on a distribution table */
//bool simulation_seq_table(Structures* structures,Options* options);

/* Simulation algorithm based on a direct distribution */
//bool simulation_seq_dir(Structures* structures, Options* options);

#endif
