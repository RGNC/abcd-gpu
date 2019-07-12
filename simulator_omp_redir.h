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

#ifndef __OMP_SIM__REDIR__
#define __OMP_SIM__REDIR__

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <omp.h>
#include "pdp_psystem_redix.h"
#include "pdp_psystem_output.h"
#include "simulator.h"

/* ABV VALUES */
#define ABV_T uint
#define ABV_WORD_SIZE 32
#define ABV_LOG_WORD_SIZE 5
#define ABV_DESPL_MASK 0x1F
#define ABV_INIT_WORD 0xFFFFFFFF

class Simulator_omp_redir: public Simulator {

public:

	Simulator_omp_redir(PDP_Psystem_REDIX* PDPps, int simulations, bool accurate=true, PDP_Psystem_output * PDPout=NULL);
	~Simulator_omp_redir() { 
		del(); 
	}
	
	bool run();
	bool step(int k);
	
	/* Methods for querying data */
	unsigned int get_object_value(  int obj, 
					int membr, 
					int env,
					int sim) {
		return structures->configuration.multiset[sim*options->num_environments*esize+env*esize+membr*msize+obj];
	}

	char get_membrane_charge(  int membr,
				   int env,
				   int sim) {
		return structures->configuration.membrane[sim*(options->num_environments*options->num_membranes)+env*options->num_membranes+membr];
	}

private:

	PDP_Psystem_REDIX::Structures structures;
	Options options;

	/*****************************/
	/* Auxiliary data structures */
	/*****************************/

	float *addition;
	unsigned int addition_size;
	uint * ini_denominator, * ini_numerator;
	uint * denominator, * numerator;
	bool accurate;
	
	PDP_Psystem_REDIX::Configuration ini_cfg;
	unsigned int msize,esize,bpsize,besize,rpsize,resize;

	/*******************************/
	PDP_Psystem_output * PDPout;
	bool will_out;

	/* Only debugging purposes */
	PDP_Psystem_redix_out_std* pdp_out;
	// TODO: delete after their replacement 
	void print_configuration(PDP_Psystem_REDIX::Configuration conf, int sim, const char* message=NULL, int step=-1);
	void print_block_applications(uint * nb, int stid, const char* message=NULL);

	/* Initialization */
	bool init();
	void del();
	//void reset(unsigned int sim_thread_id=0);

	/* Parallel runs */
	bool run_parallel_sim(int time);

	/* Micro phases */
	bool selection_phase1(int simulation, int stid);
	bool selection_phase2(int simulation, int stid);
	bool selection_phase3(int simulation, int stid);

	/* Main phases */
	int selection(int simulation, int stid);
	int execution(int simulation, int stid);
	
	
	/***************************************************************/
	/* The following defines how to use activation bit vectors ABV */
	/***************************************************************/
	unsigned int asize,abv_size;
	ABV_T * abv, *d_abv;


	void initialize_abv () {
		abv_size=asize*options->num_environments*options->num_parallel_simulations;
		abv = new unsigned int [abv_size];
	}

	bool is_active (uint block, uint env, uint sim) {
		return (abv[sim*options->num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
		               >> ((~block)&ABV_DESPL_MASK))
		        & 0x1;
	}

	void set_activation (bool value, uint block, uint env, uint sim) {
		abv[sim*options->num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)] =
				(abv[sim*options->num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)]
				      & ~(0x1<<((~block)&ABV_DESPL_MASK)))
				|(value << ((~block)&ABV_DESPL_MASK));
	}

	void deactivate(uint block, uint env, uint sim) {
		abv[sim*options->num_environments*asize+env*asize+(block>>ABV_LOG_WORD_SIZE)] &=
				~(0x1<<((~block)&ABV_DESPL_MASK));
	}

	void reset_activations (int stid) {
		for (unsigned int i=stid*asize*options->num_environments; i<(stid+1)*asize*options->num_environments; i++)
			abv[i]=ABV_INIT_WORD;
	}
	
	/**********************************************************/
	/* The following is for the mutually consistency checking */
	/**********************************************************/
	
	unsigned int * m_c_conflicts;
	char * m_c_charges;
	
	/* Safely add and mult uint numbers, returns true if overflow */
	inline bool safe_u_add(uint& op1, uint op2){
		uint c_test=op1+op2;
		if (c_test-op2 != op1) 
			return true;
		else op1=c_test;

		return false;
	}

	inline bool safe_u_mul(uint& op1, uint op2) {
		uint c_test=op1*op2;
		if (c_test/op2 != op1) 
			return true;
		else op1=c_test;	

		return false;
	}
};

#endif
