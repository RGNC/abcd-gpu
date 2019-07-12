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

#include "simulator_omp_redir.h"

#include <math.h>
#include <limits.h>
#include <iostream>
#include <timestat.h>
#include <stdlib.h>

/************************************************************/
/* The following sets how the arrays are indexed on the CPU */
/************************************************************/
#define BM_IDX (stid*options->num_environments*besize+env*besize+block)
//#define AD_IDX(o,m) (stid*options->num_environments*esize+env*esize+(m)*msize+(o))
#define AD_IDX(o,m) (stid*esize+(m)*msize+(o))
#define NB_IDX (stid*options->num_environments*besize+env*besize+block)
#define NRB_IDX (stid*(options->num_environments*rpsize+(resize-rpsize))+env*besize+block)
#define NR_P_IDX (stid*(options->num_environments*rpsize+(resize-rpsize))+env*rpsize+r)
#define NR_E_IDX (stid*(options->num_environments*rpsize+(resize-rpsize))+options->num_environments*rpsize+(r-rpsize))
#define CH_IDX(m) (sim*(options->num_environments*options->num_membranes)+env*options->num_membranes+(m))
#define MU_IDX(o,m) (sim*options->num_environments*esize+env*esize+(m)*msize+(o))
#define MC_IDX(m) (stid*options->num_membranes+(m))

using namespace std;

Simulator_omp_redir::Simulator_omp_redir(PDP_Psystem_REDIX* PDPps, int simulations, bool accurate, PDP_Psystem_output* PDPout) {
		this->structures=PDPps->structures;
		this->options=PDPps->options;
		this->options->num_simulations = simulations;
		this->accurate=accurate;
		// The real output (binary, csv ...)
		this->PDPout=PDPout;
		/* Configure the standard output class (debugging purposes only) */
		pdp_out = new PDP_Psystem_redix_out_std(PDPps);
		
		init();
}

bool Simulator_omp_redir::init() {

	/* Set auxiliary sizes info */
	esize=options->num_objects*options->num_membranes;
	msize=options->num_objects;
	bpsize=options->num_rule_blocks;
	besize=options->num_blocks_env+options->num_rule_blocks;
	rpsize=structures->pi_rule_size;
	resize=structures->pi_rule_size+structures->env_rule_size;
	asize=((besize>>ABV_LOG_WORD_SIZE)+1);
	
	char * def_dev = getenv("OMP_NUM_THREADS");	
	if (def_dev==NULL)
		omp_set_num_threads(1);

	// TODO: ONLY USE MAX_PARALLEL_SIM FOR NB AND NR (SAVES MEMORY!!)
	options->num_parallel_simulations = omp_get_max_threads();
	
	if (options->num_simulations<options->num_parallel_simulations) {
		options->num_parallel_simulations=options->num_simulations;
		omp_set_num_threads(options->num_simulations);
	}
	
	if (options->verbose>0)
	cout << "=> Executing " << options->num_parallel_simulations << " threads" << endl;
	
	/* Calculating the amount of memory, and simulations to handle */
	//unsigned int dep_mem;
	
	// Temporally like this until auto-compression process
	/*dep_mem=options->num_membranes*options->num_environments*sizeof(CHARGE); //membrane
	dep_mem+=options->num_objects*options->num_membranes*options->num_environments*sizeof(MULTIPLICITY); //multiset
	dep_mem+=(options->num_rule_blocks+options->num_blocks_env)*options->num_environments*sizeof(MULTIPLICITY);//nb
	dep_mem+=((structures->pi_rule_size)*options->num_environments+structures->env_rule_size)*sizeof(MULTIPLICITY);//nr
	if (!accurate)
		dep_mem+=options->num_objects*options->num_membranes*options->num_environments*sizeof(float); //addition
	else 
		dep_mem+=options->num_objects*options->num_membranes*options->num_environments*sizeof(uint); //numerator
	dep_mem+=asize*options->num_environments*sizeof(ABV_T); // ABV activations*/
	
	/* Create initial configuration for all simulations */
	ini_cfg = structures->configuration;

	structures->configuration.membrane_size=options->num_membranes*options->num_environments*options->num_simulations;
	structures->configuration.membrane = new CHARGE[structures->configuration.membrane_size];

	structures->configuration.multiset_size = options->num_objects*options->num_membranes*options->num_environments*options->num_simulations;
	structures->configuration.multiset = new MULTIPLICITY[structures->configuration.multiset_size];
		
	for (int sim=0; sim<options->num_simulations; sim++) {
		for (int env=0; env<options->num_environments; env++) {
			for (int m=0;m<options->num_membranes; m++) {
				structures->configuration.membrane[CH_IDX(m)]=ini_cfg.membrane[env*options->num_membranes+m];
			}
			for (int o=0;o<options->num_membranes*options->num_objects;o++) {
				structures->configuration.multiset[MU_IDX(o,0)]=ini_cfg.multiset[env*esize+o];
			}
		}
	}	
	
	/* Initialize Auxiliary Data Structures */
	/* Initialize Nb */
	structures->nb_size= (options->num_rule_blocks+options->num_blocks_env)*options->num_environments*options->num_parallel_simulations;
	structures->nb = new MULTIPLICITY [structures->nb_size];
	//for (unsigned int i=0; i<structures->nb_size;i++) structures->nb[i]=999999;

	/* Initialize Nr */
	structures->nr_size= ((structures->pi_rule_size)*options->num_environments+structures->env_rule_size)*options->num_parallel_simulations;
	structures->nr = new MULTIPLICITY [structures->nr_size];

	/* Create and initialize auxiliary structures */
	//addition_size=options->num_objects*options->num_membranes*options->num_environments*options->num_parallel_simulations;
/*	addition_size=options->num_objects*options->num_membranes*options->num_parallel_simulations;
	addition = new float[addition_size];
	for (unsigned int i=0; i<addition_size; i++)
		addition[i]=0.0;*/


	/* Initialize auxiliary structures for normalization */
	addition_size=options->num_objects*options->num_membranes*options->num_parallel_simulations;
	bool finished = false;
	
	while (!finished) {
		/* Use n/d notation for row additions */
		if (accurate) {
			/* Check overflows */
			finished=true;
			bool overflow=false;
			
			ini_denominator = new uint[esize];
			ini_numerator = new uint[esize];
			denominator = ini_denominator;
			numerator = new uint [addition_size];
			addition = NULL;
			
			for (int i=0;i<esize;i++) {
				ini_denominator[i]=1;
				ini_numerator[i]=0;
			}
			
			for (uint block=0; block<besize; block++) {
				for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
					unsigned int obj=structures->lhs.object[o];
					unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
					unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);
					
					uint a=ini_denominator[membr*options->num_objects+obj];
					uint b=mult;
					
					int multiple1 = a % b;
					int multiple2=1;
					if (multiple1!=0)
						multiple2 = b % a;
					
					/* If a is multiple of b */
					if (multiple1==0) {
						overflow=safe_u_add(ini_numerator[membr*options->num_objects+obj],a/b);
					}
					/* If b is multiple of a */
					else if (multiple2==0) {
						overflow=safe_u_mul(ini_numerator[membr*options->num_objects+obj],b/a);
						overflow=overflow||safe_u_add(ini_numerator[membr*options->num_objects+obj],1);
						ini_denominator[membr*options->num_objects+obj]=b;
					}
					/* If they are no multiple */
					else {
						overflow=safe_u_mul(ini_numerator[membr*options->num_objects+obj],b);
						overflow=overflow||safe_u_add(ini_numerator[membr*options->num_objects+obj],a);
						overflow=overflow||safe_u_mul(ini_denominator[membr*options->num_objects+obj],b);
					}
					if (overflow) break;
				}
				if (overflow) {
					if (options->verbose>0) {
						cout << "Warning: overflow detected in initialization of row sums (accurate mode n/d), switching to float" << endl;
					}
					accurate=false;
					finished=false;
					delete [] ini_denominator;
					delete [] ini_numerator;
					delete [] numerator;
					ini_numerator=numerator=ini_denominator=denominator=NULL;
					break;
				}
			}
		}
		/* Use float notation for row additions */
		else {
			addition = new float[addition_size];
			finished=true;
			ini_numerator=numerator=ini_denominator=denominator=NULL;
		}
	}	
	
	/* Initialize the new data structure for activation bit vector */
	initialize_abv();
	
	/* Initialize data structures for mutually consistency checking*/
	int mc_size=options->num_membranes*options->num_parallel_simulations;
	m_c_conflicts = new unsigned int[mc_size];
	m_c_charges = new char [mc_size];
	
	unsigned int mem=((accurate)?(addition_size*sizeof(uint)+2*esize*sizeof(uint)):(addition_size*sizeof(float)))+
			structures->nb_size*sizeof(unsigned int)+mc_size*(sizeof(int)+sizeof(char))+
			structures->nr_size*sizeof(unsigned int)+structures->configuration.membrane_size*sizeof(char)+
			structures->configuration.multiset_size*sizeof(unsigned int);
	options->mem+=mem;

	/* Print new information */
	if (options->verbose>1) {
		cout << "=> Memory used by the virtual table algorithm:" << endl;

		cout << "Addition: " << addition_size*sizeof(float) << " (" << addition_size*sizeof(float)/1024 << "KB)" << endl;
		cout << "Nb: " << structures->nb_size*sizeof(unsigned int) << " (" << structures->nb_size*sizeof(unsigned int)/1024 << "KB)" << endl;
		cout << "Nr: " << structures->nr_size*sizeof(unsigned int) << " (" << structures->nr_size*sizeof(unsigned int)/1024 << "KB)" << endl;
		cout << "Membrane charges: " << structures->configuration.membrane_size*sizeof(char) << " (" << structures->configuration.membrane_size*sizeof(char)/1024 << "KB)" << endl;
		cout << "Multisets: " << structures->configuration.multiset_size*sizeof(unsigned int) << " (" << structures->configuration.multiset_size*sizeof(unsigned int)/1024 << "KB)" << endl;

		int count=0;
		float div=1;
		char unit[6]={' ','K','M','G','T','P'};
		while ((mem/div)>1023 && count<3){
			div*=1024;
			count++;
		}
		cout << "TOTAL: " << mem << " (" << mem/div << " " << unit[count] << "B)" << endl << endl;
	}

	return true;
}

void Simulator_omp_redir::del() {
	if (!accurate)
		delete [] addition;
	else {
		delete [] numerator;
		delete [] ini_numerator;
		delete [] denominator;
	}
	PDP_Psystem_REDIX::Configuration aux;
	aux=structures->configuration;
	structures->configuration=ini_cfg;
	delete []aux.membrane;
	delete []aux.multiset;
	delete []structures->nb;
	delete []structures->nr;
	delete []abv;
	delete[] m_c_conflicts;
	delete[] m_c_charges;
}

bool Simulator_omp_redir::step(int k) {
    /* Selected method from:
	 M.A. Martínez-del-Amor, I. Karlin, R.E. Jensen, M.J. Pérez-Jiménez, A.C. Elster
	 Parallel Simulation of Probabilistic P systems on Multicore Platforms,
	 In Proc. of the 10th BWMC, pp. 17-26 */
	return run_parallel_sim(k);
}

bool Simulator_omp_redir::run () {
	return step(options->time);
}

bool Simulator_omp_redir::run_parallel_sim (int K){

	pdp_out->print_simulator_process("[2] STARTING THE SIMULATION: Using reduced virtual table algorithm, with OpenMP and parallel simulations");
	
	/* Repeat for each simulation */
	#pragma omp parallel for schedule (dynamic,1)
	for (int simu=0; simu<options->num_simulations; simu++) {
		int stid=omp_get_thread_num();
		
		#pragma omp critical (PRINT)
		pdp_out->print_sim_thread(simu,stid);


		/* MAIN LOOP */
		for (int i=0; i<K; i++) {
		    
			#pragma omp critical (PRINT)
			pdp_out->print_step(i);
			
			/* Selection of rules */
			selection(simu,stid);

			/* Execution of rules */
			execution(simu,stid);
			
			#pragma omp critical (PRINT)
			pdp_out->print_configuration(simu);

			#pragma omp critical (OUTPUT)
			if ((i+1)%options->cycles==0)
				PDPout->write_configuration(structures->configuration.multiset,structures->configuration.membrane,simu,i+1,structures->stringids.id_objects);
		}
	}

	return true;
}

int Simulator_omp_redir::selection(int sim, int stid){

	/* PHASE 1: DISTRIBUTION */
	if (!selection_phase1(sim,stid))
		return 0;
	
	/* PHASE 2: MAXIMALITY */
	if (!selection_phase2(sim,stid))
		return 0;
	
	/* PHASE 3: PROBABILITY */
	if (!selection_phase3(sim,stid))
		return 0;
	
	return 1;
}

bool Simulator_omp_redir::selection_phase1(int sim, int stid) {
	#pragma omp critical (PRINT)
	pdp_out->print_dcba_phase(1);
	
	/* Filters */
	reset_activations(stid);
	
	bool c_conflict=false;
	//char * m_c_charges = new char [options->num_membranes];

	
	for (int env=0; env < options->num_environments; env++) {
		c_conflict = false;
		for (int m=0;m<options->num_membranes;m++) {
			m_c_charges[MC_IDX(m)]=4;
			m_c_conflicts[MC_IDX(m)]=besize+1;
		}
		
		for (unsigned int block=0; block<besize; block++) {
			/*** Filter 1 ***/
			uint membr=structures->ruleblock.membrane[block];
			bool active=false;

			// Case for rule blocks in Pi
			if (IS_MEMBRANE(membr)) {
				uint am=GET_MEMBRANE(membr);
				char ch=GET_ALPHA(membr);
				// only active those with charge alpha in LHS
				active=(structures->configuration.membrane[CH_IDX(am)] == ch);
			}
			// Case for rule blocks for communication
			else if (IS_ENVIRONMENT(membr)) {
				active=(GET_ENVIRONMENT(membr)==env);
			}

			/** Filter 2 **/
			if (active) {
				// Using new registers avoid memory accesses on the for loop
				uint o_init=structures->ruleblock.lhs_idx[block];
				uint o_end=structures->ruleblock.lhs_idx[block+1];
				for (uint o=o_init; o < o_end; o++) {
					uint obj=structures->lhs.object[o];
					membr=structures->lhs.mmultiplicity[o];
					uint mult=GET_MULTIPLICITY(membr);
					membr=GET_MEMBR(membr);

					// Check if we have enough objects to apply the block
					if (structures->configuration.multiset[MU_IDX(obj,membr)]<mult) {
						active=false;
						break;
					}
				}
			}

			if (!active)
				deactivate(block,env,stid);
			else if (active && IS_MEMBRANE(membr)) {
				uint am=GET_MEMBRANE(membr);
				char ch=GET_ALPHA(membr);
				if (m_c_charges[MC_IDX(am)]==4)
					m_c_charges[MC_IDX(am)]=ch;
				else if (m_c_charges[MC_IDX(am)]!=ch) {
					m_c_conflicts[MC_IDX(am)] = block;
					c_conflict = true;
				}
			}

			structures->nb[NB_IDX]= 0;
		}
	}
	
	if (c_conflict) {
		#pragma omp critical (PRINT)
		cout << "Error: Mutually consistence checking failed for simulation " << sim << " (stid=" << stid << ")" << endl;
		//TODO: More information from m_c_conflicts!
		return false;
	}
	/*delete [] m_c_conflicts;
	delete [] m_c_charges;*/
	
	/* Normalization & Update*/
	for (int env=0; env < options->num_environments; env++) {
		for (int a=0; a < options->accuracy; a++) {
			if (!accurate) {
				for (unsigned int i=0; i<esize; i++) {
					addition[AD_IDX(i,0)]=0.0;
				}
			} else {
				for (unsigned int i=0; i<esize; i++) {
					numerator[AD_IDX(i,0)]=ini_numerator[i];
				}
			}

			/** Normalization - step 1 **/
			for (unsigned int block=0; block<besize; block++) {
				if (!accurate) {
					// If block is active
					if (is_active(block,env,stid)) {
						for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
							unsigned int obj=structures->lhs.object[o];
							//unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
							unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);
							float inv=structures->lhs.imultiplicity[o];

							addition[AD_IDX(obj,membr)]+=inv;
						}
					}
				} else {
					if (!is_active(block,env,stid)) {
						for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
							unsigned int obj=structures->lhs.object[o];
							unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
							unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);

							uint d=denominator[membr*options->num_objects+obj];
							numerator[AD_IDX(obj,membr)]-=d/mult;
						}
					}
				}
			}

			/* Normalization - step 2 *
			 * Column minimum calculation */
			for (unsigned int block=0; block<besize; block++) {
				unsigned int minimum=0;
				// If block is activated
				if (is_active(block,env,stid)) {
					minimum=UINT_MAX;

					//cout << endl << "sim " << sim << ", env " << env << ", block " << block << ":" << endl;
					for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
						unsigned int obj=structures->lhs.object[o];
						unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
						unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);

						unsigned int value=UINT_MAX;
						if (!accurate)
							value = structures->configuration.multiset[MU_IDX(obj,membr)] / (mult*mult*addition[AD_IDX(obj,membr)]);
						else
							value = (structures->configuration.multiset[MU_IDX(obj,membr)]*denominator[membr*options->num_objects+obj]) / (mult*mult*numerator[AD_IDX(obj,membr)]);

						//cout << "obj" << obj << "*" << mult << "'" << membr << "**" << structures->configuration.multiset[MU_IDX(obj,membr)] << ", add=" << addition[AD_IDX(obj,membr)] << " => value=" << value << endl;

						minimum = (value<minimum)? value : minimum;
					}

					//cout << "MINIMUM = " << minimum << endl;
					//structures->nr[NB_IDX]=minimum;
				}
				structures->nr[NRB_IDX]=minimum;
			}		

			/* Update */		
			for (unsigned int block=0; block<besize; block++) {
				if (structures->nr[NRB_IDX]>0) {
					//cout << "B: sim=" << sim << ", env=" << env << ", block=" << block << endl;
					for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
						unsigned int obj=structures->lhs.object[o];
						unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
						unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);

						//structures->configuration.multiset[MU_IDX(obj,membr)]-=block_min[BM_IDX]*mult;
						if (structures->configuration.multiset[MU_IDX(obj,membr)] < structures->nr[NRB_IDX]*mult) {
							#pragma omp critical (PRINT)
							cout << "Error deleting LHS: stid=" << stid << ", sim=" << sim << ", env=" << env << ", block=" << block << ", membr=" << membr << ", o=" << obj << ": deleting " << structures->nr[NB_IDX]*mult << ", there is " << structures->configuration.multiset[MU_IDX(obj,membr)] << endl;
						}
						else {
							structures->configuration.multiset[MU_IDX(obj,membr)] -= structures->nr[NRB_IDX]*mult;
						}
					}
					structures->nb[NB_IDX]+=structures->nr[NRB_IDX];
				}
			}
			/** Filter 2 **/
			for (unsigned int block=0; block<besize; block++) {
				if (is_active(block,env,stid)) {
					uint o_init=structures->ruleblock.lhs_idx[block];
					uint o_end=structures->ruleblock.lhs_idx[block+1];
					for (uint o=o_init; o < o_end; o++) {
						uint obj=structures->lhs.object[o];
						uint membr=structures->lhs.mmultiplicity[o];
						uint mult=GET_MULTIPLICITY(membr);
						membr=GET_MEMBR(membr);

						// Check if we have enough objects to apply the block
						if (structures->configuration.multiset[MU_IDX(obj,membr)]<mult) {
							deactivate(block,env,stid);
							break;
						}
					}
				}
			}
		}
	}
        
        // TODO: try to avoid this critical section, since it might affect negatively performance
	#pragma omp critical (PRINT)
	pdp_out->print_block_selection(stid);
	#pragma omp critical (PRINT)
	pdp_out->print_temporal_configuration(sim);
	
	return true;
}


bool Simulator_omp_redir::selection_phase2(int sim, int stid) {
	#pragma omp critical (PRINT)
	pdp_out->print_dcba_phase(2);
	
	for (int env=0; env<options->num_environments; env++) {
		/* For each active block *
		 * In a random order */
		unsigned int nblocks=0; /* Number of active blocks */
		/* Initialize array for random loop */
		unsigned int nr_idx_aux=stid*(options->num_environments*rpsize+(resize-rpsize))+env*besize;
		for (unsigned int block=0; block<besize; block++) {
			if (is_active(block,env,stid)) {
				structures->nr[nr_idx_aux+(nblocks++)]=block;
			}
		}

		while (nblocks>0) {
			/* Choose random block */
			unsigned int i=random()%(nblocks--);//--nblocks;
			unsigned int block=structures->nr[nr_idx_aux+i];

			/* Swap it */
			structures->nr[nr_idx_aux+i]=structures->nr[nr_idx_aux+nblocks];
			structures->nr[nr_idx_aux+nblocks]=0;

			unsigned int app=UINT_MAX;

			/* Check applications (minimum of available objects) */
			for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
				unsigned int obj=structures->lhs.object[o];
				unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
				unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);

				unsigned int value=structures->configuration.multiset[MU_IDX(obj,membr)]/mult;

				app = (value < app)? value : app;
			}

			/* If there are applications */
			if (app>0) {
				structures->nb[NB_IDX]+=app;

				for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
					unsigned int obj=structures->lhs.object[o];
					unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
					unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);

					structures->configuration.multiset[MU_IDX(obj,membr)]-=app*mult;
				}
			}
		}
	}
	
	#pragma omp critical (PRINT)
	pdp_out->print_block_selection(stid);
	#pragma omp critical (PRINT)
	pdp_out->print_temporal_configuration(sim);
		
	return true;
}

bool Simulator_omp_redir::selection_phase3(int sim, int stid) {
	#pragma omp critical (PRINT)
	pdp_out->print_dcba_phase(3);

	/* create a generator chosen by the
	 * environment variable GSL_RNG_TYPE */
	const gsl_rng_type * T_random;
	gsl_rng_env_setup();
	T_random = gsl_rng_default;
	gsl_rng * r_variable = gsl_rng_alloc(T_random);

#ifndef DEBUG
	struct timeval tval;
	gettimeofday(&tval,NULL);
	unsigned int seed = (tval.tv_usec+sim)*(stid+10);
#else
	unsigned int seed = 0;
#endif

	gsl_rng_set(r_variable,seed);


	for (int env=0; env<options->num_environments; env++) {
		for (unsigned int block=0; block<besize; block++) {
			int rule_ini=structures->ruleblock.rule_idx[block];
			int rule_end=structures->ruleblock.rule_idx[block+1];

			unsigned int N=0;

			if (block<bpsize) N=structures->nb[NB_IDX];
			else if (env==GET_ENVIRONMENT(structures->ruleblock.membrane[block])) N=structures->nb[NB_IDX];

			if (N==0) {
				for (unsigned int r = rule_ini; r < rule_end; r++) {
					if (r<rpsize) structures->nr[NR_P_IDX] = 0;
					else structures->nr[NR_E_IDX] = 0;
				}
				continue;
			}

			// TODO: check if this is more efficient than adding to a previous phase, or wherever
			/* Execution phase: update charges */
			/*   So execution phase only works with rules information */
			structures->configuration.membrane[CH_IDX(GET_MEMBRANE(structures->ruleblock.membrane[block]))]=GET_BETA(structures->ruleblock.membrane[block]);

			float cr=0.0,d=1.0;

			//start_timer();
			for (unsigned int r = rule_ini; r < rule_end; r++) {
				float p=0.0;
				unsigned int val=0;

				if (r>=rpsize) {
					p=structures->probability[options->num_environments*rpsize+(r-rpsize)];
				}
				else {
					p=structures->probability[env*rpsize+r];
				}

				cr = p / d;

				if (cr > 0.0) {
					if (r == rule_end-1)
						//structures->nr[env*structures->rule_size+r] = N;
						val=N;
					else {
						val=gsl_ran_binomial (r_variable, (double) cr, N);
						//structures->nr[env*structures->rule_size+r] = gsl_ran_binomial (r_variable, (double) cr, N);
						//cout << "Binomial " << N << ", " << (double) cr << " = " << structures->nr[env*structures->rule_size+r] << endl;
					}
				}

				if (r<rpsize) structures->nr[NR_P_IDX] = val;
				else structures->nr[NR_E_IDX] = val;

				N-=val;
				d*=(1-cr);
			}			
		}
	}

	gsl_rng_free (r_variable);
	
	#pragma omp critical (PRINT)
	pdp_out->print_rule_selection(stid);
	#pragma omp critical (PRINT)
	pdp_out->print_temporal_configuration(sim);
	
	return true;
}


int Simulator_omp_redir::execution(int sim, int stid) {
	#pragma omp critical (PRINT)
	pdp_out->print_dcba_phase(4);
	
	if (options->verbose>6) cout << endl << "========> Adding rules:" << endl;
	for (int env=0; env<options->num_environments; env++) {

		for (unsigned int r=0; r<rpsize; r++) {
			/* If there is applications */
			if (structures->nr[NR_P_IDX]>0) {
				if (options->verbose>6) cout << "Rule " << r << endl;
				for (unsigned int o=structures->rule.rhs_idx[r]; o<structures->rule.rhs_idx[r+1]; o++) {
					unsigned int obj=structures->rhs.object[o];
					unsigned int mult=GET_MULTIPLICITY(structures->rhs.mmultiplicity[o]);
					unsigned int membr=GET_MEMBR(structures->rhs.mmultiplicity[o]);

					structures->configuration.multiset[MU_IDX(obj,membr)]+=structures->nr[NR_P_IDX]*mult;

					//if (options->verbose>1) cout << "\t adds [obj_" << obj << "]^" << membr << "*" << structures->nr[NR_P_IDX]*mult << endl;
					//structures->nr[NR_P_IDX]=0;
				}
			}
		}
	}
	
	if (options->verbose>6) cout << endl << "Communication rules" << endl;

	for (unsigned int r=rpsize;r<resize;r++) {
		if (options->verbose>6) cout << "Rule " << r << endl;
		if (structures->nr[NR_E_IDX]>0) {
			for (unsigned int o=structures->rule.rhs_idx[r]; o<structures->rule.rhs_idx[r+1]; o++) {
				unsigned int obj=structures->rhs.object[o];
				unsigned int env=structures->rhs.mmultiplicity[o];

				structures->configuration.multiset[MU_IDX(obj,0)]+=structures->nr[NR_E_IDX];
				if (options->verbose>6) cout << "\t adds (obj_" << obj << ")^" << env << "*" << structures->nr[NR_E_IDX] << endl;
				//structures->nr[NR_E_IDX]=0;
			}
		}
	}

	return 0;
}


/***************************************************************************/
/***************************/
/* Only debugging purposes */
/***************************/

void Simulator_omp_redir::print_configuration (PDP_Psystem_REDIX::Configuration conf, int sim, const char* message, int step) {

	if (options->verbose<=1) return;

	#pragma omp critical (PRINT)
	{
	if (message!=NULL) {
	    cout << endl << "--------------------------" 
		<< endl << message;
	    if (step>=0) cout << " " << step;
	}
		
	cout << endl << "--------------------------" << endl << "CONFIGURATION" << endl << endl;
	cout << "Simulation: "<<sim<<endl;	
	
	//cout << "Multiset:" << endl;

	for (int env=0; env<options->num_environments; env++) {
		cout << "\tEnvironment " << env << ":";

		for (int memb=0; memb<options->num_membranes; memb++) {
			cout << endl << "\t\tMembrane " << memb << "(charge " << CHARGE_TO_CHAR(conf.membrane[CH_IDX(memb)]) << "): ";
			for (int obj=0; obj<options->num_objects; obj++) {
				cout << "Obj " << obj
						<< "*" << conf.multiset[MU_IDX(obj,memb)] << ", ";
			}
		}
		cout << endl;
	}
	
	// TODO: Print a full configuration as follows
	/*cout << "Charges:" << endl;
	
	for (int env=0; env<options->num_environments; env++) {
		cout << "Environment " << env << ": ";

		for (int memb=0; memb<options->num_membranes; memb++) {
			cout << "Membrane " << memb << "=" << CHARGE_TO_CHAR(conf.membrane[CH_IDX(memb)]) << ", ";
		}
		cout << endl;
	}*/	

	cout << "--------------------------" << endl;
	}
}

void Simulator_omp_redir::print_block_applications(uint * nb, int stid, const char* message) {

	if (options->verbose<=1) return;
	
	
	#pragma omp critical (PRINT)
	{
	if (message!=NULL) cout << endl << "--------------------------" 
		<< endl << message << endl;
	
	cout << "--------------------------" 
		<< endl << "BLOCKS APPLICATIONS" << endl;
	
	cout << "Simulation thread: " << stid << endl;
	for (int env=0; env<options->num_environments; env++) {
		cout << endl << "Environment " << env << endl << "\tBlocks: ";
		for (unsigned int block=0; block<besize; block++) {
			cout << "B" << block << "*" << nb[stid*options->num_environments*besize+env*besize+block] << ", ";
		}
	}
	cout << endl << "--------------------------" << endl;
	}
}
