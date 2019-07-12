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

#include "simulator_gpu_dir.h"

#include <math.h>
#include <limits.h>
#include <iostream>
#include <timestat.h>


// TODO: Unify this
#define IS_PI_BLOCK_ACTIVE(b) ((b)>=0 && (b)<bpsize && active_block[(sim*options->num_environments*besize+env*besize+(b))])
#define IS_ENV_BLOCK_ACTIVE(b) ((b)<besize && (b)>=bpsize && active_block[(sim*options->num_environments*besize+env*besize+(b))])

/* The following sets how the arrays are indexed */
#define AD_IDX(o,m) (sim*options->num_environments*esize+env*esize+(m)*msize+(o))
#define NB_IDX (sim*options->num_environments*besize+env*besize+block)
#define NR_P_IDX (sim*(options->num_environments*rpsize+(resize-rpsize))+env*rpsize+r)
#define NR_E_IDX (sim*(options->num_environments*rpsize+(resize-rpsize))+options->num_environments*rpsize+(r-rpsize))
#define CH_IDX(m) (sim*(options->num_environments*options->num_membranes)+env*options->num_membranes+(m))
#define MU_IDX(o,m) (sim*options->num_environments*esize+env*esize+(m)*msize+(o))

//unsigned int addition_size;
//float *debug_addition;

using namespace std;


void Simulator_gpu_dir::gold_selection_phase1_acu() {
	if (options->verbose>0)
	cout << endl << "--------------------------" << endl
	<< "Launching CPU code for phase 1: " << endl;
		
	
	/* Filters */
	if (options->verbose>0)
	cout << "Launching CPU code for filters: ";
	sdkResetTimer(&counters.timer);
	sdkStartTimer(&counters.timer);
	
	reset_activations();
	
	counters.timek1cpu=counters.timek2cpu=counters.timek3cpu=0.0f;
	
	for (int sim=0; sim < options->num_parallel_simulations; sim++) {
		for (int env=0; env < options->num_environments; env++) {
			for (unsigned int block=0; block<besize; block++) {
				
				/*** Filter 1 ***/
				uint membr=structures->ruleblock.membrane[block];
				bool active=false;

				// Case for rule blocks in Pi
				if (IS_MEMBRANE(membr)) {
					uint am=GET_MEMBRANE(membr);
					char ch=GET_ALPHA(membr);
					active=(structures->configuration.membrane[CH_IDX(am)] == ch);

				/*if (active) {
					if (m_c_charges[am]==4)
						m_c_charges[am]=ch;
					else if (m_c_charges[am]!=ch) {
						m_c_conflicts[am] = block;
						c_conflict = true;
					}

				}*/
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
					deactivate(block,env,sim);
				
				structures->nb[NB_IDX]= 0;
			}
		}
	}
	
	sdkStopTimer(&counters.timer);
	counters.timek1cpu=sdkGetTimerValue(&counters.timer);
	if (options->verbose>0)
	cout << counters.timek1cpu << "ms." << endl;
	
	for (int a=0; a < options->accuracy; a++) {
		/* Normalization */
		if (options->verbose>0)
		cout << "Launching CPU code for normalization: ";
		sdkResetTimer(&counters.timer);
		sdkStartTimer(&counters.timer);

		for (int sim=0; sim < options->num_parallel_simulations; sim++) {
			for (int env=0; env < options->num_environments; env++) {

				for (unsigned int i=0; i<esize; i++) {
					numerator[AD_IDX(i,0)]=ini_numerator[i];
				}

				/** Normalization - step 1 **/
				for (unsigned int block=0; block<besize; block++) {
					// If block is active
					if (! is_active(block,env,sim)) {
						for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
							unsigned int obj=structures->lhs.object[o];
							unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
							unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);
							//float inv=structures->lhs.imultiplicity[o];
	
							//denominator[AD_IDX(obj,membr)]*=mult;
							uint d=denominator[membr*options->num_objects+obj];
							numerator[AD_IDX(obj,membr)]-=d/mult;
						}
					}
				}
				/*for (unsigned int block=0; block<besize; block++) {
					// If block is active
					if (is_active(block,env,sim)) {
						for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
							unsigned int obj=structures->lhs.object[o];
							unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
							unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);
							//float inv=structures->lhs.imultiplicity[o];

							numerator[AD_IDX(obj,membr)]+=denominator[AD_IDX(obj,membr)]/mult;
						}
					}
				}*/

				/* Normalization - step 2 *
				 * Column minimum calculation */
				for (unsigned int block=0; block<besize; block++) {
					unsigned int minimum=0;
					// If block is activated
					if (is_active(block,env,sim)) {
						minimum=UINT_MAX;

						//cout << endl << "sim " << sim << ", env " << env << ", block " << block << ":" << endl;
						for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
							unsigned int obj=structures->lhs.object[o];
							unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
							unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);

							//unsigned int value = structures->configuration.multiset[MU_IDX(obj,membr)] / (mult*mult);
							//value = value / addition[AD_IDX(obj,membr)];
							//unsigned int value = structures->configuration.multiset[MU_IDX(obj,membr)] / (mult*mult*addition[AD_IDX(obj,membr)]);
							unsigned int value = (structures->configuration.multiset[MU_IDX(obj,membr)]*denominator[membr*options->num_objects+obj]) / (mult*mult*numerator[AD_IDX(obj,membr)]);
							
							//cout << "obj" << obj << "*" << mult << "'" << membr << "**" << structures->configuration.multiset[MU_IDX(obj,membr)] << ", n/d=" << numerator[AD_IDX(obj,membr)] << "/" << denominator[membr*options->num_objects+obj] << " => value=" << value << endl;
							
							minimum = (value<minimum)? value : minimum;
						}
						
						//cout << "MINIMUM = " << minimum << endl;

						//structures->nr[NB_IDX]=minimum;
					}
					structures->nr[NB_IDX]=minimum;
				}
			}
		}

		sdkStopTimer(&counters.timer);
		counters.timek2cpu+=sdkGetTimerValue(&counters.timer);
		if (options->verbose>0)
		cout << counters.timek2cpu << "ms." << endl;

		/* Update */
		if (options->verbose>0)
		cout << "Launching CPU code for updating: ";
		sdkResetTimer(&counters.timer);
		sdkStartTimer(&counters.timer);

		for (int sim=0; sim < options->num_parallel_simulations; sim++) {
			for (int env=0; env < options->num_environments; env++) {
				for (unsigned int block=0; block<besize; block++) {
					if (structures->nr[NB_IDX]>0) {
						for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
							unsigned int obj=structures->lhs.object[o];
							unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
							unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);

							//structures->configuration.multiset[MU_IDX(obj,membr)]-=block_min[BM_IDX]*mult;
							if (structures->configuration.multiset[MU_IDX(obj,membr)] < structures->nr[NB_IDX]*mult)
								cout << "Error deleting LHS: sim=" << sim << ", env=" << env << ", block=" << block << ", membr=" << membr << ", o=" << obj << ": deleting " << structures->nr[NB_IDX]*mult << ", there is " << structures->configuration.multiset[MU_IDX(obj,membr)] << endl;
							else {
								structures->configuration.multiset[MU_IDX(obj,membr)] -= structures->nr[NB_IDX]*mult;
							}
						}
						structures->nb[NB_IDX]+=structures->nr[NB_IDX];
					}
				}
				/** Filter 2 **/
				for (unsigned int block=0; block<besize; block++) {
					if (is_active(block,env,sim)) {
						// Using new registers avoid memory accesses on the for loop
						uint o_init=structures->ruleblock.lhs_idx[block];
						uint o_end=structures->ruleblock.lhs_idx[block+1];
						for (uint o=o_init; o < o_end; o++) {
							uint obj=structures->lhs.object[o];
							uint membr=structures->lhs.mmultiplicity[o];
							uint mult=GET_MULTIPLICITY(membr);
							membr=GET_MEMBR(membr);

							// Check if we have enough objects to apply the block
							if (structures->configuration.multiset[MU_IDX(obj,membr)]<mult) {
								deactivate(block,env,sim);
								break;
							}
						}
					}
				}
			}
		}

		sdkStopTimer(&counters.timer);
		counters.timek3cpu+=sdkGetTimerValue(&counters.timer);
		if (options->verbose>0)
		cout << counters.timek3cpu << "ms." << endl;
	}
	
	counters.timesp1cpu=counters.timek1cpu+counters.timek2cpu+counters.timek3cpu;
	
	print_block_applications(structures->nb);
	print_configuration(structures->configuration);
	
}

void Simulator_gpu_dir::gold_selection_phase1() {
	if (options->verbose>0)
	cout << endl << "--------------------------" << endl
	<< "Launching CPU code for phase 1: " << endl;
		
	/* Filters */
	if (options->verbose>0)
	cout << "Launching CPU code for filters: ";
	sdkResetTimer(&counters.timer);
	sdkStartTimer(&counters.timer);
	
	reset_activations();
	
	counters.timek1cpu=counters.timek2cpu=counters.timek3cpu=0.0f;
	
	for (int sim=0; sim < options->num_parallel_simulations; sim++) {
		for (int env=0; env < options->num_environments; env++) {
			for (unsigned int block=0; block<besize; block++) {
				
				/*** Filter 1 ***/
				uint membr=structures->ruleblock.membrane[block];
				bool active=false;

				// Case for rule blocks in Pi
				if (IS_MEMBRANE(membr)) {
					uint am=GET_MEMBRANE(membr);
					char ch=GET_ALPHA(membr);
					active=(structures->configuration.membrane[CH_IDX(am)] == ch);

				/*if (active) {
					if (m_c_charges[am]==4)
						m_c_charges[am]=ch;
					else if (m_c_charges[am]!=ch) {
						m_c_conflicts[am] = block;
						c_conflict = true;
					}

				}*/
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
					deactivate(block,env,sim);
				
				structures->nb[NB_IDX]= 0;
			}
		}
	}
	
	sdkStopTimer(&counters.timer);
	counters.timek1cpu=sdkGetTimerValue(&counters.timer);
	if (options->verbose>0)
	cout << counters.timek1cpu << "ms." << endl;
	
	for (int a=0; a < options->accuracy; a++) {
		/* Normalization */
		if (options->verbose>0)
		cout << "Launching CPU code for normalization: ";
		sdkResetTimer(&counters.timer);
		sdkStartTimer(&counters.timer);

		for (int sim=0; sim < options->num_parallel_simulations; sim++) {
			for (int env=0; env < options->num_environments; env++) {

				for (unsigned int i=0; i<esize; i++) {
					addition[AD_IDX(i,0)]=0.0;
				}

				/** Normalization - step 1 **/
				for (unsigned int block=0; block<besize; block++) {
					// If block is active
					if (is_active(block,env,sim)) {
						for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
							unsigned int obj=structures->lhs.object[o];
							//unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
							unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);
							float inv=structures->lhs.imultiplicity[o];

							addition[AD_IDX(obj,membr)]+=inv;
						}
					}
				}

				/* Normalization - step 2 *
				 * Column minimum calculation */
				for (unsigned int block=0; block<besize; block++) {
					unsigned int minimum=0;
					// If block is activated
					if (is_active(block,env,sim)) {
						minimum=UINT_MAX;

						//cout << endl << "sim " << sim << ", env " << env << ", block " << block << ":" << endl;
						for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
							unsigned int obj=structures->lhs.object[o];
							unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
							unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);

							//unsigned int value = structures->configuration.multiset[MU_IDX(obj,membr)] / (mult*mult);
							//value = value / addition[AD_IDX(obj,membr)];
							unsigned int value = structures->configuration.multiset[MU_IDX(obj,membr)] / (mult*mult*addition[AD_IDX(obj,membr)]);
							
							//cout << "obj" << obj << "*" << mult << "'" << membr << "**" << structures->configuration.multiset[MU_IDX(obj,membr)] << ", add=" << addition[AD_IDX(obj,membr)] << " => value=" << value << endl;
							
							minimum = (value<minimum)? value : minimum;
						}
						
						//cout << "MINIMUM = " << minimum << endl;

						structures->nr[NB_IDX]=minimum;
					}
					structures->nr[NB_IDX]=minimum;
				}
			}
		}

		sdkStopTimer(&counters.timer);
		counters.timek2cpu+=sdkGetTimerValue(&counters.timer);
		if (options->verbose>0)
		cout << counters.timek2cpu << "ms." << endl;

		/* Update */
		if (options->verbose>0)
		cout << "Launching CPU code for updating: ";
		sdkResetTimer(&counters.timer);
		sdkStartTimer(&counters.timer);

		for (int sim=0; sim < options->num_parallel_simulations; sim++) {
			for (int env=0; env < options->num_environments; env++) {
				for (unsigned int block=0; block<besize; block++) {
					if (structures->nr[NB_IDX]>0) {
						for (unsigned int o=structures->ruleblock.lhs_idx[block]; o<structures->ruleblock.lhs_idx[block+1]; o++) {
							unsigned int obj=structures->lhs.object[o];
							unsigned int mult=GET_MULTIPLICITY(structures->lhs.mmultiplicity[o]);
							unsigned int membr=GET_MEMBR(structures->lhs.mmultiplicity[o]);

							//structures->configuration.multiset[MU_IDX(obj,membr)]-=block_min[BM_IDX]*mult;
							if (structures->configuration.multiset[MU_IDX(obj,membr)] < structures->nr[NB_IDX]*mult)
								cout << "Error deleting LHS: sim=" << sim << ", env=" << env << ", block=" << block << ", membr=" << membr << ", o=" << obj << ": deleting " << structures->nr[NB_IDX]*mult << ", there is " << structures->configuration.multiset[MU_IDX(obj,membr)] << endl;
							else {
								structures->configuration.multiset[MU_IDX(obj,membr)] -= structures->nr[NB_IDX]*mult;
							}
						}
						structures->nb[NB_IDX]+=structures->nr[NB_IDX];
					}
				}
				/** Filter 2 **/
				for (unsigned int block=0; block<besize; block++) {
					if (is_active(block,env,sim)) {
						// Using new registers avoid memory accesses on the for loop
						uint o_init=structures->ruleblock.lhs_idx[block];
						uint o_end=structures->ruleblock.lhs_idx[block+1];
						for (uint o=o_init; o < o_end; o++) {
							uint obj=structures->lhs.object[o];
							uint membr=structures->lhs.mmultiplicity[o];
							uint mult=GET_MULTIPLICITY(membr);
							membr=GET_MEMBR(membr);

							// Check if we have enough objects to apply the block
							if (structures->configuration.multiset[MU_IDX(obj,membr)]<mult) {
								deactivate(block,env,sim);
								break;
							}
						}
					}
				}
			}
		}

		sdkStopTimer(&counters.timer);
		counters.timek3cpu+=sdkGetTimerValue(&counters.timer);
		if (options->verbose>0)
		cout << counters.timek3cpu << "ms." << endl;
	}
	
	counters.timesp1cpu=counters.timek1cpu+counters.timek2cpu+counters.timek3cpu;
	
	print_block_applications(structures->nb);
	print_configuration(structures->configuration);
}


void Simulator_gpu_dir::gold_selection_phase2() {
	
	if (options->verbose>0)
	cout << endl << "--------------------------" << endl <<
		"Launching CPU code for phase 2: ";

	sdkResetTimer(&counters.timer);
	sdkStartTimer(&counters.timer);
	
	for (int sim=0; sim < options->num_parallel_simulations; sim++)
		for (int env=0; env<options->num_environments; env++) {
			/* For each active block *
			 * In a random order */
			unsigned int nblocks=0; /* Number of active blocks */
			/* Initialize array for random loop */
			unsigned int nr_idx_aux=sim*options->num_environments*besize+env*besize;
			for (unsigned int block=0; block<besize; block++) {
				if (is_active(block,env,sim)) {
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
	
	sdkStopTimer(&counters.timer);
	counters.timesp2cpu=sdkGetTimerValue(&counters.timer);
	
	if (options->verbose>0)
	cout << counters.timesp2cpu << "ms." << endl;

	print_block_applications(structures->nb);
	print_configuration(structures->configuration);

}

void Simulator_gpu_dir::gold_selection_phase3() {
	//unsigned int etid=(mode==1)?0:omp_get_thread_num();

	if (options->verbose>0)
	cout << endl << "--------------------------" << endl <<
		"Launching CPU code for phase 3: ";
	
	sdkResetTimer(&counters.timer);
	sdkStartTimer(&counters.timer);

	/* create a generator chosen by the
	 * environment variable GSL_RNG_TYPE */
	const gsl_rng_type * T_random;
	gsl_rng_env_setup();
	T_random = gsl_rng_default;
	gsl_rng * r_variable = gsl_rng_alloc(T_random);

	for (int sim=0; sim < options->num_parallel_simulations; sim++)
		for (int env=0; env<options->num_environments; env++) {
			if (options->verbose>1)
				cout << "sim " << sim << ", env " << env << ":" << endl;
			for (unsigned int block=0; block<besize; block++) {
				int rule_ini=structures->ruleblock.rule_idx[block];
				int rule_end=structures->ruleblock.rule_idx[block+1];

				unsigned int N=0;

				if (block<bpsize) N=structures->nb[NB_IDX];
				else if (env==GET_ENVIRONMENT(structures->ruleblock.membrane[block])) N=structures->nb[NB_IDX];

				if (options->verbose>1)
					cout << endl << "\tBlock " << block << ", N=" << N << ":";

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
				structures->configuration.membrane[GET_MEMBRANE(structures->ruleblock.membrane[block])]=GET_BETA(structures->ruleblock.membrane[block]);

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

					if (options->verbose>1)
						cout << "r_" << r-rule_ini << "(p=" << p << ",n=";

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

					if (options->verbose>1)
						cout << val << "), ";
				}
				
				/*double time=end_timer();
				if (time>30000.0) {
					cout << endl << "RNG ERROR: In sim " << sim << ", env " << env <<
						", block " << block << ": N=" << structures->nb[NB_IDX];
					for (unsigned int r = rule_ini; r < rule_end; r++) {
						float p=0.0;
						if (r>=rpsize) {
							p=structures->probability[options->num_environments*rpsize+(r-rpsize)];
						}
						else {
							p=structures->probability[env*rpsize+r];
						}
						
						cout << "r_" << r-rule_ini << "(p=" << p << ",n=";
						
						if (r<rpsize) cout << structures->nr[NR_P_IDX] << "), ";
						else cout << structures->nr[NR_E_IDX] << "), ";
					}
						
				}*/
			}}

	gsl_rng_free (r_variable);
	
	sdkStopTimer(&counters.timer);
	counters.timesp3cpu=sdkGetTimerValue(&counters.timer);
	if (options->verbose>0)
	cout << counters.timesp3cpu << "ms." << endl;

}


unsigned int Simulator_gpu_dir::gold_execution() {
	/* PHASE 4: EXECUTION */
	/* For each environemnt */

	if (options->verbose>0) {
	cout << endl << "--------------------------" << endl;
	cout << "Launching CPU code for phase 4: ";
	}
	
	sdkResetTimer(&counters.timer);
	sdkStartTimer(&counters.timer);
	
	for (int sim=0; sim<options->num_parallel_simulations; sim++)
	for (int env=0; env<options->num_environments; env++) {
		if (options->verbose>1)
			cout << endl << "Environment " << env << endl;

		for (unsigned int r=0; r<rpsize; r++) {
			/* If there is applications */
			if (structures->nr[NR_P_IDX]>0) {
				if (options->verbose>1) cout << "Rule " << r << endl;
				for (int o=structures->rule.rhs_idx[r]; o<structures->rule.rhs_idx[r+1]; o++) {
					unsigned int obj=structures->rhs.object[o];
					unsigned int mult=GET_MULTIPLICITY(structures->rhs.mmultiplicity[o]);
					unsigned int membr=GET_MEMBR(structures->rhs.mmultiplicity[o]);

					structures->configuration.multiset[MU_IDX(obj,membr)]+=structures->nr[NR_P_IDX]*mult;

					if (options->verbose>1) cout << "\t adds [obj_" << obj << "]^" << membr << "*" << structures->nr[NR_P_IDX]*mult << endl;
					//structures->nr[NR_P_IDX]=0;
				}
			}
		}
	}
	
	if (options->verbose>1) cout << endl << "Communication rules" << endl;

	for (int sim=0; sim < options->num_parallel_simulations; sim++)
	for (unsigned int r=rpsize;r<resize;r++) {
		if (options->verbose>1) cout << "Rule " << r << endl;
		if (structures->nr[NR_E_IDX]>0) {
			for (unsigned int o=structures->rule.rhs_idx[r]; o<structures->rule.rhs_idx[r+1]; o++) {
				unsigned int obj=structures->rhs.object[o];
				unsigned int env=structures->rhs.mmultiplicity[o];

				structures->configuration.multiset[MU_IDX(obj,0)]+=structures->nr[NR_E_IDX];
				if (options->verbose>1) cout << "\t adds (obj_" << obj << ")^" << env << "*" << structures->nr[NR_E_IDX] << endl;
				//structures->nr[NR_E_IDX]=0;
			}
		}
	}
	
	sdkStopTimer(&counters.timer);
	counters.timesp4cpu=sdkGetTimerValue(&counters.timer);
	
	if (options->verbose>0)
	cout << counters.timesp4cpu << "ms." << endl;

	return 0;
}


/***************************************************************************/
/***************************/
/* Only debugging purposes */
/***************************/

void Simulator_gpu_dir::print_configuration (PDP_Psystem_REDIX::Configuration conf) {

	if (options->verbose<=1) return;

	cout << endl << "--------------------------" << endl << "CONFIGURATION" << endl << endl;
	//cout << "Multiset:" << endl;

	for (int sim=0; sim<options->num_simulations; sim++) {
		cout << "Simulation: "<<sim<<endl;	
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
	}}
	
	// TODO: Print a full configuration as follows
	/*cout << "Charges:" << endl;
	
	for (int sim=0; sim<options->num_simulations; sim++) {
		cout << "Simulation: "<<sim<<endl;	
	for (int env=0; env<options->num_environments; env++) {
		cout << "Environment " << env << ": ";

		for (int memb=0; memb<options->num_membranes; memb++) {
			cout << "Membrane " << memb << "=" << CHARGE_TO_CHAR(conf.membrane[CH_IDX(memb)]) << ", ";
		}
		cout << endl;
	}}*/

	cout << "--------------------------" << endl;
}

void Simulator_gpu_dir::print_block_applications(uint * nb) {

	if (options->verbose<=1) return;
	
	cout << endl << "--------------------------" << endl << "BLOCKS APPLICATIONS" << endl;
	
	for (int sim=0; sim<options->num_simulations; sim++)
	for (int env=0; env<options->num_environments; env++) {
		cout << endl << "Environment " << env << endl << "\tBlocks: ";
		for (unsigned int block=0; block<besize; block++) {
			cout << "B" << block << "*" << nb[NB_IDX] << ", ";
		}
	}
	cout << endl << "--------------------------" << endl;

}

