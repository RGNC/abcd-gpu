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

#include "simulator_seq_table.h"

#include <math.h>
#include <limits.h>
#include <iostream>
#include <omp.h>
//#include <lib/inc/timestat.h>

using namespace std;

/*************/
/* MAIN LOOP */
/*************/

bool Simulator_seq_table::run(){
	step(options->time);
}

bool Simulator_seq_table::step(int k) {
	
	//init_time();

	cout << "[2] STARTING THE SIMULATION: Using table based algorithm" << endl;

	/* Create initial table */
	init_table();

	//cout << "Create table:" <<endl; print_time(stdout);

	print_table();

	/* Repeat for each simulation */
	for (int simu=0; simu<options->num_simulations; simu++) {
		init_activations();

		//cout << endl << "Create activations: "; print_time(stdout);

		/* MAIN LOOP */
		for (int i=0; i<k; i++) {
			selection();

			//cout << endl << "Selection: "; print_time(stdout);
			execution();

			//cout << endl << "Execution: "; print_time(stdout); cout << endl;
			print_configuration();
		}

		delete_activations();
	}

	delete_table();

	return true;
}


/***************************************/
/* Auxiliary functions Implementations */
/***************************************/

float * Simulator_seq_table::init_table() {
	rowsize=options->num_objects*options->num_membranes;
	colsize=options->num_rule_blocks;

	cout << "Creating table T: "<< rowsize*colsize*sizeof(float) << "Bytes";

	/* Create table T */
	table=new float[colsize*rowsize];

	/* Initialize to 0 */
	for (unsigned int i=0; i<rowsize; i++)
		for (unsigned int j=0; j< colsize; j++)
			table[i*colsize+j]=0;

	/* Assign values to 1/multiplicity */
	for (int b=0; b < options->num_rule_blocks; b++) {
		for (unsigned int o=structures->ruleblock.lhs_ini[b]; o<structures->ruleblock.lhs_ini[b]+structures->ruleblock.lhs_length[b]; o++) {
			table[(structures->lhs.membrane[o]*options->num_objects+structures->lhs.object[o])*colsize+b]=
					1/(float) structures->lhs.multiplicity[o];
		}
	}

	return table;
}

void Simulator_seq_table::init_activations() {
	//unsigned int rowsize=options->num_objects*options->num_membranes;
	//unsigned int colsize=options->num_rule_blocks;

	/* Initialize auxiliary structures */
	active_col = new bool[colsize*options->num_environments];
	active_row = new bool[rowsize*options->num_environments];
	init_active_row = new bool[rowsize];
	addition = new float[rowsize*options->num_environments];
	block_min = new unsigned int[colsize*options->num_environments];

	/* Initialize Nb */
	structures->nb_size= ((options->num_rule_blocks)*options->num_environments+options->num_blocks_env);//*options->num_parallel_simulations;
	structures->nb = new PDP_Psystem_SAB::MULTIPLICITY [structures->nb_size];

	/* Initialize Nr */
	structures->nr_size= ((structures->rule_size)*options->num_environments+structures->env_rule_size);//*options->num_parallel_simulations;
	structures->nr = new PDP_Psystem_SAB::MULTIPLICITY [structures->nr_size];

	unsigned int nrmem = structures->nr_size * sizeof(PDP_Psystem_SAB::MULTIPLICITY);
	unsigned int nbmem = structures->nb_size * sizeof(PDP_Psystem_SAB::MULTIPLICITY);


	/* Assign initial values */
	for (unsigned int i=0;i<colsize*options->num_environments;i++) {
		active_col[i]=true;
		block_min[i]=0;
	}

	/* Inactive rows with 0 value */
	for (unsigned int i=0; i<rowsize;i++) {
		init_active_row[i]=false;
		for (unsigned int j=0;j<colsize;j++) {
			if (table[i*colsize+j]>0)
				init_active_row[i]=true;
		}
	}

	for (unsigned int i=0;i<rowsize*options->num_environments;i++) {
		active_row[i]=init_active_row[i%rowsize];
		addition[i]=0;
	}

	if (options->verbose>0) {
		cout << endl << "--------------------------" << endl << "Memory used by the table algorithm:" << endl;

		unsigned long int tmem=colsize*rowsize*sizeof(float);
		unsigned long int armem=rowsize*(options->num_environments+1)*sizeof(bool);
		unsigned long int acmem=colsize*options->num_environments*sizeof(bool);
		unsigned long int admem=rowsize*options->num_environments*sizeof(float);
		unsigned long int bmmem=colsize*options->num_environments*sizeof(unsigned int);

		cout << "Table: " << tmem << " (" << tmem/1024 << "KB)" << endl;
		cout << "Active rows: " << armem << " (" << armem/1024 << "KB)" << endl;
		cout << "Active columns: " << acmem << " (" << acmem/1024 << "KB)" << endl;
		cout << "Addition: " << admem << " (" << admem/1024 << "KB)" << endl;
		cout << "Block min: " << bmmem << " (" << bmmem/1024 << "KB)" << endl;
		cout << "Nr: " << nrmem << " (" << nrmem/1024 << "KB)" << endl;
		cout << "Nb: " << nbmem << " (" << nbmem/1024 << "KB)" << endl;

		unsigned long int mem=tmem+armem+acmem+admem+bmmem+nrmem+nbmem;
		int count=0;
		float div=1;
		char unit[6]={' ','K','M','G','T','P'};
		while ((mem/div)>1023 && count<3){
			div*=1024;
			count++;
		}
		cout << "TOTAL: " << mem << " (" << mem/div << " " << unit[count] << "B)" << endl << endl;
	}
}

void Simulator_seq_table::delete_table() {
	delete [] table;
}

void Simulator_seq_table::delete_activations() {
	delete [] active_col;
	delete [] active_row;
	delete [] init_active_row;
	delete [] addition;
	delete [] block_min;
	delete [] structures->nb;
	delete [] structures->nr;
}


/***************/
/* MAIN PHASES */
/***************/

unsigned int Simulator_seq_table::selection(){
	//unsigned int esize=options->num_environments;
	//unsigned int bsize=options->num_simulations*options->num_environments;
	unsigned int esize=options->num_objects*options->num_membranes;
	unsigned int msize=options->num_objects;

	/* Initialize nr for starting new selection */
	for (unsigned int j=0; j<structures->nr_size; j++){
		structures->nr[j]=0;
	}
	for (unsigned int j=0; j<structures->nb_size; j++){
		structures->nb[j]=0;
	}

	/* PHASE 1: DISTRIBUTION */
	selection_phase1();

	/* PHASE 2: MAXIMALITY */
	selection_phase2();

	/* PHASE 3: PROBABILITY */
	selection_phase3();

	return 1;
}

unsigned int Simulator_seq_table::execution() {
	unsigned int esize=options->num_objects*options->num_membranes;
	unsigned int msize=options->num_objects;

	/* PHASE 4: EXECUTION */
	/* For each environemnt */
	for (unsigned int env=0; env<options->num_environments; env++) {
		if (options->verbose>1)
			cout << endl << "Environment " << env << endl;

		for (unsigned int rule=0; rule<structures->rule_size; rule++) {
			if (structures->nr[env*structures->rule_size+rule]>0) {
				for (int o=structures->rule.rhs_ini[rule]; o<structures->rule.rhs_ini[rule]+structures->rule.rhs_length[rule]; o++) {
					unsigned int obj=structures->rhs.object[o];
					unsigned int mult=structures->rhs.multiplicity[o];
					unsigned int membr=structures->rhs.membrane[o];

					structures->configuration.multiset[env*esize+membr*msize+obj]+=structures->nr[env*structures->rule_size+rule]*mult;
				}
			}
		}
	}

	if (options->verbose>1) {
		cout << endl <<endl << "*************************************" << endl << "Phase 4." << endl;
		print_configuration();
	}

	return 0;

}


/*********************/
/* Selection methods */
/*********************/

void Simulator_seq_table::selection_phase1() {
	unsigned int esize=options->num_objects*options->num_membranes;
	unsigned int msize=options->num_objects;

	/* For each environment */
	for (int env=0; env<options->num_environments; env++) {

		/* Filter 1 */
		for (int block=0; block<options->num_rule_blocks; block++) {
			unsigned int am=structures->ruleblock.active_membrane[block];
			char ch=structures->ruleblock.lhs_charge[block];

			/* If not equal charges in lhs */
			if (structures->configuration.membrane[env*options->num_membranes+am] != ch)
				active_col[env*colsize+block]=false;
		}

		/* Filter 2 */
		for (int memb=0; memb<options->num_membranes; memb++) {
			for (int obj=0; obj<options->num_objects; obj++) {
				unsigned int oxm=memb*msize+obj;
				/* If object has no multiplicity */
				if (structures->configuration.multiset[env*esize+oxm]==0) {
					active_row[env*rowsize+oxm]=false;
					/* Unset corresponding blocks */
					for (int block=0; block<colsize; block++) {
						if (table[oxm*colsize+block]>0)
							active_col[env*colsize+block]=false;
					}
				}
			}
		}

		/* TODO: Generate and select table with mutually consistent blocks */

		int A=options->accuracy;
		bool block_sel=false;

		do {
			/* Normalization - step 1 */
			for (int oxm=0; oxm<rowsize; oxm++) {
				addition[env*rowsize+oxm]=0;
				if (active_row[env*rowsize+oxm]) {
					for (int block=0; block<colsize; block++)
						if (active_col[env*colsize+block] && table[oxm*colsize+block] > 0)
							addition[env*rowsize+oxm]+=table[oxm*colsize+block];
				}
			}

			/* Normalization - step 2 *
			 * Column minimum calculation */
			for (int block=0; block<colsize; block++) {
				if (active_col[env*colsize+block]) {
					unsigned int minimum=0;
					bool min_set=false;
					for (unsigned int oxm=0; oxm<rowsize; oxm++) {
						if (active_row[env*rowsize+oxm] && table[oxm*colsize+block]>0) {

							unsigned int value =
									(unsigned int) floor(((double) table[oxm*colsize+block]*table[oxm*colsize+block]
														  / addition[env*rowsize+oxm])
														  * structures->configuration.multiset[env*esize+oxm]);

							if (! min_set){
								minimum=value;
								min_set=true;
							}
							if (value<minimum) {
								minimum=value;
							}
						}
					}

					//structures->nb[env*colsize+block]+=minimum;
					block_min[env*colsize+block]=minimum;
				}
			}

			if (options->verbose>1) cout << endl <<endl << "*************************************" << endl << "Phase 1. Environment " << env << ", A=" << A << endl;
			print_activations(env);

			/* Deleting LHS *
			 * Adding block applications *
			 * Filter 2 */
			block_sel=false;
			for (int block=0; block<colsize; block++) {
				if (/*active_col[env*colsize+block] &&*/ block_min[env*colsize+block]>0) {
					block_sel=true;
					/* Consume LHS */
					for (unsigned int o=structures->ruleblock.lhs_ini[block]; o<structures->ruleblock.lhs_ini[block]+structures->ruleblock.lhs_length[block]; o++) {
						unsigned int obj=structures->lhs.object[o];
						unsigned int mult=structures->lhs.multiplicity[o];
						unsigned int membr=structures->lhs.membrane[o];

						/*cout << "ELIMINANDO: bloque " << block << " con " << block_min[env*colsize+block] << endl;
						cout << "\t Obj " << obj << "*" << mult << "'memb" << membr << endl;
						cout << "\t " << structures->configuration.multiset[env*esize+membr*msize+obj] << " - " <<  block_min[env*colsize+block] << "*" << mult << "=";*/
						/* Delete block application */
						structures->configuration.multiset[env*esize+membr*msize+obj]-=block_min[env*colsize+block]*mult;

						//cout << structures->configuration.multiset[env*esize+membr*msize+obj] << endl;

						/* Filter 2 */
						if (structures->configuration.multiset[env*esize+membr*msize+obj]==0) {
							unsigned int oxm=membr*msize + obj;
							active_row[env*rowsize+oxm]=false;
							/* Unset corresponding blocks */
							for (unsigned int block=0; block<colsize; block++) {
								if (table[oxm*colsize+block]>0)
									active_col[env*colsize+block]=false;
							}
						}
					}

					/* Add applications to block */
					structures->nb[env*colsize+block]+=block_min[env*colsize+block];
					block_min[env*colsize+block]=0;
				}
			}

			//cout << endl <<endl << "*************************************" << endl << "Phase 1. Environment " << env << ", A=" << A << ". After updating LHS" << endl;
			//print_activations_t(structures,options,env);

			//print_configuration_t(structures,options);

			A--;
		} while (A>0 && block_sel);
	}
}

void Simulator_seq_table::selection_phase2(){
	unsigned int esize=options->num_objects*options->num_membranes;
	unsigned int msize=options->num_objects;

	if (options->verbose>1)
		cout << endl <<endl << "*************************************" << endl << "Phase 2." << endl;

	/* For each environment */
	for (int env=0; env<options->num_environments; env++) {

		/* For each active block *
		 * In a random order */
		unsigned int nblocks=0; /* Number of active blocks */
		/* Initialize array for random loop */
		for (unsigned int block=0; block<colsize; block++) {
			if (active_col[env*colsize+block]) block_min[env*colsize+(nblocks++)]=block;
		}

		while (nblocks>0) {
			/* Choose random block */
			unsigned int i=random()%(nblocks--);
			unsigned int block=block_min[env*colsize+i];
			/* Pivot it */
			block_min[env*colsize+i]=block_min[env*colsize+nblocks];
			block_min[env*colsize+nblocks]=0;

			unsigned int app=0;
			bool min_set=false;
			/* Check applications (minimum of available objects) */
			for (unsigned int o=structures->ruleblock.lhs_ini[block]; o<structures->ruleblock.lhs_ini[block]+structures->ruleblock.lhs_length[block]; o++) {
				unsigned int obj=structures->lhs.object[o];
				unsigned int mult=structures->lhs.multiplicity[o];
				unsigned int membr=structures->lhs.membrane[o];

				unsigned int value=(unsigned int) floor(structures->configuration.multiset[env*esize+membr*msize+obj]/mult);

				if (! min_set){
					app=value;
					min_set=true;
				}
				if (value<app) {
					app=value;
				}
			}

			//cout << "In environment " << env << ", checking block " << block << ", having " << app << " applications" << endl;
			/* If there are applications */
			if (app>0) {
				structures->nb[env*colsize+block]+=app;

				for (unsigned int o=structures->ruleblock.lhs_ini[block]; o<structures->ruleblock.lhs_ini[block]+structures->ruleblock.lhs_length[block]; o++) {
					unsigned int obj=structures->lhs.object[o];
					unsigned int mult=structures->lhs.multiplicity[o];
					unsigned int membr=structures->lhs.membrane[o];

					structures->configuration.multiset[env*esize+membr*msize+obj]-=app*mult;
				}
			}
		}
	}

	//print_activations_t(structures,options);
	print_block_applications();
	if (options->verbose>1) cout << endl << endl;

}

void Simulator_seq_table::selection_phase3() {
	unsigned int esize=options->num_objects*options->num_membranes;
	unsigned int msize=options->num_objects;

	if (options->verbose>1)
		cout << endl <<endl << "*************************************" << endl << "Phase 3." << endl;

	/* create a generator chosen by the
	 * environment variable GSL_RNG_TYPE */
	const gsl_rng_type * T_random;
	gsl_rng_env_setup();
	T_random = gsl_rng_default;
	gsl_rng * r_variable = gsl_rng_alloc(T_random);

	for (unsigned int env=0; env<options->num_environments; env++) {
		if (options->verbose>1) cout << endl << "Environment " << env << endl;
		for (unsigned int block=0; block<options->num_rule_blocks; block++) {

			int rule_ini=structures->ruleblock.rule_ini[block];
			int rule_length=structures->ruleblock.rule_length[block];

			unsigned int N=structures->nb[env*options->num_rule_blocks+block];

			if (options->verbose>1)
				cout << endl << "\tBlock " << block << ", N=" << N << ":";

			if (N==0) continue;

			float cr=0.0,d=1.0;

			for (int r = rule_ini; r < rule_length+rule_ini; r++) {
				cr = structures->probability[env*structures->rule_size+r] / d;

				if (options->verbose>1)
					cout << "r_" << r-rule_ini << "(p=" << structures->probability[env*structures->rule_size+r] << ",n=";

				if (cr > 0.0) {
					if (r == rule_length+rule_ini-1)
						structures->nr[env*structures->rule_size+r] = N;
					else {
						structures->nr[env*structures->rule_size+r] = gsl_ran_binomial (r_variable, (double) cr, N);
						//cout << "Binomial " << N << ", " << (double) cr << " = " << structures->nr[env*structures->rule_size+r] << endl;
					}
				}
				else {
					structures->nr[env*structures->rule_size+r] = 0;
				}

				N-=structures->nr[env*structures->rule_size+r];
				d*=(1-cr);

				if (options->verbose>1)
					cout << structures->nr[env*structures->rule_size+r] << "), ";
			}
		}
	}

	gsl_rng_free (r_variable);

}


/***************************/
/* Only debugging purposes */
/***************************/

void Simulator_seq_table::print_configuration () {

	if (options->verbose==0) return;

	cout << endl << "--------------------------" << endl << "CONFIGURATION" << endl << endl;

	cout << "Multiset" << endl;

	for (int sim=0; sim<options->num_parallel_simulations; sim++) {
		for (int env=0; env<options->num_environments; env++) {
			cout << "Simulation " << sim << ", environment " << env << ":";

			for (int memb=0; memb<options->num_membranes; memb++) {
				cout << endl << "\t\tMembrane " << memb << ": ";
				for (int obj=0; obj<options->num_objects; obj++) {
					cout << "Obj " << obj
							<< "*" << structures->configuration.multiset[sim*options->num_objects*options->num_membranes*options->num_environments+env*options->num_objects*options->num_membranes+memb*options->num_objects+obj] << ", ";
				}
			}

			cout << endl;
		}
	}
}

void Simulator_seq_table::print_table(){

	if (options->verbose<=1) return;

	cout << endl << "--------------------------" << endl << "Table T:" << endl;
	cout << "Sizes: column (" << colsize << "), row (" << rowsize << ")" << endl;

	for (int j=0; j< colsize; j++) {
		cout << "\t| " << j;
	}
	cout << endl;

	cout.precision(2);

	for (int i=0; i<rowsize; i++) {
		cout << i << " |";
		for (int j=0; j< colsize; j++) {
			cout << "\t| " << table[i*colsize+j];
		}
		cout << endl;
	}

	cout.precision(6);
}

void Simulator_seq_table::print_activations (unsigned int env) {

	if (options->verbose<=1) return;

	cout << endl << "Activated columns: ";
	for (int i=colsize*env;i<colsize*(env+1);i++)
		cout << active_col[i] << ", ";

	cout << endl << "Activated rows: ";
	for (int i=rowsize*env;i<rowsize*(env+1);i++)
			cout << active_row[i] << ", ";

	cout << endl << "Addition vector: ";
	for (int i=rowsize*env;i<rowsize*(env+1);i++)
			cout << addition[i] << ", ";

	cout << endl << "Minimum vector: ";
	for (int i=colsize*env;i<colsize*(env+1);i++)
			cout << block_min[i] << ", ";

	cout<<endl;

	print_configuration();
}

void Simulator_seq_table::print_block_applications() {

	if (options->verbose<=1) return;

	cout << endl << "--------------------------" << endl << "BLOCKS APPLICATIONS" << endl;

	for (unsigned int env=0; env<options->num_environments; env++) {
		cout << endl << "Environment " << env << endl << "\tBlocks: ";
		for (unsigned int block=0; block<options->num_rule_blocks; block++) {
			cout << block << "*" << structures->nb[env*options->num_rule_blocks+block] << ", ";
		}
	}
	cout << endl;
}



/* DEPRECATED */

unsigned int execution_par_2 (PDP_Psystem_SAB::Structures structures, Options options) {
	//unsigned int esize=options->num_environments;
    unsigned int bsize=options->num_simulations*options->num_environments;
    //unsigned int osize=options->num_objects*bsize;
	const gsl_rng_type * T_random;
	gsl_rng ** r_variable = new gsl_rng*[bsize];

	/* create a generator chosen by the 
	* environment variable GSL_RNG_TYPE */
	gsl_rng_env_setup();
	T_random = gsl_rng_default;
	for (unsigned int i=0; i<bsize; i++)
		r_variable[i] = gsl_rng_alloc (T_random);

	omp_set_num_threads(bsize);
	cout << "tengo " << omp_get_num_procs() << " procesadores, y puedo poner " << omp_get_max_threads() << " hilos como máximo" << endl;
	unsigned int nt=0;

	unsigned int * app=new unsigned int[structures->rule_size*bsize];

	for (int block=0; block<options->num_rule_blocks; block++) {
		
		int rule_ini=structures->ruleblock.rule_ini[block];
		int rule_length=structures->ruleblock.rule_length[block];

//		#pragma omp parallel for shared(bsize,structures,nt,rule_ini,rule_length)
		for (unsigned int simenv=0; simenv<bsize; simenv++){
			unsigned int N=structures->nr[block*bsize+simenv];
			float cr=0.0,d=1.0;

			
			for (int r = rule_ini; r < rule_length+rule_ini; r++) {
				cr = structures->probability[r*bsize+simenv] / d;
				if (cr > 0.0) {
					if (r == rule_length+rule_ini-1)
						app[r*bsize+simenv] = N;
					else {
						app[r*bsize+simenv] = gsl_ran_binomial (r_variable[simenv], (double) cr, N);
						cout << "Binomial " << N << ", " << (double) cr << " = " << app[r*bsize+simenv]<<endl;
					}
				}
				else
				{
					app[r*bsize+simenv] = 0;
				}

				N-=app[r*bsize+simenv];
				d*=1-cr;
			}


			nt=omp_get_num_threads();
		}

		cout << "Bloque " << block << ":" << endl;
		for (unsigned int simenv=0; simenv<bsize; simenv++){
			cout << "Caso " << simenv << ":" << endl;
			cout << "\t N=" << structures->nr[block*bsize+simenv] << endl;
			cout << "\t Num reglas=" << rule_length << endl;
			for (int r = rule_ini; r < rule_length+rule_ini; r++) {	
				cout << "\t\t Regla " << r-rule_ini << ": p=" << structures->probability[r*bsize+simenv] << ", n=" << app[r*bsize+simenv] << endl; 
			}
		}
	}

	

	cout << "Numero de hilos ejecutados " << nt << endl;

	for (unsigned int i=0; i<bsize; i++)
		gsl_rng_free (r_variable[i]);
	delete [] r_variable;
	delete [] app;

	return 0;
}
