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

#include "pdp_psystem.h"
// Add here new input modules
#include "pdp_psystem_source_random.h"
#include "pdp_psystem_source_binary.h"
#include "pdp_psystem_source_pl5.h"
// Add here new output modules
#include "pdp_psystem_output_binary.h"
#include "pdp_psystem_output_csv.h"
// Add here a new simulator engine
#include "simulator.h"
#include "simulator_seq_table.h"
#include "simulator_seq_dir.h"
#include "simulator_omp_dir.h"
#include "simulator_gpu_dir.h"
#include "simulator_omp_redir.h"
// Standard libraries
#include <iostream>
#include <time.h>
#include <string>
#include <timestat.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
#define DISABLED_OUTPUT -1
#define CSV_OUTPUT 0
#define BINARY_OUTPUT 1
#define BINARY_INPUT 0
#define PLINGUA5_INPUT 1

int main (int argc, char* argv[]) {
	//Structures* structures = new Structures;
	Options options = new struct _options;
	char c='\0';
	int mode=0;
	int par=0;
	int example=0;
	bool accu=true;
	bool random_system=true;
	int outputtype=DISABLED_OUTPUT;//CSV_OUTPUT;
	int input_format = BINARY_INPUT;
	string input_file="small.bin";
	string filter_file="";
	/* Default */
	options->num_rule_blocks = 384;
	options->num_blocks_env = 128;
	options->max_num_rules = 3;
	options->num_simulations = 3;//10;
	options->num_parallel_simulations = 1;
	options->num_environments = 4;
	options->num_objects = 7024;
	options->num_membranes=3;
	options->max_lhs=4;
	options->max_rhs=4;
	options->time=1;
	options->cycles=1;
	options->accuracy=2;
	options->verbose=0;
	options->debug = false;
	options->GPU_filter=false;
	options->output_filter=NULL;
	options->fast=false;
	options->error_cycle=1;
	options->micro=false;
	options->modular=false;
	options->modules = 1;

	options->objects_init_config=1.f;
	//By default, errors are brought each the cycle of steps, along with the configuration data
	bool error_with_common_cycle=true;
	// TODO: use getopt_long function to support arguments of type "--blocks"
	while ((c = getopt (argc, argv, "RO:Q:B:P:L:E:C:X:i:f:s:t:c:o:w:a:e:DGI:M:NFv:h")) != -1) {
		switch (c) {
		/* For randomly generated system */
		case 'R':
			random_system=true;
			break;
		case 'O':
			options->num_objects = atoi(optarg);
			break;
		case 'Q':
			options->num_membranes = atoi(optarg);
			break;
		case 'B':
			options->num_rule_blocks = atoi(optarg);
			options->num_blocks_env = options->num_rule_blocks/4;
			break;		
		case 'P':
			options->max_num_rules = atoi(optarg);
			break;
		case 'L':
			options->max_lhs=options->max_rhs=atoi(optarg);
			break;
		case 'E':
			options->num_environments = atoi(optarg);
			break;		
		case 'C':
			options->objects_init_config=atof(optarg);
			if(options->objects_init_config>1.f){
				options->objects_init_config=1.f;
			}else if(options->objects_init_config<0.f){
				options->objects_init_config=0.f;
			}
			break;		
		case 'X': /* Preconfigured random example */
			example = atoi(optarg);
			break;
		/* Options for the desired simulation */
		case 'i':
			random_system=false;
			input_file = optarg;
			break;
		case 'f':
			input_format = atoi(optarg);
			break;
		case 's':
			options->num_simulations = atoi(optarg);
			break;	
		case 't':
			options->time = atoi(optarg);
			break;
		case 'c':
			options->cycles = atoi(optarg);
			break;
		case 'o':
			outputtype=atoi(optarg);
			break;
		case 'w':
			filter_file = optarg;
			break;
		case 'a':
			options->accuracy = atoi(optarg);
			break;
		case 'e':
			options->error_cycle=atoi(optarg);
			error_with_common_cycle=false;
			break;	
		case 'D':
			options->micro=true;
			break;	
		case 'G':
			options->fast=true;
			break;						
		case 'I':
			mode=atoi(optarg);
			break;
			/*if (mode >= 0 && mode <= 5)
				break;
			cout << "Incorrect mode: " << mode << endl;*/
		case 'M':
			par=atoi(optarg);
			break;
		case 'N':
			options->modular=true;
			break;
		case 'F':
			accu=false;
			break;
		case 'v':
			options->verbose = atoi(optarg);
			break;
		case 'h':
		case '?':
			cout << "Copyright (C) 2019, RGNC, University of Seville" << endl <<
				"This program comes with ABSOLUTELY NO WARRANTY. This is free software, and you are welcome to redistribute it under certain conditions" << endl << endl;
			cout << "A simulator for Population Dynamics P systems using DCBA. Version 1.1 beta" << endl<<endl;
			cout << "Usage: ./abcdgpu <options>" <<endl<<
					"-------------" << endl <<
				    "Options for the simulation (check readme.txt for more help):"<<endl<<
					" -i: specify path to input file" << endl <<
					" -f: specify format of the input file" << endl << 
					"     -> 0 for binary format (revision 16-09-2014 supported) (default)" << endl <<
					"     -> 1 for binary from P-Lingua 5" << endl <<
					" -s: specify number of simulations"<< endl <<					
					" -t: specify time steps" << endl <<
					" -c: specify steps per cycle of model" << endl <<					
					" -o: specify output format type" << endl <<
					"     -> 0 for csv (comma-separated values) file (default)" << endl <<
					"     -> 1 for binary file (experimental, not working yet)" << endl <<
					" -w: specify filter output file (csv output only)" << endl <<									
					" -a: specify accuracy in the algorithms" << endl <<
					" -e: specify steps per cycle of error data retrieval. If not specified, defaults to steps per cycle of model." << endl <<
					" -N: flag to enable modular DCBA (only compatible with PLingua 5 binary files, when defining modules). Experimental feature" << endl <<
					//" -D: flag to enable micro-DCBA version for GPU. Experimental feature" << endl << // this feature is now broken. TODO: fix it
					" -G: flag for fast, less accurate RNG initialization (GPU-only)" << endl <<						
					" -I: specify selection of implementation: " << endl <<
					"     -> 0 for OpenMP simulator (default). For a multicore parallel execution, execute 'export OMP_NUM_THREADS=X' on the terminal (X should be the number of processors - 1 for best performance)," << endl <<
					"     -> 1 for GPU simulator. It requires to have a CUDA capable GPU on the system, with compute capability 3.5 or later." << endl <<
					"     -> (abandoned simulators, might be broken): 10 for table based simulator, 11 for sequential simulator, 12 for parallel OpenMP simulator" << endl <<
					" -M: specify selection of an execution mode:" << endl <<
					"     -> if I=1, M sets the behaviour of the GPU simulator: 0 runs GPU simulator (default), 1 (profiling) runs also the CPU gold code, 2 (profiling) runs with CPU and phase2 basic kernel" << endl <<
					"     -> if I=12, M is the parallelism level: 0 for environments, 1 for simulations, 2 for hybrid-2s and 3 for hybrid-2e" << endl <<
					" -F: flag to unset accurate mode (demotes to float row additions)" << endl <<
					" -v: specify verbosity level (from 0 to 6), as follows:"<< endl<<
					"     -> 0 (default): just show execution time and required simulation memory" << endl <<
					"     -> 1: verbosity 0 + simulator stage" << endl <<
					"     -> 2: verbosity 1 + simulations and transitons steps process" << endl <<
					"     -> 3: verbosity 2 + DCBA phases per transition step" << endl <<
					"     -> 4: verbosity 3 + print PDP system configuration after each transition step" << endl <<
					"     -> 5: verbosity 4 + print blocks and rules selection after each DCBA phase" << endl <<
					"     -> 6: verbosity 5 + print PDP system configuration after each DCBA phase" << endl <<
					"-------------" << endl <<
					"Special options for the random system generator:" << endl <<
					" -R: flag to activate the random system generator (by default, in case an input file is not provided)" << endl <<
					" -O: specify number of objects" << endl <<
					" -Q: specify number of membranes" << endl <<
					" -B: specify number of rule blocks" << endl <<
					" -P: specify maximum number of rules per block" << endl <<
					" -L: specify maximum number of objects in the LHS/RHS" << endl <<
					" -E: specify number of environments" << endl <<
					" -C: specify proportion of objects appearing in the initial configuration (between 0.0 and 1.0)" << endl <<
					" -X: specify to select one of the prefixed examples: 0 (default: configured through parameters), 1 (small), 2 (medium), 3 (large)" << endl <<
					"-------------" << endl <<
					"Finally, the simulator always outputs the total simulation time in ms and the total memory required in Bytes." << endl;
			return 0;
		}
	}
	if(error_with_common_cycle){
		options->error_cycle=options->cycles;
	}
	if (mode==20) {
		cout << "Glad to see you master, here you have your sample binary file..." << endl;
		PDP_Psystem_source_binary* source=new PDP_Psystem_source_binary(options);
		source->write_test_binary();
		delete source;
		cout << "Wrote the binary file..."<<endl;
		return 0;
	}	

	PDP_Psystem_source* source=NULL;
	PDP_Psystem_output* output=NULL;
	PDP_Psystem* PDPps=NULL;
	Simulator *simulator=NULL;
	//Binary output not yet supported, disabling...
	if(outputtype!=CSV_OUTPUT && filter_file!=""){
		cout<<"Warning: binary output filter is not supported. Disabling filtering..." <<endl;
		filter_file="";

	}
	/* Read the input */	
	if (!random_system) {
		if (input_format == PLINGUA5_INPUT)
			source=new PDP_Psystem_source_plingua5(input_file.c_str(),options,filter_file);
		else
	    	source=new PDP_Psystem_source_binary(input_file.c_str(),options,filter_file);
	}
	else {
		// Never output in a random system
		options->cycles=options->time+2;
		input_file="";

		// Pre-configured random systems
	    if (example == 3) {
		    options->num_rule_blocks = 100000;
		    options->num_blocks_env = options->num_rule_blocks/4;
		    options->max_num_rules = 5;
		    options->num_simulations = 50;
		    options->num_parallel_simulations = 1;
		    options->num_parallel_environments = 1;
		    options->num_environments= 15;
		    options->num_objects=7000;
		    options->num_membranes=4;
		    options->max_lhs=5;
		    options->max_rhs=4;
		    options->time=1;
		    options->cycles=1;
		    options->accuracy=1;
		    //options->verbose=1;
	    }
	    else if (example == 2) {
		    options->num_rule_blocks = 384;
		    options->num_blocks_env = 128;
		    options->max_num_rules = 3;
		    options->num_simulations = 3;//10;
		    options->num_parallel_simulations = 1;
		    options->num_environments = 4;
		    options->num_objects = 1024;
		    options->num_membranes=3;
		    options->max_lhs=4;
		    options->max_rhs=4;
		    options->time=1;
		    options->cycles=1;
		    options->accuracy=2;
		    //options->verbose=1;
	    }
	    else if (example == 1) {
		    options->num_rule_blocks = 4;
		    options->num_blocks_env = options->num_rule_blocks/4;
		    options->max_num_rules = 3;
		    options->num_simulations = 1;
		    options->num_parallel_simulations = 1;
		    options->num_environments= 2;
		    options->num_objects=6;
		    options->num_membranes=2;
		    options->max_lhs=5;
		    options->max_rhs=4;
		    options->time=1;
		    options->cycles=1;
		    options->accuracy=1;
		    //options->verbose=1;
	    }
	    
	    source=new PDP_Psystem_source_random(options);
	}


	/* Initiate the simulator */
	if (mode==0) { // CPU simulator
		PDPps = new PDP_Psystem_REDIX(source);

		if (outputtype==CSV_OUTPUT)
			output = new PDP_Psystem_output_csv(input_file.c_str(),options);
		else if (outputtype==BINARY_OUTPUT)
			output = new PDP_Psystem_output_binary(input_file.c_str(),options);

		simulator = new Simulator_omp_redir((PDP_Psystem_REDIX*)PDPps,options->num_simulations,accu,output);
	}
	else if (mode==1) { // GPU simulator
		PDPps = new PDP_Psystem_REDIX(source);

		if (outputtype==CSV_OUTPUT)
			output = new PDP_Psystem_output_csv(input_file.c_str(),options);
		else if (outputtype==BINARY_OUTPUT)
			output = new PDP_Psystem_output_binary(input_file.c_str(),options);

		simulator = new Simulator_gpu_dir((PDP_Psystem_REDIX*)PDPps,par,accu,output);
	}
	
	/* Deprecated simulators */
	else if (mode == 10) {
		/* Create the structures for the simulator */
		PDPps = new PDP_Psystem_SAB(source);
		simulator = new Simulator_seq_table((PDP_Psystem_SAB*)PDPps);
	}
	else if (mode==11) {
		PDPps = new PDP_Psystem_SAB(source);
		simulator = new Simulator_seq_dir((PDP_Psystem_SAB*)PDPps);
	}
	else if (mode==12) {
		PDPps = new PDP_Psystem_SAB(source);
		simulator = new Simulator_omp_dir((PDP_Psystem_SAB*)PDPps,par);
	}	
	

	delete source;

	start_timer();
	
	/* Run the main loop of the simulator */
	if (simulator && simulator->run()){
		if (options->verbose>0) cout << endl << "Successfully finished." << endl;
	}
	else {
		if (!simulator)
			cout << "No simulator associated to this mode, use -h option" << endl;
		else
			cout << "Problems encountered in the simulator. Halted with errors" << endl;
	}

	double d=end_timer();
	cout << d << " " << options->mem << endl;
	
	delete PDPps;

	delete output;

	delete simulator;

	return 0;

}
