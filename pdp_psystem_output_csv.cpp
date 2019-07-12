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
 * pdp_psystem_output_binary.cpp
 *
 *  Created on: 26/11/2014
 *      Author: miguel
 */


#include "pdp_psystem_output_csv.h"
#include <string.h>
#include <string>


#define CH_IDX(m) (sim*(options->num_environments*options->num_membranes)+e*options->num_membranes+(m))
#define MU_IDX(o,m) (sim*options->num_environments*esize+e*esize+(m)*msize+(o))
#define MAX(a,b) ((a)>(b))?(a):(b)


PDP_Psystem_output_csv::PDP_Psystem_output_csv(const char * file,Options opt) {
	if (file==NULL || strlen(file)==0) {
		donotoutput=true;
		return;
	}
	else donotoutput=false;

	options=opt;

	string instr=file;
	string outstr=instr.append("_output.csv");

	/*if (file != NULL) {
		this->file = new char[strlen(file+1)];
		strcpy(this->file,file);
	}
	else {
		this->file = new char [8];
		this->file[0]='o';this->file[1]='u';this->file[2]='t';this->file[3]='.';this->file[4]='b';this->file[5]='i';this->file[6]='n';this->file[7]='\0';
	}*/

	outfile.open(outstr.c_str(),ios::out);

	outfile << "SIMULATION, STEP, ENVIRONMENT, MEMBRANE, OBJECT, MULTIPLICITY" << endl;
}

bool PDP_Psystem_output_csv::write_configuration_filtered(unsigned int* filtered_multisets, char * config_charges, int sim, int step, char** objstrings) {
	if (filtered_multisets==NULL || config_charges==NULL) return false;
	if (donotoutput) return true;

	bool useobjstring = objstrings != NULL;


	int esize=options->num_objects*options->num_membranes;
	int msize=options->num_objects;

	if(options->output_filter!=NULL){

		//Filter, output only some things
		if(options->GPU_filter){

			//GPU filter, multiset only contains data which is already in order
			unsigned int sim_offset=sim*options->objects_to_output;
			for(int i=0;i<options->objects_to_output;i++){
				int obj_index=options->output_filter[i];
				//Object index
				unsigned int o= obj_index%options->num_objects;
				//Membrane index
				int m=((obj_index-o)/options->num_objects)%options->num_membranes;
				//Environment index
				int e=(((obj_index-o)/options->num_objects)-m)/options->num_membranes;

				unsigned int multip=filtered_multisets[i+sim_offset];

				if (multip>0) {

					outfile << sim << ","
							<< step << ","
							<< e  << ","
							<< m  << ",";
					if (useobjstring) {
						outfile << "\"" << objstrings[o] << "\"";
					}
					else
						outfile << o;
					outfile << "," << multip << endl;
				}

			}
		}
		else{
			//CPU standard filter, need to access each multiplicity
			for(int i=0;i<options->objects_to_output;i++){
				int obj_index=options->output_filter[i];
				//Object index
				unsigned int o= obj_index%options->num_objects;
				//Membrane index
				int m=((obj_index-o)/options->num_objects)%options->num_membranes;
				//Environment index
				int e=(((obj_index-o)/options->num_objects)-m)/options->num_membranes;

				unsigned int multip=filtered_multisets[MU_IDX(o,m)];
				if (multip>0) {
					outfile << sim << ","
							<< step << ","
							<< e  << ","
							<< m  << ",";
					if (useobjstring) {
						outfile << "\"" << objstrings[o] << "\"";
					}
					else
						outfile << o;
					outfile << "," << multip << endl;
				}

			}

		}


	}


	outfile.flush();
	return true;
}

bool PDP_Psystem_output_csv::write_configuration(unsigned int* config_multisets, char * config_charges, int sim, int step, char** objstrings) {

	if (config_multisets==NULL || config_charges==NULL) return false;
	if (donotoutput) return true;

	bool useobjstring = objstrings != NULL;


	int esize=options->num_objects*options->num_membranes;
	int msize=options->num_objects;

	// No filter, get all data
	// First loop to extract
	for (int e=0;e<options->num_environments;e++) {
		for (int m=0;m<options->num_membranes;m++) {
			for (unsigned int o=0; o<options->num_objects;o++) {

				unsigned int multip=config_multisets[MU_IDX(o,m)];
				if (multip>0) {
					outfile << sim << ","
							<< step << ","
							<< e  << ","
							<< m  << ",";
					if (useobjstring) {
						outfile << "\"" << objstrings[o] << "\"";
					}
					else
						outfile << o;
					outfile << "," << multip << endl;
				}
			}
		}
	}



	outfile.flush();
	return true;
}
