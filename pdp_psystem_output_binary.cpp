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


#include "pdp_psystem_output_binary.h"
#include <string.h>

#define CH_IDX(m) (sim*(options->num_environments*options->num_membranes)+e*options->num_membranes+(m))
#define MU_IDX(o,m) (sim*options->num_environments*esize+e*esize+(m)*msize+(o))
#define MAX(a,b) ((a)>(b))?(a):(b)


PDP_Psystem_output_binary::PDP_Psystem_output_binary(const char * file,Options opt) {
	if (file==NULL || strlen(file)==0) {
		donotoutput=true;
		return;
	}
	else donotoutput=false;

	options=opt;
	myword=(unsigned int*) buffer;
	myhalfword=(unsigned short int*) buffer;
	myhalfword2=(unsigned short int*) buffer+2;

	object4= new unsigned int [options->num_objects];
	multip4= new unsigned int [options->num_objects];

	pobjects = get_precision(options->num_objects);

	string instr=file;
	string outstr=instr.insert(instr.length()-4,"_output");

	/*if (file != NULL) {
		this->file = new char[strlen(file+1)];
		strcpy(this->file,file);
	}
	else {
		this->file = new char [8];
		this->file[0]='o';this->file[1]='u';this->file[2]='t';this->file[3]='.';this->file[4]='b';this->file[5]='i';this->file[6]='n';this->file[7]='\0';
	}*/

	outfile.open(outstr.c_str(),ofstream::binary |  ios::out);
}


bool PDP_Psystem_output_binary::write_configuration(unsigned int* config_multisets, char * config_charges, int sim, int step, char** objstrings) {

	if (config_multisets==NULL || config_charges==NULL) return false;
	if (donotoutput) return true;

	int esize=options->num_objects*options->num_membranes;
	int msize=options->num_objects;

	// Header of the chunk
	*myhalfword=sim;
	*myhalfword2=step;

	outfile.write(buffer,4);

	// First loop to extract
	for (int e=0;e<options->num_environments;e++) {
		for (int m=0;m<options->num_membranes;m++) {
			int maxprec=0;
			unsigned int next=0;

			for (unsigned int o=0; o<options->num_objects;o++) {
				unsigned int multip=config_multisets[MU_IDX(o,m)];
				if (multip>0) {
					maxprec=MAX(maxprec,get_precision(multip));
					object4[next]=o;
					multip4[next++]=multip;
				}
			}

			// Info Byte
			buffer[0]=0; // Reserved
			buffer[1]=config_charges[CH_IDX(m)]; // Charge
			buffer[2]=maxprec; // Max precision required for multiplicities
			buffer[3]=get_precision(next); // Precision for number of objects

			outfile.write(buffer,4);

			// Num objects
			int prec=(1<<buffer[3]);
			*myword=next;
			outfile.write(buffer+(4-prec),prec);

			int mulprec=(1<<maxprec);
			int objprec=(1<<pobjects);
			for (unsigned int o=0; o<next;o++) {
				//object ID
				outfile.write(((char *)(object4+o))+(4-objprec),objprec);

				//Multip
				outfile.write(((char *)(multip4+o))+(4-mulprec),mulprec);
			}
		}
	}

	outfile.flush();
	return true;
}
