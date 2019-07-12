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
  Definition file for the fields of the binary file
  This file defines BINARY data type
  I.e, bits, bytes, bit masks, etc. 
*/

/**********************************************/
/* Types */
#ifndef BINBIT_H_
#define BINBIT_H_

#include <string>
#include <sstream>
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

typedef unsigned int MembraneID;
typedef unsigned short int LabelID;
typedef unsigned int ObjectID;
typedef unsigned int RuleID;

/**********************************************/
/* Values */

#define HEADER 0xAF12FA00
#define HEADER_BITMASK 0xFFFFFF00
#define HEADER_VERSION_BITMASK 0x0000000F
#define HEADER_VARIANT_BITMASK 0x000000F0
#define END_OF_TEXT '\0'

/**********************************************/
/* Sizes */

// We can just support up to 4 bytes for size fields
#define MAX_FIELD_SIZE 4
#define MAX_TEXT_SIZE 256

/*   About header */
#define HEADER_SIZE 4

/*   About sub-header */
#define SUBHEADER_SIZE 3
#define BIT_ACCURACY_MASK_SIZE 2
#define LISTING_CHAR_STRINGS_SIZE 1

/* About rule blocks */
#define BLOCK_INF_BYTE_SIZE 1
#define CHARGE_SIZE 1
#define PROB_SIZE 2
#define PROB_PREC 10000.0f

/*   About the multiset of objects */
#define NUMBER_INITIAL_MULTISETS_SIZE 2
#define NUMBER_DIFERENT_OBJECTS_SIZE 2
#define OBJECT_ID_SIZE_LONG 2
#define OBJECT_ID_SIZE_SHORT 1
#define OBJECT_MULTIPLICITY_SIZE 2

/*   About membranes */
#define NUMBER_LABELS_SIZE 2
#define NUMBER_MEMBRANES_SIZE 2
#define MEMBRANE_ID_SIZE_LONG 2
#define MEMBRANE_ID_SIZE_SHORT 1
#define MEMBRANE_ID_LABEL_SIZE_LONG 2
#define MEMBRANE_ID_LABEL_SIZE_SHORT 1
#define MEMBRANE_CHARGE_SIZE 1
#define CHARGE_BIT_MASK 0x03

#define NUMBER_RULES_SIZE 2

/* Auxiliary functions */

unsigned int read_bytes (ifstream& is, int size);

bool write_bytes (ifstream& is, char buffer[], int buffer_size);

int read_text (ifstream& is, char text_buffer[], int text_buffer_size, char end_text);

int check_file_error (bool condition, string message, ifstream& is);

#endif /* BINBIT_H_ */
