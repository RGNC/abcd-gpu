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


#include "binbit.h"

/********************************************/
/* AUXILIAR FUNCTIONS */

/* Auxiliary function:
    Read 1 to 4 bytes and returns them into a integer 
    (rather than a char buffer for bitmasks */
unsigned int read_bytes(ifstream& is, int size) {
    if (size>MAX_FIELD_SIZE || size <= 0) {
	cerr << "Error while reading file: In read_bytes, cannot read "<<size<<" bytes, max supported number is "<<MAX_FIELD_SIZE<<endl;
	exit(1);
    }
	
    unsigned int out = 0x0;
    char buffer[MAX_FIELD_SIZE];
    for (int i = 0; i < MAX_FIELD_SIZE; i++) buffer[i] = 0;

    is.read(buffer, size);
    // check for I/O errors
    if (is.fail() || is.eof()) {
	is.close();
	if ((is.rdstate() & ifstream::eofbit) != 0)
	    cerr << "Error while reading file: In read_bytes, cannot read bytes because reached an unexpected end of file." << endl;
	else if ((is.rdstate() & ifstream::failbit) != 0)
	    cerr << "Error while reading file: In read_bytes, cannot read bytes because of an internal error." << endl;
	else if ((is.rdstate() & ifstream::badbit) != 0)
	    cerr << "Error while reading file: In read_bytes, cannot read bytes because of an internal I/O error." << endl;

	// TODO: Change this feature for something more flexible
	exit(1);
    }


    for (int i = 0; i < size; i++) {
	out = out | ((unsigned char) buffer[i]) << ((size - i - 1)*8);
    }

    //is >> out;
    return out;
}

/* Auxiliary function:
    Read a text ended on end_text and returns the size of it */
int read_text(ifstream& is, char text_buffer[], int text_buffer_size = 256, char end_text = '\0') {
    char c = 'p';
    int i = 0;
    for (i = 0; (i < text_buffer_size) && (c != end_text); i++) {
	is >> c;
	// check for I/O errors
	if (is.fail() || is.eof()) {
	    is.close();
	    if ((is.rdstate() & ifstream::eofbit) != 0)
		cerr << "Error while reading file: In read_text, cannot read char because reached an unexpected end of file." << endl;
	    else if ((is.rdstate() & ifstream::failbit) != 0)
		cerr << "Error while reading file: In read_text, cannot read char because of an internal error." << endl;
	    else if ((is.rdstate() & ifstream::badbit) != 0)
		cerr << "Error while reading file: In read_text, cannot read char because of an internal I/O error." << endl;

	    exit(1);
	}

	if (i == text_buffer_size - 1)
	    text_buffer[i] = end_text;
	else
	    text_buffer[i] = c;
    }

    return i - 1;
}

/* Auxiliary function:
    If condition is true, there is an error: close the binary file, print the message and returns true. */
int check_file_error(bool condition, string message, ifstream& is) {
    if (condition) {
	is.close();
	cerr << "Error while reading file: " << message << endl;
	exit(1); // TODO: handle the error in other way
	return 1;
    } else {
	return 0;
    }
}



