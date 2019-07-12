/*
    ABCD-GPU: Simulating Population Dynamics P systems on the GPU, by DCBA
    ABCD-GPU is a subproject of PMCGPU (Parallel simulators for Membrane
                                        Computing on the GPU)

    Copyright (c) 2018  Research Group on Natural Computing, Universidad de Sevilla
    					Dpto. Ciencias de la Computación e Inteligencia Artificial
    					Escuela Técnica Superior de Ingeniería Informática,
    					Avda. Reina Mercedes s/n, 41012 Sevilla (Spain)

    Author: Andrés Doncel Ramírez

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

#ifndef COMPETITION_H_
#define COMPETITION_H_
namespace competition{

void print_header();
void print_partition(int* partition, int* alphabet,int num_rules,int num_objects);
void print_comparing_partition(int* partition, int* alphabet,int* partition2, int* alphabet2,int num_rules,int num_objects);
void print_rules(unsigned int* rules_size,unsigned int* lhs_object,int num_rules,int num_objects);
void reset_partition(int* partition,int* alphabet,int num_rules,int num_objects);
int initialize_rules(int *data, int size);
void initialize_lhs(int *data, int size);

//Old version
void make_partition(int* partition,unsigned int* rules_size,unsigned int*lhs_object,int * alphabet,int num_rules,int num_objects);
//New version, takes membranes into account
void make_partition(int* partition,unsigned int* rules_size,unsigned int*lhs_object,int * alphabet,int num_rules,int num_objects,unsigned int *membrane, unsigned int* mmultiplicity);

void make_partition_gpu(int* partition,unsigned int* rules_size,unsigned int*lhs_object,int * alphabet,int num_rules,int num_objects,int num_membranes ,unsigned int* mmultiplicity,int mult_size);

/*Identical version to the GPU, except sequential
 *
 */
void make_partition_2(int* partition,
		unsigned int* rules_size,
		unsigned int* lhs_object,
		int * alphabet,
		int num_rules,
		int num_objects,
		int num_membranes,
		unsigned int* mmultiplicity,
		int mult_size);

//New version, takes membranes into account
bool check_compete(int block_a,int block_b,unsigned int* rules_size,unsigned int * lhs_object,unsigned int *membrane, unsigned int* mmultiplicity);
//Old version
bool check_compete(int block_a,int block_b,unsigned int* rules_size,unsigned int * lhs_object);


void compare_partition(int* partition, int* alphabet,int* partition2, int* alphabet2,int num_rules,int num_objects);
int normalize_partition(int* partition, int* trans_partition,int size);
int initialize_partition_structures(int* partition,
		int num_partitions,
		int num_rules,
		unsigned int** accum_offsets,
		unsigned int** part_indexes,
		unsigned int** ordered,
		unsigned int* compacted_blocks,
		unsigned int* large_blocks,
		int cu_threads);

int reorder_ruleblocks(PDP_Psystem_REDIX::Structures structures,
		unsigned int* ordered,
		Options options);
}

#endif /* COMPETITION_H_ */
