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
    along with ABCD-GPU.  If not, see <http://www.gnu.org/licenses/>.

 ============================================================================
 Name        : competition.cu
 Author      : AndresDR
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include "pdp_psystem_redix.h"
#include "competition.h"
#include <stdio.h>


namespace competition{

__global__ void  make_partition_phase_0_kernel_3(
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int ALPHABET,
		int total_lhs);

__global__ void make_partition_phase_1_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET,
		int num_membranes);
__global__ void make_partition_phase_1_5_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET);
__global__ void make_partition_phase_2_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET,
		int num_membranes);
__global__ void get_partition_kernel(int* partition,unsigned int* rules_size,
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET);

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


__device__ bool change=true;
/**
 * https://pdfs.semanticscholar.org/4569/2750cb1bf50da6451e70ae06b6519992e4ec.pdf
 * CUDA kernel that computes partition for overlapping rules
 */
__global__ void make_partition_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int* lhs_object,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET,
		int num_membranes,
		unsigned int* mmultiplicity) {
	static const int BLOCK_SIZE = 256;
	const int blockCount1 = (NUM_RULES+BLOCK_SIZE-1)/BLOCK_SIZE;
	const int blockCount2 = ((ALPHABET*num_membranes)+BLOCK_SIZE-1)/BLOCK_SIZE;
	change=true;
	bool exit_loop=false;


	while(change || !exit_loop){

		//If nothing changed, we must give a last pass
		exit_loop=!change;
		change=false;
		make_partition_phase_1_kernel_3<<<blockCount1,BLOCK_SIZE>>>( partition, rules_size, lhs_object,mmultiplicity, alphabet, NUM_RULES, ALPHABET, num_membranes);
		make_partition_phase_1_5_kernel_3<<<blockCount1,BLOCK_SIZE>>>( partition, rules_size, lhs_object,mmultiplicity, alphabet, NUM_RULES, ALPHABET);
		make_partition_phase_2_kernel_3<<<blockCount2,BLOCK_SIZE>>>( partition,  rules_size, lhs_object, alphabet, NUM_RULES, ALPHABET, num_membranes);

		cudaDeviceSynchronize();
//		printf("iter:\n");
//		for(int i=0;i<ALPHABET;i++)
//		{
//			printf("%3u, ",i);
//
//		}
//		printf("\n");
//		for(int i=0;i<ALPHABET;i++)
//		{
//			printf("%3u, ",alphabet[i]);
//		}
//		printf("\n");
	}


}
__global__ void make_partition_phase_0_kernel_3(
				unsigned int*lhs_object,
				unsigned int* mmultiplicity,
				int ALPHABET,
				int total_lhs){
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx < total_lhs){
		lhs_object[idx]=lhs_object[idx]+
                GET_MEMBR(mmultiplicity[idx])*ALPHABET;
	}
}

__global__ void make_partition_phase_1_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET,
		int num_membranes) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx < NUM_RULES){

		unsigned rule_id_begin=rules_size[idx];
		unsigned rule_id_end=rules_size[idx+1];
		int val=ALPHABET*num_membranes;

		for (unsigned int k=rule_id_begin; k<rule_id_end; k++){

			val=min(val,alphabet[lhs_object[k]]);
		}

		partition[idx]=val;

	}
}
__global__ void make_partition_phase_1_5_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int* lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx < NUM_RULES){

		unsigned rule_id_begin=rules_size[idx];
		unsigned rule_id_end=rules_size[idx+1];

		unsigned val=partition[idx];

		for (unsigned int j=rule_id_begin; j<rule_id_end; j++){
			int old_val=atomicMin((alphabet+lhs_object[j]),val);
			if(old_val!=val)
				change=true;
		}


	}
}

//Pointer jumping
__global__ void make_partition_phase_2_kernel_3(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET,
		int num_membranes) {

	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < ALPHABET*num_membranes){
		int i =alphabet[idx];
		int i_1 =alphabet[i];
		if(i_1!=i){
			//We must do another loop
			change=true;
		}
		while(i_1!=i)
		{
			i=i_1;
			i_1=alphabet[i];
		}
		atomicMin(alphabet+idx,i_1);

	}
}


__global__ void get_partition_kernel(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < NUM_RULES){

		partition[idx]=alphabet[lhs_object[rules_size[idx]]];
		//printf("set rule %i %i with object %u and partition %i \n",idx,rules_size[idx],lhs_object[rules_size[idx]],partition[idx]);

	}
}


/**
 * Host function that copies the data and launches the work on GPU
 */
void make_partition_gpu(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		int * alphabet,
		int num_rules,
		int num_objects,
		int num_membranes,
		unsigned int* mmultiplicity,
		int mult_size){
	int * d_partition;
	unsigned int * d_rules_size;
	unsigned int * d_lhs_object;

	unsigned int * d_mmultiplicity;
	int * d_alphabet;
	int total_lhs=rules_size[num_rules];


	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_partition, sizeof(int)*num_rules));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_rules_size,sizeof(int)*(num_rules+1)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_lhs_object, sizeof(unsigned int)*total_lhs));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_alphabet, sizeof(int)*num_objects*num_membranes));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_mmultiplicity, sizeof(MULTIPLICITY)*mult_size));

	CUDA_CHECK_RETURN(cudaMemcpy(d_partition, partition, sizeof(int)*num_rules, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_rules_size, rules_size, sizeof(int)*(num_rules+1), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_lhs_object, lhs_object,sizeof(unsigned int)*total_lhs, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_alphabet, alphabet, sizeof(int)*num_objects*num_membranes, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_mmultiplicity, mmultiplicity, sizeof(MULTIPLICITY)*mult_size, cudaMemcpyHostToDevice));

	static const int BLOCK_SIZE = 256;
	const int blockCount = (total_lhs+BLOCK_SIZE-1)/BLOCK_SIZE;
	const int blockCount1 = (num_rules+BLOCK_SIZE-1)/BLOCK_SIZE;


	make_partition_phase_0_kernel_3<<<blockCount,BLOCK_SIZE>>>(
			d_lhs_object,
			d_mmultiplicity,
			num_objects,
			total_lhs);
    make_partition_kernel_3<<<1,1>>>(d_partition,
    		d_rules_size,
    		d_lhs_object,
    		d_alphabet,
    		num_rules,
    		num_objects,
    		num_membranes,
    		d_mmultiplicity);
	get_partition_kernel<<<blockCount1, BLOCK_SIZE>>> (
			d_partition,
			d_rules_size,
			d_lhs_object,
			d_mmultiplicity,
			d_alphabet,
			num_rules,
			num_objects);

    cudaDeviceSynchronize();


	CUDA_CHECK_RETURN(cudaMemcpy(partition, d_partition, sizeof(int)*num_rules, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(alphabet, d_alphabet, sizeof(int)*num_objects*num_membranes, cudaMemcpyDeviceToHost));


	CUDA_CHECK_RETURN(cudaFree(d_partition));
	CUDA_CHECK_RETURN(cudaFree(d_rules_size));
	CUDA_CHECK_RETURN(cudaFree(d_lhs_object));
	CUDA_CHECK_RETURN(cudaFree(d_alphabet));
	CUDA_CHECK_RETURN(cudaFree(d_mmultiplicity));
}

int initialize_rules(int *data, int size,int max_objects_per_rule)
{
	data[0]=0;
	for (int i = 1; i < size; ++i){

		data[i] =data[i-1]+(rand()%max_objects_per_rule)+1;
	}
	return data[size-1];
}
void initialize_lhs(int *data, int size,int num_rules,int num_objects)
{
	for (int i = 0; i < size; ++i){
		data[i] =rand()%num_objects;
	}

}

/**
 * Modified sequential version of partition kernel version 3
 * It works
 */
void make_partition(int* partition, unsigned int* rules_size,unsigned int*lhs_object, int * alphabet,int num_rules,int num_objects){
	bool change=true;
	bool exit_loop=false;

	while(change || !exit_loop){

		//If nothing changed, we must give a last pass
		exit_loop=!change;
		change=false;

		for(int i=0;i<num_rules;i++){
			for(int j=i+1;j<num_rules;j++){
				if(partition[j]!=partition[i] && check_compete(i,j,rules_size,lhs_object)){
					if(partition[j]<partition[i]){
						partition[i]=partition[j];
						change=true;
					}else{
						partition[j]=partition[i];
					}
				}
			}
		}
		for(int idx=0;idx<num_rules;idx++){
			int i =partition[idx];
			int i_1 =partition[i];
			bool must_go_on=false;
			while(i_1!=i)
			{
				must_go_on=true;
				i=i_1;
				i_1=partition[i];
			}
			partition[idx]=min(partition[idx],i_1);
			if(must_go_on){
				change=true;
			}
		}
		}
}

void make_partition_phase_0(
				unsigned int*lhs_object,
				unsigned int* mmultiplicity,
				int ALPHABET,
				int total_lhs){

	for(int idx=0;idx<total_lhs;idx++)
	{
		lhs_object[idx]+= GET_MEMBR(mmultiplicity[idx])*ALPHABET;
	}
}
void undo_make_partition_phase_0(
				unsigned int*lhs_object,
				unsigned int* mmultiplicity,
				int ALPHABET,
				int total_lhs){

	for(int idx=0;idx<total_lhs;idx++)
	{
		lhs_object[idx]-= GET_MEMBR(mmultiplicity[idx])*ALPHABET;
	}
}
void make_partition_phase_1(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET,
		int num_membranes) {

	for(int idx=0;idx<NUM_RULES;idx++){

		unsigned rule_id_begin=rules_size[idx];
		unsigned rule_id_end=rules_size[idx+1];
		int val=ALPHABET*num_membranes;

		for (unsigned int k=rule_id_begin; k<rule_id_end; k++){
			val=min(val,alphabet[lhs_object[k]]);
		}

		partition[idx]=val;

	}
}
bool make_partition_phase_1_5(int* partition,
		unsigned int* rules_size,
		unsigned int* lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET) {

	bool change=false;
	for(int idx=0;idx<NUM_RULES;idx++){

		unsigned rule_id_begin=rules_size[idx];
		unsigned rule_id_end=rules_size[idx+1];

		unsigned val=partition[idx];

		for (unsigned int j=rule_id_begin; j<rule_id_end; j++){
			int old_val=alphabet[lhs_object[j]];
			if(old_val!=val){
				change=true;
				alphabet[lhs_object[j]]=min(old_val,val);
			}
		}


	}
	return change;
}

//Pointer jumping
bool make_partition_phase_2(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET,
		int num_membranes) {
	bool change=false;

	for(int idx=0;idx<ALPHABET*num_membranes;idx++)
	{
		int i =alphabet[idx];
		int i_1 =alphabet[i];
		if(i_1!=i){
			//We must do another loop
			change=true;
		}
		while(i_1!=i)
		{
			i=i_1;
			i_1=alphabet[i];
		}
		alphabet[idx]=i_1;

	}
	return change;
}


void get_partition(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		unsigned int* mmultiplicity,
		int * alphabet,
		int NUM_RULES,
		int ALPHABET) {

	for(int idx=0;idx<NUM_RULES;idx++)
		{
		partition[idx]=alphabet[lhs_object[rules_size[idx]]];
		//printf("set rule %i %i with object %u and partition %i \n",idx,rules_size[idx],lhs_object[rules_size[idx]],partition[idx]);
	}
}


/**
 * Sequential version of partition kernel version 3
 * It works!!
 */
void make_partition_2(int* partition,
		unsigned int* rules_size,
		unsigned int* lhs_object,
		int * alphabet,
		int num_rules,
		int num_objects,
		int num_membranes,
		unsigned int* mmultiplicity,
		int mult_size){
	int total_lhs=rules_size[num_rules];

	make_partition_phase_0(
				lhs_object,
				mmultiplicity,
				num_objects,
				total_lhs);

	bool change=true;
	bool exit_loop=false;

	while(change || !exit_loop){
		//If nothing changed, we must give a last pass
		exit_loop=!change;
		change=false;

		make_partition_phase_1( partition, rules_size, lhs_object,mmultiplicity, alphabet, num_rules, num_objects, num_membranes);
		change=make_partition_phase_1_5( partition, rules_size, lhs_object,mmultiplicity, alphabet, num_rules, num_objects)||change;
		change=make_partition_phase_2( partition,  rules_size, lhs_object, alphabet, num_rules, num_objects, num_membranes)||change;
	}

	get_partition(
			partition,
			rules_size,
			lhs_object,
			mmultiplicity,
			alphabet,
			num_rules,
			num_objects);
	undo_make_partition_phase_0(
					lhs_object,
					mmultiplicity,
					num_objects,
					total_lhs);

}


/**
 * Modified sequential version of partition kernel version 3
 * It works, but too slow
 */
void make_partition(int* partition,
		unsigned int* rules_size,
		unsigned int*lhs_object,
		int * alphabet,
		int num_rules,
		int num_objects,
		unsigned int *membrane,
		unsigned int* mmultiplicity){
	bool change=true;
	bool exit_loop=false;

	while(change || !exit_loop){

		//If nothing changed, we must give a last pass
		exit_loop=!change;
		change=false;

		for(int i=0;i<num_rules;i++){
			for(int j=i+1;j<num_rules;j++){
				if(partition[j]!=partition[i] && check_compete(i,j,rules_size,lhs_object,membrane,mmultiplicity)){
					if(partition[j]<partition[i]){
						partition[i]=partition[j];
						change=true;
					}else{
						partition[j]=partition[i];
					}
				}
			}
		}
		for(int idx=0;idx<num_rules;idx++){
			int i =partition[idx];
			int i_1 =partition[i];
			bool must_go_on=false;
			while(i_1!=i)
			{
				must_go_on=true;
				i=i_1;
				i_1=partition[i];
			}
			partition[idx]=min(partition[idx],i_1);
			if(must_go_on){
				change=true;
			}
		}
		}
}



bool check_compete(int block_a,int block_b,unsigned int* rules_size,unsigned int * lhs_object){
	bool res=false;
	for (unsigned int j=rules_size[block_a]; j<rules_size[block_a+1]; j++){
		for (unsigned int k=rules_size[block_b]; k<rules_size[block_b+1]; k++){

			if(lhs_object[j]==lhs_object[k]){
				res=true;
				break;
			}
		}
	}
	return res;
}

bool check_compete(int block_a,int block_b,unsigned int* rules_size,unsigned int * lhs_object,unsigned int *membrane, unsigned int* mmultiplicity){
	bool res=false;
	for (unsigned int j=rules_size[block_a]; j<rules_size[block_a+1]; j++){
		for (unsigned int k=rules_size[block_b]; k<rules_size[block_b+1]; k++){

			//If they share an object in the same membrane
			if(lhs_object[j]==lhs_object[k]
		         &&GET_MEMBR(mmultiplicity[j])==GET_MEMBR(mmultiplicity[k])){

					// Also the blocks stand for different membranes OR
					// They stand for the same membrane and have the same charge
					//if(GET_MEMBRANE(membrane[block_a])!=GET_MEMBRANE(membrane[block_b])
					//		|| (GET_ALPHA(membrane[block_a])==GET_ALPHA(membrane[block_b]))){
						res=true;
						break;

					//}
			}
		}
	}
	return res;
}

void reset_partition(int* partition,int* alphabet,int num_rules,int num_objects) {
	for (int i = 0; i < num_objects; i++) {
		alphabet[i] = i;
	}
	//At most, they will be all independent
	for (int i = 0; i < num_rules; i++) {
		partition[i] = i;
	}
}

void print_header(int num_rules,int num_objects,int max_objects_per_rule){
	std::cout<< "--- " << num_rules <<" rules generated with at most "
			<< max_objects_per_rule<< " objects each and "
			<<num_objects <<" objects in alphabet" <<" ---"<< std::endl;

}
void print_rules(unsigned int* rules_size,unsigned int* lhs_object,int num_rules,int num_objects) {

	for (int i = 0; i < num_rules; i++) {
		std::cout << "Rule " << i << std::endl;
		for (int j = rules_size[i]; j < rules_size[i + 1]; j++) {
			std::cout << "\t Object " << lhs_object[j] << std::endl;
		}
	}
}

void print_partition( int* partition, int* alphabet,int num_rules,int num_objects) {


	for (int i = 0; i < num_rules; i++) {
		std::cout << "\t Rule " << i << " belongs to part " << partition[i]
				<< std::endl;
	}
//	for (int i = 0; i < ALPHABET; i++) {
//		std::cout << "\t Object " << i << " belongs to part " << alphabet[i]
//				<< std::endl;
//	}
}
void print_comparing_partition(int* partition, int* alphabet,int* partition2, int* alphabet2,int num_rules,int num_objects) {

	for (int i = 0; i < num_rules; i++) {
		std::cout << "\t Rule " << i << " belongs to part: " << partition[i]
		          << "\t | \t " << partition2[i]
				<< std::endl;
	}
//	for (int i = 0; i < ALPHABET; i++) {
//		std::cout << "\t Object " << i << " belongs to part " << alphabet[i]
//				<< "\t | \t " << alphabet2[i]
//				<< std::endl;
//	}
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

void compare_partition(int* partition, int* alphabet,int* partition2, int* alphabet2,int num_rules,int num_objects){

//	for(int i=0;i<ALPHABET;i++){
//		if(alphabet[i]!=alphabet2[i]){
//			std::cout<<"Alphabet not matching"<<std::endl;
//			break;
//		}
//	}

	for(int i=0;i<num_rules;i++){
		if(partition[i]!=partition2[i]){
			std::cout<<"Partition not matching"<<std::endl;
			return;
		}
	}

	std::cout<<"Works great!!!"<<std::endl;
}
int normalize_partition(int* partition, int* trans_partition,int size){

	for(int i=0;i<size;i++){
		trans_partition[i]=-1;
	}

	int part_index=0;
	for (int i=0;i<size;i++){
		for(int j=0;j<i;j++){
			if(partition[j]==partition[i]){
				trans_partition[i]=trans_partition[j];
				break;
			}
		}
		if(trans_partition[i]==-1){
			trans_partition[i]=part_index;
			part_index++;
		}
	}

	return part_index;
}
int initialize_partition_structures(int* partition,
		int num_partitions,
		int num_rules,
		uint** accum_offsets,
		uint** part_indexes,
		uint** ordered,
		uint* compacted_blocks,
		uint* large_blocks,
		int cu_threads){

	//Size of each partition
	uint* offsets=new uint[num_partitions];
	//To keep track of the index of each rule when ordering
	uint* current_offset=new uint[num_partitions];


	//Accumulated sizes of partitions
	*accum_offsets=new uint[num_partitions+1];

	int *expanded_accum_offsets=new int[num_partitions+1];

	//For each partition, if too large, where their rules will be moved to the beginning
	uint* real2ord=new uint[num_partitions];
	uint* ord2real=new uint[num_partitions];
	//Initialize
	for(int i=0;i<num_partitions;i++){
		offsets[i]=0;
		current_offset[i]=0;
		real2ord[i]=i;
		ord2real[i]=i;
	}
	//Parts with only one ruleblock
	int unique_blocks=num_partitions;

	//Index where the large blocks will be placed. After the loop, number of large blocks
	uint next_reorder=0;
	//Get offsets (partition sizes)
	for(int i=0;i<num_rules;i++){
		int part=partition[i];
		offsets[part]++;

		if(offsets[part]==2){
			unique_blocks--;
		}
		if(offsets[part]>=cu_threads
				&& real2ord[part]>=next_reorder){
			//Mark the size too big;

			int ord2real_small=ord2real[next_reorder];
			int real2ord_small=real2ord[ord2real_small];
			int real2ord_large=real2ord[part];
			int ord2real_large=ord2real[real2ord_large];

			ord2real[next_reorder]=ord2real_large;
			ord2real[real2ord_large]=ord2real_small;
			real2ord[ord2real_small]=real2ord_large;
			real2ord[part]=real2ord_small;

			next_reorder++;

		}
	}

	*part_indexes=new uint[num_partitions-unique_blocks-next_reorder+1];
	(*part_indexes)[0]=next_reorder;
	//Accumulates blocks into greater chunks
	int num_compact_blocks=0;
	int start_position=next_reorder;

	//Inclusive scan for sizes
	//Accum offsets only has the respective increments(e.g.: 0,3,5,8)
	(*accum_offsets)[0]=0;
	//Expanded also has intermediate results, if unique blocks between them (e.g.: 0,3,3,5,8,8,8)
	expanded_accum_offsets[0]=0;
	int compact_index=0;


	for(int i=0;i<num_partitions;i++){
		//Accumulated offsets (partition sizes)
		int val=0;
		if(offsets[ord2real[i]]!=1){
			//Dependent block
			val=offsets[ord2real[i]];
			(*accum_offsets)[compact_index+1]=(*accum_offsets)[compact_index]+val;
			compact_index++;

			if(compact_index>next_reorder){
				if(compact_index==num_partitions-unique_blocks||
						(*accum_offsets)[compact_index]-(*accum_offsets)[start_position]>=cu_threads){

					num_compact_blocks++;
					(*part_indexes)[num_compact_blocks]=compact_index;
					start_position=compact_index;
				}
			}

		}

		expanded_accum_offsets[i+1]=expanded_accum_offsets[i]+val;
	}
	*large_blocks=next_reorder;
	*compacted_blocks=num_compact_blocks;

	//Sort rules
	*ordered=new uint[num_rules];
	int independent_offset=num_rules-unique_blocks;
	for(int i=0;i<num_rules;i++){
		int part=partition[i];
		int ord_part=real2ord[part];
		if(offsets[part]==1){
			//Independent block, put it at the end
			(*ordered)[independent_offset]=i;
			independent_offset++;
		}else{
		//Put the rule on its corresponding position
		(*ordered)[expanded_accum_offsets[ord_part]+current_offset[ord_part]]=i;
		//Advance index for that partition one unit
		current_offset[ord_part]++;
		}
	}

//	std::cout<< num_partitions<<" "<<unique_blocks<< " " <<next_reorder <<std::endl;

	//Print for debugging purposes
//	std::cout<< "partitions" <<std::endl;
//	for(int i=0;i<num_rules;i++){
//		std::cout<<partition[i] <<std::endl;
//	}
//	std::cout<< "offset" <<std::endl;
//	for(int i=0;i<num_partitions;i++){
//		std::cout<<offsets[i] <<std::endl;
//	}
//	std::cout<< "accum" <<std::endl;
//	for(int i=0;i<num_partitions+1;i++){
//		std::cout<<(*accum_offsets)[i] <<std::endl;
//	}
//	std::cout<< "extended accum" <<std::endl;
//	for(int i=0;i<num_partitions+1;i++){
//		std::cout<<expanded_accum_offsets[i] <<std::endl;
//	}
//	std::cout<< "part_indexes" <<std::endl;
//	for(int i=0;i<num_compact_blocks+1;i++){
//		std::cout<<(*part_indexes)[i]<<": ";
//		std::cout<<(*accum_offsets)[(*part_indexes)[i]]<<", ";
//
//	}
//	std::cout<<std::endl;

//	std::cout<< "block chunks: "<<num_compact_blocks <<std::endl;
//	for(int i=1;i<num_compact_blocks;i++){
//		std::cout<<((*accum_offsets)[(*part_indexes)[i]])-((*accum_offsets)[(*part_indexes)[i-1]])<<", ";
//
//	}
//	std::cout<<std::endl;
//
//
//	std::cout<< "reordered sizes o2r" <<std::endl;
//	for(int i=0;i<num_partitions;i++){
//		std::cout<<ord2real[i]<<", ";
//	}
//	std::cout<<std::endl;
//	std::cout<< "reordered sizes r2o" <<std::endl;
//	for(int i=0;i<num_partitions;i++){
//		std::cout<<real2ord[i]<<", ";
//	}
//	std::cout<<std::endl;
//
//	std::cout<< "ordered rules, unique blocks: " << unique_blocks <<std::endl;
//	for(int i=0;i<num_rules;i++){
//		std::cout<<(*ordered)[i] <<std::endl;
//	}
//

	delete [] current_offset;
	delete [] expanded_accum_offsets;
	delete [] offsets;
	delete [] real2ord;
	delete [] ord2real;

	return unique_blocks;
}
//Sorts the blocks and everything related to them. After that, the blocks in the same partition are coalesced
int reorder_ruleblocks(PDP_Psystem_REDIX::Structures structures,
		uint* ordered,
		Options options){

	PDP_Psystem_REDIX::Structures reordered=new PDP_Psystem_REDIX::struct_structures;

	reordered->ruleblock.lhs_idx = new LHS_IDX [options->num_rule_blocks+1];
	reordered->ruleblock.rule_idx = new RULE_IDX [options->num_rule_blocks+1];
	reordered->ruleblock.membrane = new MEMBRANE [options->num_rule_blocks];

	reordered->ruleblock_size=structures->pi_rule_size+1;
	reordered->rule.rhs_idx = new RHS_IDX [reordered->ruleblock_size];

	/* Create empty data for LHS */

	reordered->lhs_size=structures->ruleblock.lhs_idx[options->num_rule_blocks];
	reordered->lhs.object = new OBJECT [reordered->lhs_size];
	reordered->lhs.mmultiplicity = new MULTIPLICITY [reordered->lhs_size];
	reordered->lhs.imultiplicity = new INV_MULTIPLICITY [reordered->lhs_size];

	/* Create empty probabilities */
	reordered->probability_size=structures->pi_rule_size*options->num_environments;
	reordered->probability = new PROBABILITY [reordered->probability_size];

	reordered->rhs_size=structures->rule.rhs_idx[structures->ruleblock.rule_idx[options->num_rule_blocks]];
	reordered->rhs.object = new OBJECT [reordered->rhs_size];
	reordered->rhs.mmultiplicity = new MULTIPLICITY [reordered->rhs_size];

	reordered->ruleblock.lhs_idx[0]=0;
	reordered->ruleblock.rule_idx[0]=0;
	reordered->rule.rhs_idx[0]=0;

	for(int i=0;i < options->num_rule_blocks;i++){
		uint block=ordered[i];
		//Auxiliar indexes
		int lhs_init=structures->ruleblock.lhs_idx[block];
		int lhs_end=structures->ruleblock.lhs_idx[block+1];
		int rule_init=structures->ruleblock.rule_idx[block];
		int rule_end=structures->ruleblock.rule_idx[block+1];

		int lhs_offset=reordered->ruleblock.lhs_idx[i];
		int rule_offset=reordered->ruleblock.rule_idx[i];

		//Calculate offset for rules and lhs
		reordered->ruleblock.membrane[i]=structures->ruleblock.membrane[block];
		reordered->ruleblock.lhs_idx[i+1]=lhs_offset+ (lhs_end-lhs_init);
		reordered->ruleblock.rule_idx[i+1]=rule_offset+ (rule_end-rule_init);

		for(int j=0;j<lhs_end-lhs_init;j++){
			//Reassign each object to its new position
			reordered->lhs.object[lhs_offset+j]=structures->lhs.object[lhs_init+j];
			reordered->lhs.mmultiplicity[lhs_offset+j]=structures->lhs.mmultiplicity[lhs_init+j];
			reordered->lhs.imultiplicity[lhs_offset+j]=structures->lhs.imultiplicity[lhs_init+j];
		}

		for(int j=0;j<rule_end-rule_init;j++){
			//Reassign each rule to its new position
			int rhs_init=structures->rule.rhs_idx[rule_init+j];
			int rhs_end=structures->rule.rhs_idx[rule_init+j+1];

			int rhs_offset=reordered->rule.rhs_idx[rule_offset+j];

			reordered->rule.rhs_idx[rule_offset+j+1]=rhs_offset+(rhs_end-rhs_init);

			for(int k=0;k<rhs_end-rhs_init;k++){
				//Reassign each rhs to its new position
				reordered->rhs.object[rhs_offset+k]=structures->rhs.object[rhs_init+k];
				reordered->rhs.mmultiplicity[rhs_offset+k]=structures->rhs.mmultiplicity[rhs_init+k];
			}
			for (int env=0; env<options->num_environments; env++) {
				//Reassign probabilities of each rule
				reordered->probability[env*structures->pi_rule_size+rule_offset+j]=structures->probability
						[env*structures->pi_rule_size+rule_init+j];
			}
		}


	}


	//Checking solution (works alright)
//	for(int i=0;i<options->num_rule_blocks;i++){
//		uint block=ordered[i];
//		if((structures->ruleblock.lhs_idx[block+1]-structures->ruleblock.lhs_idx[block])
//		           !=(reordered->ruleblock.lhs_idx[i+1]-reordered->ruleblock.lhs_idx[i])){
//
//			std::cout<<"i: "<< i << " "<<structures->ruleblock.lhs_idx[block]<<" "
//					<<reordered->ruleblock.lhs_idx[i]<<std::endl;
//		}
//		if(structures->ruleblock.membrane[block]!=reordered->ruleblock.membrane[i]){
//			std::cout<<"i: "<< i << " "<<structures->ruleblock.membrane[block]<<" "
//					<<reordered->ruleblock.membrane[i]<<std::endl;
//		}
//		if((structures->ruleblock.rule_idx[block+1]-structures->ruleblock.rule_idx[block])
//		          !=(reordered->ruleblock.rule_idx[i+1]-reordered->ruleblock.rule_idx[i])){
//			std::cout<<"i: "<< i << " "<<structures->ruleblock.rule_idx[block]<<" "
//					<<reordered->ruleblock.rule_idx[i]<<std::endl;
//		}
//
//	}
//
//
//	std::cout<< "ordered"<<std::endl;
//	for(int i=0;i<options->num_rule_blocks;i++){
//		std::cout<<ordered[i]<<", ";
//	}
//	std::cout<<std::endl;
//
//	std::cout<< "lhs"<<std::endl;
//	for(int i=0;i<structures->lhs_size;i++){
//		std::cout<<structures->lhs.object[i]<<", ";
//	}
//	std::cout<<std::endl;
//	for(int i=0;i<reordered->lhs_size;i++){
//		std::cout<<reordered->lhs.object[i]<<", ";
//	}
//	std::cout<<std::endl;
//
//	std::cout<< "rhs idx"<<std::endl;
//	for(int i=0;i<structures->pi_rule_size;i++){
//		std::cout<<structures->rule.rhs_idx[i]<<", ";
//	}
//	std::cout<<std::endl;
//	for(int i=0;i<structures->pi_rule_size;i++){
//		std::cout<<reordered->rule.rhs_idx[i]<<", ";
//	}
//	std::cout<<std::endl;
//
//
//	std::cout<< "rhs object"<<std::endl;
//	for(int i=0;i<structures->rhs_size;i++){
//		std::cout<<structures->rhs.object[i]<<", ";
//	}
//	std::cout<<std::endl;
//	for(int i=0;i<reordered->rhs_size;i++){
//		std::cout<<reordered->rhs.object[i]<<", ";
//	}
//	std::cout<<std::endl;
//
//	std::cout<< "rule"<<std::endl;
//	for(int i=0;i<structures->ruleblock_size;i++){
//		std::cout<<structures->ruleblock.rule_idx[i]<<", ";
//	}
//	std::cout<<std::endl;
//	for(int i=0;i<options->num_rule_blocks+1;i++){
//		std::cout<<reordered->ruleblock.rule_idx[i]<<", ";
//	}
//	std::cout<<std::endl;
//
//	std::cout<< "prob"<<std::endl;
//	for(int i=0;i<structures->probability_size;i++){
//		std::cout<<structures->probability[i]<<", ";
//	}
//	std::cout<<std::endl;
//	for(int i=0;i<reordered->probability_size;i++){
//		std::cout<<reordered->probability[i]<<", ";
//	}
//	std::cout<<std::endl;

	//Warning: take into account the real size of what we must copy

	memcpy(structures->ruleblock.lhs_idx,reordered->ruleblock.lhs_idx,options->num_rule_blocks*sizeof(LHS_IDX));
	memcpy(structures->ruleblock.rule_idx,reordered->ruleblock.rule_idx,options->num_rule_blocks*sizeof(RULE_IDX));
	memcpy(structures->ruleblock.membrane,reordered->ruleblock.membrane,options->num_rule_blocks*sizeof(MEMBRANE));

	memcpy(structures->rule.rhs_idx,reordered->rule.rhs_idx,reordered->ruleblock_size*sizeof(RHS_IDX));

	memcpy(structures->lhs.object,reordered->lhs.object,reordered->lhs_size*sizeof(OBJECT));
	memcpy(structures->lhs.mmultiplicity,reordered->lhs.mmultiplicity,reordered->lhs_size*sizeof(MULTIPLICITY));
	memcpy(structures->lhs.imultiplicity,reordered->lhs.imultiplicity,reordered->lhs_size*sizeof(INV_MULTIPLICITY));

	memcpy(structures->probability,reordered->probability,reordered->probability_size*sizeof(PROBABILITY));

	memcpy(structures->rhs.object,reordered->rhs.object,reordered->rhs_size*sizeof(OBJECT));
	memcpy(structures->rhs.mmultiplicity,reordered->rhs.mmultiplicity,reordered->rhs_size*sizeof(MULTIPLICITY));



	//Free resources
	delete [] reordered->ruleblock.lhs_idx;
	delete [] reordered->ruleblock.rule_idx;
	delete [] reordered->ruleblock.membrane;

	delete [] reordered->rule.rhs_idx;


	delete [] reordered->lhs.object;
	delete [] reordered->lhs.mmultiplicity;
	delete [] reordered->lhs.imultiplicity;

	delete [] reordered->probability;

	delete [] reordered->rhs.object;
	delete [] reordered->rhs.mmultiplicity;

	delete reordered;
	return 0;
}

}
