// GPU programming
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#define   ROW  32
#define   COL 32
#define   NUM_ACTION 4
#define   GAMMA 0.9
#define   ALPHA 0.1
#define   DEL_EPS 0.001
#define NUM_AGENTS 128

float epsilon;
short* d_action;
curandState* d_randstate;
float* d_qtable;
float* d_epsilon;
bool* d_active;

__global__ void Agent_init(bool* d_active,curandState* d_randstate)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	d_active[tid] = true;
	curand_init(clock() + tid, tid, 0, &d_randstate[tid]);
}
__global__ void Q_init(float* d_qtable)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	d_qtable[idx] = 0;
}
void agent_init()
{
	epsilon = 1;
	int d_q_size = ROW * COL * NUM_ACTION;
	cudaMalloc((void**)&d_qtable, sizeof(float) * d_q_size);
	cudaMalloc((void**)&d_epsilon, sizeof(float));
	cudaMalloc((void**)&d_action, sizeof(short) * NUM_AGENTS);
	cudaMalloc((void**)&d_randstate, sizeof(curandState) * NUM_AGENTS);
	cudaMalloc((void**)&d_active, sizeof(bool) * NUM_AGENTS);
	cudaMemcpy(d_epsilon, &epsilon, sizeof(float), cudaMemcpyHostToDevice);//Assign 1 to d_eps

	Agent_init << <1, NUM_AGENTS >> > (d_active,d_randstate);
	Q_init << <ROW * COL, NUM_ACTION >> > (d_qtable);

}
__global__ void Agent_init_episode(bool* d_active) {
	int idx = threadIdx.x;
	d_active[idx] = true;
}

void agent_init_episode()
{
	// add your codes
	Agent_init_episode << <1, NUM_AGENTS >> > (d_active);
}

float agent_adjustepsilon()
{
	// add your codes
	float new_eps = epsilon - DEL_EPS;
	if (new_eps > 1) { new_eps = 1; }
	else if (new_eps < 0.1) { new_eps = 0.1; }
	epsilon = new_eps;
	cudaMemcpy(d_epsilon, &epsilon, sizeof(float), cudaMemcpyHostToDevice);
	return epsilon;
}
__global__ void Agent_action(int2* cstate, short* d_action, curandState* d_randstate, float* d_epsilon, float* d_q_table, bool* d_active)
{
	int agent_id = threadIdx.x;
	int pos_x = cstate[agent_id].x;
	int pos_y = cstate[agent_id].y;
	short action = 0;
	float randstate = curand_uniform(&d_randstate[agent_id]);
	if (d_active[agent_id] == false) {
		return;
	}
	if (randstate < *d_epsilon)
	{
		action = (short)(ceil(curand_uniform(&d_randstate[agent_id]) * NUM_ACTION)-1);
	}
	else
	{
		int index = (pos_x + pos_y * COL) * NUM_ACTION;
		float q_best = d_q_table[index];
		for (int i = 0; i < NUM_ACTION; ++i) {
			if (d_q_table[index + i] > q_best) {
				action = (short)i;
				q_best = d_q_table[index + i];
			}
		}
	}
	d_action[agent_id] = action;
}
short* agent_action(int2* cstate)
{
	// add your codes
	Agent_action <<<1, NUM_AGENTS >>> (cstate, d_action, d_randstate, d_epsilon, d_qtable, d_active );
	return d_action;
}
__global__ void Agent_update(int2* cstate, int2* nstate, float* rewards, float* d_q_table, short* d_action , bool* d_active)
{
	int agent_id = threadIdx.x;
	int pos_x = cstate[agent_id].x;
	int pos_y = cstate[agent_id].y;
	int npos_x = nstate[agent_id].x;
	int npos_y = nstate[agent_id].y;
	int index = (pos_x + pos_y * COL) * NUM_ACTION + d_action[agent_id];
	int n_index = (npos_x + npos_y * COL) * NUM_ACTION;
	float q_best = d_q_table[n_index];
	short next_action;
	float reward = rewards[agent_id];
	if (d_active[agent_id] == false) {
		return;
	}
	if (reward == 0) {
		for (int i = 0; i < NUM_ACTION; ++i) {
			if (d_q_table[n_index + i] > q_best) {
				next_action = (short)i;
				q_best = d_q_table[n_index + i];
			}
		}
		d_q_table[index] += ALPHA * ( GAMMA * q_best - d_q_table[index]);
	}
	else {
		d_q_table[index] += ALPHA * (reward - d_q_table[index]);
		d_active[agent_id] = 0;
	}
}
void agent_update(int2* cstate, int2* nstate, float* rewards)
{
	// add your codes
	Agent_update << <1, NUM_AGENTS >> > (cstate, nstate, rewards, d_qtable, d_action, d_active);
}
