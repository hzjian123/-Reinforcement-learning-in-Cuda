#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <torch/torch.h>
#include <string.h>

#define   ROW  32   // length of environment
#define   COL 32    // width of environment
#define   NUM_ACTION 4
#define   GAMMA 0.9 // Q-learning parameter
#define   ALPHA 0.1 // learning rate
#define   DEL_EPS 0.001 // delta_epsilon for each episode
#define NUM_AGENTS 128
#define ENV_SIZE ROW*COL
#define HIDDEN_SIZE 256 // hidden size of networks
#define MAX_STEP 200    // max steps of a single agent

float epsilon;
short d_action[NUM_AGENTS];
curandState* d_randstate;
float* d_epsilon;
int* d_active;  // record if one agent is active
int step = 0;   // record current step
int* d_step;

float Rewards[MAX_STEP * NUM_AGENTS];    // record rewards of all agents in all steps
float Values[MAX_STEP * NUM_AGENTS];   // record values of all agents in all steps
float log_probs[MAX_STEP * NUM_AGENTS]; // record log(d_action) of all agents in all steps
int Done[NUM_AGENTS];   // record the step where every agent becomes inactive

torch::Tensor temp_a, temp_c;   // use extra tensor to help torch track where the loss comes from
int2 h_cstate[NUM_AGENTS];  // host copy of cstate to do network forward on CPU
int2 h_nstate[NUM_AGENTS];  // host copy of nstate to do network forward on CPU

// Initialize action net
struct Actor_Net : torch::nn::Module {
    // input is a one-hot encoding current state, its length is ENV_SIZE, i.e., 32 * 32
    // output is a set of probability distribtuion of 4 possible actions
    Actor_Net() {
        fc1 = register_module("fc1", torch::nn::Linear(ENV_SIZE, HIDDEN_SIZE));
        fc2 = register_module("fc2", torch::nn::Linear(HIDDEN_SIZE, NUM_ACTION));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::softmax(fc2->forward(x), /*dim=*/1);
        return x;
    }
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};
// Initialize Critic net
struct Critic_Net : torch::nn::Module {
    // input is a one-hot encoding next state, its length is ENV_SIZE, i.e., 32 * 32
    // output is a critic value
    Critic_Net() {
        fc1 = register_module("fc1", torch::nn::Linear(ENV_SIZE, HIDDEN_SIZE));
        fc2 = register_module("fc2", torch::nn::Linear(HIDDEN_SIZE, 1));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};

// init models
Critic_Net model_c;
Actor_Net model_a;

torch::optim::SGD optimizer_c(
    model_c.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
torch::optim::SGD optimizer_a(
    model_a.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

__global__ void Agent_init(curandState* d_randstate)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(clock() + tid, tid, 0, &d_randstate[tid]);
}

void agent_init()
{
    // torch init
    torch::manual_seed(1);
    torch::DeviceType device_type;
    // use CPU due to compatibility issues
    //if (torch::cuda::is_available()) {
    //    std::cout << "CUDA available! Training on GPU." << std::endl;
    //    device_type = torch::kCUDA;
    //}
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
    torch::Device device(device_type);
    model_c.to(device);
    model_a.to(device);

    // init epsilon and allocate necessary GPU variables
    epsilon = 1;
    cudaMalloc((void**)&d_epsilon, sizeof(float));
    cudaMalloc((void**)&d_action, sizeof(short) * NUM_AGENTS);
    cudaMalloc((void**)&d_randstate, sizeof(curandState) * NUM_AGENTS);
    cudaMalloc((void**)&d_active, sizeof(int) * NUM_AGENTS);
    cudaMalloc((void**)&Rewards, sizeof(float) * MAX_STEP * NUM_AGENTS);
    cudaMalloc((void**)&Values, sizeof(float) * MAX_STEP * NUM_AGENTS);
    cudaMalloc((void**)&log_probs, sizeof(float) * MAX_STEP * NUM_AGENTS);
    cudaMalloc((void**)&Done, sizeof(int) * NUM_AGENTS);
    cudaMalloc((void**)&d_step, sizeof(int));
    cudaMemcpy(d_epsilon, &epsilon, sizeof(float), cudaMemcpyHostToDevice);

    // use kernel to initialize agents
    Agent_init << <1, NUM_AGENTS >> > (d_randstate);

}
__global__ void Agent_init_episode(int* d_active) {
    int idx = threadIdx.x;
    d_active[idx] = true;   // acitivate all agents when one episode begins
}

void agent_init_episode()
{  
    // reset Rewards, Values, log_probs and step to 0 when one episode begins
    memset(Rewards, 0, sizeof(float) * MAX_STEP * NUM_AGENTS);
    memset(Values, 0, sizeof(float) * MAX_STEP * NUM_AGENTS);
    memset(log_probs, 0, sizeof(float) * MAX_STEP * NUM_AGENTS);
    step = 0;
    Agent_init_episode << <1, NUM_AGENTS >> > (d_active);
}

float agent_adjustepsilon()
{
    float new_eps = epsilon - DEL_EPS;
    if (new_eps > 1) { new_eps = 1; }
    else if (new_eps < 0.1) { new_eps = 0.1; }
    epsilon = new_eps;
    cudaMemcpy(d_epsilon, &epsilon, sizeof(float), cudaMemcpyHostToDevice);
    return epsilon;
}

__global__ void Agent_action(short* d_action, float* can_actions, curandState* d_randstate, int* d_active, float* log_probs, int* d_step)
{
    int agent_id = threadIdx.x;
    if (d_active[agent_id] == false)    return;

    float rand_num = curand_uniform(&d_randstate[agent_id]);
    float accu = 0.0;
    float accu_temp = 0.0;
    // use the output of actor_network, i.e., a probability distribution of 4 actions, 
    // to decide which action to take
    for (int i = 0; i < NUM_ACTION; i++)
    {
        accu += can_actions[i + agent_id * NUM_ACTION];
        if (accu_temp < rand_num && rand_num <= accu)
        {
            d_action[agent_id] = (short)i;
            log_probs[MAX_STEP * agent_id + d_step[0]] = logf(i);
        }
        accu_temp = accu;
    }
}

short* agent_action(int2* cstate)
{
    int i;
    int index;
    int pos_x, pos_y;
    int state[ENV_SIZE];    // create a one-hot encoding array
    short action_sample = 0;
    float can_actions[NUM_ACTION * NUM_AGENTS]; // a new array to record candidate actions of all agents
    // copy gpu cstate to a cpu cstate variable
    cudaMemcpy(h_cstate, cstate, sizeof(h_cstate), cudaMemcpyDeviceToHost);

    // actor network feed forward on CPU
    for (i = 0; i < NUM_AGENTS; i++)
    {
        if (d_active[i] == 0)  {continue;}
        // set the one-hot encoding array to be all zero
        memset(state, 0, sizeof(int) * ENV_SIZE);
        pos_x = h_cstate[i].x;
        pos_y = h_cstate[i].y;
        index = pos_x + pos_y * COL;
        state[index] = 1;   // one-hot encoding
        // turn state(array) to t_state(tensor) to feed forward
        torch::Tensor t_state = torch::from_blob(state, { ENV_SIZE });
        torch::Tensor output_a = model_a.forward(t_state);
        // turn output_a(tensor) back to f_output_a(array) to further work on GPU
        auto f_output_a = output_a.data<float>();
        memcpy(can_actions + i * NUM_ACTION, f_output_a, sizeof(f_output_a));
        // try to save part of network output in a tensor to help torch
        // track data flow direction later
        if (i == NUM_AGENTS - 1) { temp_a = output_a; }
    }
    // allocate memory for output of actor_network to use it on GPU
    cudaMalloc((void**)&can_actions, sizeof(float) * NUM_ACTION * NUM_AGENTS);
    cudaMemcpy(d_step, &step, sizeof(int), cudaMemcpyHostToDevice);
    Agent_action << <1, NUM_AGENTS >> > (d_action, can_actions, d_randstate, d_active, log_probs, d_step);

    return d_action;
}

// just a simple sum function
float sum(int *arr, int len) {
    float Sum = 0;
    for (int i = 0; i < len; i++) {
        Sum += arr[i];
    }
    return Sum;
}
__global__ void Cal_loss(int2* cstate, int2* nstate, int* d_active, float* Rewards, float* Values, float* log_probs, int* Done, int* loss_actor, float* loss_critic, float* d_Q_Val) {
    int agent_id = threadIdx.x;
    float q_Val = 0;
    int index, i, j;
    //calculate Q_val
    for (i = Done[agent_id]; i > 0; i--) {
        index = Done[agent_id] - i;
        q_Val = Rewards[MAX_STEP * agent_id + index] + q_Val * GAMMA;
        d_Q_Val[index + agent_id * MAX_STEP] = q_Val;
    }

    // calculate advantage
    float Adv[MAX_STEP];
    for (i = 0; i < Done[agent_id]; i++)
    {
        Adv[i] = d_Q_Val[MAX_STEP * agent_id + i] - Values[MAX_STEP * agent_id + i];
    }

    float Sum_C = 0;
    for (j = 0; j < Done[agent_id]; j++) {
        Sum_C += Adv[j] * Adv[j];
    }
    // calculate loss of critic network
    loss_critic[agent_id] = Sum_C / Done[agent_id];
    float Sum_A = 0;
    for (j = 0; j < Done[agent_id]; j++) {
        Sum_A += Adv[j] * log_probs[MAX_STEP * agent_id + j];
    }
    // calculate loss of actor network
    loss_actor[agent_id] = -(Sum_A) / Done[agent_id];
}

void agent_update(int2* cstate, int2* nstate, float* rewards)
{
    int i, pos_x, pos_y, index;
    int state[ENV_SIZE];
    int loss_actor[NUM_AGENTS];
    float loss_critic[NUM_AGENTS];
    float d_Q_Val[MAX_STEP * NUM_AGENTS];
    memset(d_Q_Val, 0, sizeof(float) * MAX_STEP * NUM_AGENTS);
    cudaMemcpy(h_nstate, nstate, sizeof(h_nstate), cudaMemcpyDeviceToHost);

    for (i = 0; i < NUM_AGENTS; i++) {
        if (d_active[i] == 0) { continue; }

        memset(state, 0, sizeof(int) * ENV_SIZE);
        pos_x = h_nstate[i].x;
        pos_y = h_nstate[i].y;
        index = pos_x + pos_y * COL;
        state[index] = 1;
        torch::Tensor t_state = torch::from_blob(state, { ENV_SIZE });
        auto output_c = model_c.forward(t_state);
        float value = output_c.data<float>()[0];
        Values[MAX_STEP * i + step] = value;
        if (rewards[i] != 0) {
            d_active[i] = 0;
            Done[i] = step;
        }
        if (i == NUM_AGENTS - 1)    temp_c = output_c;
    }
    memcpy(Rewards + step * NUM_AGENTS, rewards, sizeof(float) * NUM_AGENTS);

    // When #active_agents < 20%, start to calculate loss on GPU and then do backward on CPU
    // This part of code works only once when an episode is going to end
    if (sum(d_active, NUM_AGENTS) / NUM_AGENTS < 0.2) {
        // calculate loss on GPU
        cudaMalloc((void**)&loss_actor, sizeof(int) * NUM_AGENTS);
        cudaMalloc((void**)&loss_critic, sizeof(float) * NUM_AGENTS);
        cudaMalloc((void**)&d_Q_Val, sizeof(float) * MAX_STEP * NUM_AGENTS);
        Cal_loss << <1, NUM_AGENTS >> > (cstate, nstate, d_active, Rewards, Values, log_probs, Done, loss_actor, loss_critic, d_Q_Val);

        // backward on CPU
        torch::Tensor Loss_actor, Loss_critic;
        for (i = 0; i < NUM_AGENTS; i++)
        {
            // turn loss(array) into t_loss(tensor)
            torch::Tensor t_loss_actor = torch::from_blob(loss_actor, { NUM_AGENTS });
            torch::Tensor t_loss_critic = torch::from_blob(loss_critic, { NUM_AGENTS });

            // temp_a and temp_c are tensors containing part of network output.
            // we try to help torch track data flow in this way(not sure if it works)
            Loss_actor = t_loss_actor[i] + temp_a * 0;
            Loss_critic = t_loss_critic[i] + temp_c * 0;

            // update networks
            Loss_actor.backward();
            optimizer_a.step();
            Loss_critic.backward();
            optimizer_c.step();
        }
    }
    step++; // step+=1 when one step ends
}
