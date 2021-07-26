import argparse
from itertools import count
import os

import gym
import scipy.optimize
import pdb

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from torch.utils.tensorboard import SummaryWriter

import swimmer
import reacher
import inverted_double_pendulum
import walker2d
import hopper
import halfcheetah

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--test-env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--reward_model', help='the reward function model')
parser.add_argument('--prefix', default='', help='the prefix of the saved model')
parser.add_argument('--mode', default=None, help='the mode')
parser.add_argument('--output_path')
args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.system('mkdir -p '+args.output_path)
writer = SummaryWriter(args.output_path)
log_stream = open(os.path.join(args.output_path, 'log.txt'), 'w')

if args.mode is not None:
    env = gym.make(args.env_name, reward_mode=args.mode)
else:
    env = gym.make(args.env_name)

test_env = gym.make(args.test_env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

if args.mode is not None:
    if args.mode == 'state_only':
        reward_net = RewardNet(num_inputs).float()
    elif args.mode == 'state_pair':
        reward_net = RewardNet(num_inputs*2).float()
    elif args.mode == 'state_action':
        reward_net = RewardNet(num_inputs+num_actions).float()
    reward_net.load_state_dict(torch.load(args.reward_model, map_location='cpu'))
    env.set_reward_net(reward_net)

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)



def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

'''
policy_net.load_state_dict(torch.load('checkpoints/trex_CustomHopper-v0_state_only_1_003065_rew_67.29.pth')['policy'])
#policy_net.load_state_dict(torch.load('checkpoints/Hopper-v3_005400_rew_2706.30.pth')['policy'])
test_env.seed(1234)
rewards = []
custom_rewards = []
for j in range(20):
        reward = []
        custom_reward = []
        state = test_env.reset()
        done = False
        while not done:
            action = select_action(state)
            action = action.data[0].numpy()
            custom_rew = reward_net.compute_reward(state)
            next_state, rew, done, _ = test_env.step(action)
            state = next_state
            reward.append(rew)
            custom_reward.append(custom_rew)
        print(test_env.sim.data.qpos[0])
        rewards.append(np.sum(reward))
        custom_rewards.append(np.sum(custom_reward))
print(np.mean(rewards))
print(np.mean(custom_rewards))
pdb.set_trace()
'''

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

def evaluate(test_env):
    test_env.seed(1234)
    rewards = []
    for j in range(20):
        reward = []
        state = test_env.reset()
        done = False
        while not done:
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, rew, done, _ = test_env.step(action)
            state = next_state
            reward.append(rew)
        rewards.append(np.sum(reward))
    return np.mean(rewards)

for i_episode in count(1):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        state = env.reset()

        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes
    batch = memory.sample()
    update_params(batch)

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, reward_sum, reward_batch))
        writer.add_scalar('train/last_rew', reward_sum, i_episode)
        writer.add_scalar('train/avg_rew', reward_batch, i_episode)
    if i_episode % args.save_interval == 0:
        rew = evaluate(test_env)
        print('Episode {}, Evaluation Reward {}'.format(i_episode, rew))
        log_stream.write('Episode {}, Evaluation Reward {}\n'.format(i_episode, rew))
        log_stream.flush()
        writer.add_scalar('test/acc', rew, i_episode)
        torch.save({'policy':policy_net.state_dict(), 'value':value_net.state_dict()}, os.path.join(args.output_path, 'epoch_{:07d}_rew_{:.2f}.pth'.format(i_episode, rew)))

writer.close()
log_stream.close()
