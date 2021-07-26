import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import gym

import models
import dataset

import pdb

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--train_demo_files', nargs='+', help='the path to the training demonstrations files')
parser.add_argument('--test_demo_files', nargs='+', help='the path to the testing demonstrations files')
parser.add_argument('--train_traj_nums', nargs='+', type=int, help='the number of trajectories for each training demonstration file')
parser.add_argument('--num_epochs', type=int, default=100, help='the path to the testing demonstrations files')
parser.add_argument('--mode', default='state_only', help='the mode of the reward function')
parser.add_argument('--dataset_mode', default='partial', help='the dataset mode')
parser.add_argument('--output_model_path', help='the output path for models and logs')
parser.add_argument('--traj_len', type=int, help='the length of the partial trajectory')
args = parser.parse_args()

if not os.path.exists(args.output_model_path):
    os.system('mkdir -p '+'/'.join(args.output_model_path.split('/')[:-1]))

if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False

env = gym.make(args.env_name)
test_env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

if args.dataset_mode == 'partial':
    train_dataset = dataset.RankingLimitDataset(args.train_demo_files, args.train_traj_nums, num_inputs, num_actions, mode=args.mode, traj_len=args.traj_len)
    test_dataset = dataset.RankingLimitDataset(args.test_demo_files, None, num_inputs, num_actions, mode=args.mode, traj_len=args.traj_len)
elif args.dataset_mode == 'traj':
    train_dataset = dataset.RankingTrajDataset(args.train_demo_files, args.train_traj_nums, num_inputs, num_actions, mode=args.mode)
    test_dataset = dataset.RankingTrajDataset(args.test_demo_files, None, num_inputs, num_actions, mode=args.mode)
else:
    raise NotImplementedError
print(len(train_dataset))
train_loader = data_utils.DataLoader(train_dataset, collate_fn=dataset.rank_collate_func, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data_utils.DataLoader(test_dataset, collate_fn=dataset.rank_collate_func, batch_size=1, shuffle=True, num_workers=4)

if args.mode == 'state_only':
    reward_net = models.RewardNet(num_inputs).float()
elif args.mode == 'state_pair':
    reward_net = models.RewardNet(num_inputs*2).float()
elif args.mode == 'state_action':
    reward_net = models.RewardNet(num_inputs+num_actions).float()
else:
    raise NotImplementedError
if use_gpu:
    reward_net = reward_net.cuda()
optimizer = optim.Adam(reward_net.parameters(), lr=0.001, weight_decay=0.0005)

best_acc = 0
for epoch in range(args.num_epochs):
    counter = 0
    acc_counter = 0
    if epoch % args.save_interval == 0:
      for iter_, data in enumerate(test_loader):
        traj1, rew1, traj2, rew2 = data
        if use_gpu:
            traj1, rew1, traj2, rew2 = [item.cuda() for item in traj1], rew1.cuda(), [item.cuda() for item in traj2], rew2.cuda()
        bs1 = len(traj1)
        bs2 = len(traj2)
        assert bs1 == bs2
        
        pred_rew1 = torch.cat([torch.sum(reward_net(item), dim=0, keepdim=True) for item in traj1], dim=0)
        pred_rew2 = torch.cat([torch.sum(reward_net(item), dim=0, keepdim=True) for item in traj2], dim=0)
        pred_rank = torch.lt(pred_rew1, pred_rew2)
        gt_rank = torch.lt(rew1, rew2)
        acc_counter += torch.sum(pred_rank==gt_rank)
        counter += bs1
        if iter_ > 10000:
            break
      print('Epoch {}, Acc {}'.format(epoch, acc_counter/counter))
      if acc_counter/counter > best_acc:
        best_acc = acc_counter/counter
        torch.save(reward_net.state_dict(), args.output_model_path)

    for iter_, data in enumerate(train_loader):
        traj1, rew1, traj2, rew2 = data
        if use_gpu:
            traj1, rew1, traj2, rew2 = [item.cuda() for item in traj1], rew1.cuda(), [item.cuda() for item in traj2], rew2.cuda()
        bs1 = len(traj1)
        bs2 = len(traj2)
        assert bs1 == bs2
        
        optimizer.zero_grad()
        pred_rew1 = torch.cat([torch.sum(reward_net(item), dim=0, keepdim=True) for item in traj1], dim=0)
        pred_rew2 = torch.cat([torch.sum(reward_net(item), dim=0, keepdim=True) for item in traj2], dim=0)
        reward_sum = torch.cat([pred_rew1, pred_rew2], dim=1)
        rank_label = (torch.lt(rew1, rew2)).long()
        loss = nn.CrossEntropyLoss()(reward_sum, rank_label)
        loss.backward()
        optimizer.step()
        if iter_ % args.log_interval == 0:
            print('epoch {}, iter {}, training loss {}'.format(epoch, iter_, loss.item()))
        if iter_ > 5000:
            break
