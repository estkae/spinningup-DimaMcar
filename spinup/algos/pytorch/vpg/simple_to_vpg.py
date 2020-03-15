import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from gym import wrappers
import scipy
from scipy import signal


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def train(env_name='Acrobot-v1', hidden_sizes=[64]*2, lr=3e-4,
          epochs=150, batch_size=4000, render=False, vf_lr=1e-3,
         gamma=0.99, lam=0.97, train_v_iters=80):
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network and value estimator (actor and critic)
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
    values_net = mlp(sizes=[obs_dim] + hidden_sizes + [1])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make normalization function
    def normalize(a):
        return (a - np.mean(a, axis=0)) / np.std(a, axis=0)

    # make function to estimate values
    def get_values(obs):
        values = values_net(obs)
        return torch.squeeze(values, -1)

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, adv):
        logp = get_policy(obs).log_prob(act)
        return -(logp * adv).mean()

    def value_loss(obs, rets):
        return ((get_values(obs) - rets) ** 2).mean()

    def discount_cumsum(x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input: 
            vector x, 
            [x0, 
             x1, 
             x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,  
             x1 + discount * x2,
             x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    # make policy and values function optimizers
    optimizer = Adam(logits_net.parameters(), lr=lr)
    optimizer_values = Adam(values_net.parameters(), lr=vf_lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths
        batch_adv = []  # for collecting episode advantage

        # reset episode-specific variables
        obs = env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep
        ep_obs = []  # list of episode observations
        ep_acts = []  # list of episode acts


        # collect experience by acting in the environment with current policy
        while True:
            # rendering
            if render:
                env.render()

            # save obs
            ep_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # save action, reward
            ep_acts.append(act)
            ep_rews.append(rew)

            if done or len(ep_rews) >= batch_size:
                # if episode is over, record info about episode
                ep_ret, ep_len = ep_rews, len(ep_rews)

                batch_rets = ep_ret
                batch_lens = ep_len
                batch_obs = ep_obs[:-1]
                batch_acts = ep_acts[:-1]           

                #Calculate discounted rewards
                batch_weights += list(discount_cumsum(ep_rews[:-1], gamma))

                # Calculate values and advantage
                vals = get_values(torch.as_tensor(ep_obs, dtype=torch.float32))
                deltas = torch.tensor(ep_rews[:-1]) + gamma * vals[1:] - vals[:-1]
                batch_adv += list(discount_cumsum(deltas.detach().numpy(), gamma * lam))

                # reset episode-specific variables
                obs, done, ep_rews, ep_obs, ep_acts = env.reset(), False, [], [], []

                # end experience loop if we have enough of it
                break

        # take a single policy gradient update step
        batch_adv = normalize(batch_adv)
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  adv=torch.as_tensor(batch_adv, dtype=torch.float32),
                                  )
        batch_loss.backward()
        optimizer.step()

        # Fit value function
        for i in range(train_v_iters):
            optimizer_values.zero_grad()
            batch_value = value_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                     rets=torch.as_tensor(batch_weights, dtype=torch.float32)
                                     )
            batch_value.backward()
            optimizer_values.step()

        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.sum(batch_rets), batch_lens))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='Acrobot-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--hid', type=int, default=64)
    args = parser.parse_args()
    print('\nUsing simplest VPG.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr,
         vf_lr=args.vf_lr, epochs=args.epochs, hidden_sizes=[args.hid]*args.l)
