from q_learner import QLearning, ExpBonusQLearning, ActiveQLearning
import numpy as np
from functools import reduce
import random
from utils.convert2base import obs_to_int_pi, s_to_sp, convert_to_base
import sys

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
import argparse
from env.env_factory import get_env
import os
import csv
from collections import deque


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--exp_mode', type=str, default='active_cen', help='active_cen|epsilon|bonus')
    parser.add_argument('--env', type=str, default='room', help='room|room30|secret_room|push_box|island')
    parser.add_argument('--multilevel', default=False, action='store_true')
    parser.add_argument('--all_subspace', default=False, action='store_true')
    parser.add_argument('--tree_subspace', default=False, action='store_true')
    parser.add_argument('--no_range_info', default=False, action='store_true')
    parser.add_argument('--mixed_explore', default=False, action='store_true')
    parser.add_argument('--random_sample', default=False, action='store_true')
    parser.add_argument('--stochastic_select_subspace', default=False, action='store_true')
    parser.add_argument('--level_penalty', type=float, default=0)
    parser.add_argument('--total_episode', type=int, default=15000)
    parser.add_argument('--eval_every_episode', type=int, default=200)
    parser.add_argument('--log_folder', type=str, default='.')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--alpha1', type=float, default=0.1, help='learning rate for exploration policy')
    parser.add_argument('--alpha2', type=float, default=0.05, help='learning rate for active policy')
    parser.add_argument('--recip_t', type=float, default=50, help='reciprocal temperature')
    parser.add_argument('--subspace_q_size', type=int, default=10, help='subspace_q_size')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--exp_batch_name', type=str, default='exp_batch_no_name')
    parser.add_argument('--replay_size', type=int, default=400000)
    parser.add_argument('--bonus_coef', type=float, default=0.05)
    args = parser.parse_args()
    return args


def eval(policy1, policy2, eval_env, raw_obs_dim):
    print('EVAL')
    episode_rew = 0
    s = eval_env.reset()
    episode_rews = []
    for _ in range(10):
        while True:
            s, _ = obs_to_int_pi(s, base=eval_env.grid_size, raw_dim=raw_obs_dim)
            a1 = policy1.select_action(s, 0)
            a2 = policy2.select_action(s, 0)
            s_next, r, done = eval_env.step([a1, a2])
            episode_rew += r
            s = s_next
            if done:
                episode_rews.append(episode_rew)
                episode_rew = 0
                s = eval_env.reset()
                break
    return np.mean(episode_rews)


def main():
    # params
    args = get_args()
    log_path = 'log/{}/{}/{}_{}'.format(args.log_folder, args.exp_batch_name, args.exp_name, args.seed)
    os.makedirs(log_path, exist_ok=True)

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    env = get_env(args.env)()
    eval_env = get_env(args.env)()
    raw_obs_dim = env.observation_space.nvec.size
    agent1_count = np.zeros((env.grid_size, env.grid_size), dtype=np.int32)
    agent2_count = np.zeros((env.grid_size, env.grid_size), dtype=np.int32)
    n_states = reduce(lambda x, y: x * y, env.observation_space.nvec)
    kargs = {'n_states': n_states,
             'n_actions': reduce(lambda x, y: x * y, env.action_space.nvec),
             'base': env.grid_size,
             'raw_dim': len(env.observation_space.nvec),
             'observation_space': env.observation_space,
             'gamma': args.gamma,
             'alpha': args.alpha1,
             }

    meta_counter = ActiveQLearning(**kargs)

    if args.exp_mode == 'active_cen':
        kargs.update({'all_subspace': args.all_subspace,
                      'tree_subspace': args.tree_subspace,
                      'no_range_info': args.no_range_info,
                      'stochastic_select_subspace': args.stochastic_select_subspace,
                      'recip_t': args.recip_t,
                      'subspace_q_size': args.subspace_q_size,
                      'replay_size': args.replay_size,
                      'level_penalty': args.level_penalty,
                      'priority_sample':not args.random_sample})
        meta_q = ActiveQLearning(**kargs)
        kargs['n_actions'] = env.action_space.nvec[0]
        q_learner1 = ActiveQLearning(**kargs)
        q_learner2 = ActiveQLearning(**kargs)

        kargs['alpha'] = args.alpha2
        q_learner_target1 = ActiveQLearning(**kargs)
        q_learner_target2 = ActiveQLearning(**kargs)
        eps = 0
    elif args.exp_mode == 'epsilon':
        kargs['n_actions'] = env.action_space.nvec[0]
        q_learner1 = QLearning(**kargs)
        q_learner2 = QLearning(**kargs)
        init_eps = 1  # linear decrease to zero
    elif args.exp_mode == 'bonus':
        kargs['n_actions'] = env.action_space.nvec[0]
        kargs['bonus_coef'] = args.bonus_coef
        q_learner1 = ExpBonusQLearning(**kargs)
        q_learner2 = ExpBonusQLearning(**kargs)
        init_eps = 0.1

    s_raw = env.reset()
    s, s_p = obs_to_int_pi(s_raw, base=env.grid_size, raw_dim=raw_obs_dim)
    episode_rew = 0
    episode_rews = deque(maxlen=200)
    train_rews = []

    episode_count = 0
    episode_actions = []
    episode_states = []
    eval_rews = []
    if args.exp_mode == 'active_cen':
        norm_ents_log = [[] for _ in range(sum(len(row) for row in meta_q.counts))]
    selected_count_id = []
    selected_level = []
    n_new_state_action_list = []
    n_new_state_action = 0
    n_actions_per_agent = env.action_space.nvec[0]
    goal_s, goal_a = None, None
    steps_to_goal = None
    episode_step, total_step_count = 0, 0
    total_step_count_list = []

    while True:
        agent1_count[s_raw[1], s_raw[0]] += 1
        agent2_count[s_raw[3], s_raw[2]] += 1
        episode_step += 1
        total_step_count += 1
        if args.exp_mode == 'epsilon' or args.exp_mode == 'bonus':
            eps = init_eps * (1 - (episode_count / args.total_episode))
        if args.mixed_explore:
            alpha = episode_count / args.total_episode
            a1 = q_learner1.select_action(s, eps, q_learner_target1.q_table, alpha)
            a2 = q_learner2.select_action(s, eps, q_learner_target2.q_table, alpha)
        else:
            a1 = q_learner1.select_action(s, eps)
            a2 = q_learner2.select_action(s, eps)
            q_learner1.update_count(s, a1)
            q_learner2.update_count(s, a2)

        a = env.action_space.nvec[0] * a1 + a2
        if meta_counter.count[s, a] == 0:
            n_new_state_action += 1
        meta_counter.update_count(s, a)
        if 'active' in args.exp_mode:
            meta_q.update_count(s, a, multilevel=args.multilevel,
                                all_subspace=args.all_subspace,
                                tree_subspace=args.tree_subspace)
        episode_actions.append([a1, a2])
        episode_states.append(s)
        if 'active' in args.exp_mode and episode_count >= 1 and (
                s == goal_s or s == s_to_sp(goal_s, base=env.grid_size, raw_dim=raw_obs_dim)):
            steps_to_goal = episode_step
        s_next_raw, r, done = env.step([a1, a2])
        s_next, s_next_p = obs_to_int_pi(s_next_raw, base=env.grid_size, raw_dim=raw_obs_dim)
        if 'active' in args.exp_mode:
            q_learner1.insert_data(s, a1, r, s_next, done)
            q_learner2.insert_data(s, a2, r, s_next, done)
            q_learner_target1.insert_data(s, a1, r, s_next, done)
            q_learner_target2.insert_data(s, a2, r, s_next, done)
            meta_q.insert_data(s, a, r, s_next, done)
        else:
            q_learner1.update_q(s, a1, r, s_next, done)
            q_learner2.update_q(s, a2, r, s_next, done)

        episode_rew += r
        s = s_next
        s_raw = s_next_raw

        if done:
            episode_states.append(s)
            episode_count += 1
            episode_rews.append(episode_rew)
            if episode_count % 20 == 0:
                print("episode {}, rew {}, avg. rew {}, n_new_state_action avg {:.4f}, step to goal {}, eps {}".
                  format(episode_count, episode_rew, np.mean(episode_rews), np.mean(n_new_state_action_list[-100:]),
                         steps_to_goal, eps))
            n_new_state_action_list.append(n_new_state_action)
            steps_to_goal = None
            n_new_state_action = 0
            episode_step = 0

            if 'active' in args.exp_mode:
                goal_s, goal_a = meta_q.get_goal(multilevel=args.multilevel, all_subspace=args.all_subspace)
                goal_a1, goal_a2 = divmod(goal_a, n_actions_per_agent)
                if episode_count % 20 == 0:
                    print('Restricted space level {},  id {}. Goal state  {}, Goal action {}'.format(
                        meta_q.level, meta_q.count_id, convert_to_base(goal_s, env.grid_size), goal_a))
                q_learner_target1.update_q_from_D(reset_q=False)
                q_learner_target2.update_q_from_D(reset_q=False)
                q_learner1.update_q_from_D(goal=(goal_s, goal_a1))
                q_learner2.update_q_from_D(goal=(goal_s, goal_a2))
                meta_q.goal_q.clear()
            episode_actions = []
            episode_states = []

            if episode_count % args.eval_every_episode == 1:
                total_step_count_list.append(total_step_count)
                if 'active' in args.exp_mode:
                    policy1, policy2 = q_learner_target1, q_learner_target2
                else:
                    policy1, policy2 = q_learner1, q_learner2
                eval_rew = eval(policy1, policy2, eval_env, raw_obs_dim)
                eval_rews.append(eval_rew)
                train_rews.append(np.mean(episode_rews))
                print("eval rew", eval_rew)
                if 'active' in args.exp_mode:
                    selected_count_id.append(meta_q.count_id)
                    selected_level.append(meta_q.level)
                    norm_ents = meta_q.compute_ent_all()
                with open(os.path.join(log_path, 'data.csv'), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['episode'] + list(range(1, args.total_episode + 1, args.eval_every_episode)))
                    writer.writerow(['step'] + total_step_count_list)
                    writer.writerow(['eval_rew'] + eval_rews)
                    writer.writerow(['train_rew'] + train_rews)
                    if 'active' in args.exp_mode:
                        writer.writerow(['level'] + selected_level)
                        writer.writerow(['count_id'] + selected_count_id)
                        for i, norm_ent in enumerate(norm_ents):
                            norm_ents_log[i].append(norm_ent)
                            writer.writerow(['norm_ent {}'.format(i)] + norm_ents_log[i])
            episode_rew = 0
            s_raw = env.reset()
            s, s_p = obs_to_int_pi(s_raw, base=env.grid_size, raw_dim=raw_obs_dim)
            if episode_count >= args.total_episode:
                break


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    main()
