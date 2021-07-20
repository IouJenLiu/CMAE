DIGITS = '0123456789abcdefghijklmnopqrstuvwxyz'
import numpy as np
import copy


def convert_to_base(decimal_number, base):
    remainder_stack = []

    while decimal_number > 0:
        remainder = decimal_number % base
        remainder_stack.append(remainder)
        decimal_number = decimal_number // base

    new_digits = []
    while remainder_stack:
        new_digits.append(DIGITS[remainder_stack.pop()])

    return ''.join(new_digits)


def obs_to_int(obs, base, raw_dim):
    """
    :param obs: obs or batch of obs
    :param base:
    :param raw_dim:
    :return: batch of level 3 represeantaion [bz, 1]
    """
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    # raw state --> level 3 representation
    if raw_dim == 5:
        ret = obs[:, 4] * base**4 + obs[:, 0] * base**3 + obs[:, 1] * base**2 + obs[:, 2] * base + obs[:, 3]
    elif raw_dim == 6:
        ret = obs[:, 0] * base**5 + obs[:, 1] * base**4 + obs[:, 2] * base**3 + obs[:, 3] * base**2 + obs[:, 4] * base + obs[:, 5]
    return ret.reshape(-1, 1)


def obs_to_int_level1(obs, base, raw_dim):
    """
    :param obs: np array (1) or (bz, 1)
    """
    if raw_dim == 5:
        return obs[:, 0] * base + obs[:, 1], obs[:, 4]  # o1 / e
    elif raw_dim == 6:
        return obs[:, 0] * base + obs[:, 1], obs[:, 4] * base + obs[:, 5]  # o1 / e


def obs_to_int_two_vars(obs, base, raw_dim):
    """
    :param obs: np array (1) or (bz, 1)
    """
    int_reps = []
    for i in range(raw_dim):
        for j in range(i + 1, raw_dim):
            int_reps.append(obs[:, i] * base + obs[:, j])
    return int_reps




def obs_to_int_level2(obs, base, raw_dim):
    """
    :param obs: np array (1) or (bz, 1)
    """
    if raw_dim == 5: # room / secret room
        return obs[:, 0] * base**3 + obs[:, 1] * base**2 + obs[:, 2] * base + obs[:, 3], \
            obs[:, 4] * base**2 + obs[:, 0] * base + obs[:, 1] # o1o2 / eo1
    elif raw_dim == 6: # box
        return obs[:, 0] * base**3 + obs[:, 1] * base**2 + obs[:, 2] * base + obs[:, 3], \
               obs[:, 0] * base**3 + obs[:, 1] * base**2 + obs[:, 4] * base + obs[:, 5]  # o1o2/o1e


def obs_to_level_int(obs, base, raw_dim, level, all_subspace):
    """
    :param obs: raw observation, np array (raw_dim) or np array (bz, raw_dim)
    convert raw observation to level int representation [(bz, 1)] * # of projected space of level
    :return a list of proj_ints [(bz), (bz), ...]
    """
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)

    if level == 3:
        return [obs_to_int(obs, base, raw_dim)]
    elif level == 0:
        return [obs[:, i] for i in range(obs.shape[1])]
    elif level == 1:
        if all_subspace:
            return obs_to_int_two_vars(obs, base, raw_dim)
        else:
            return obs_to_int_level1(obs, base, raw_dim)
    elif level == 2:
        return obs_to_int_level2(obs, base, raw_dim)


def obs_to_projected_int(obs, base, raw_dim, level, count_id, all_subspace):
    """
    :return: the int_rep of projected space (level, count_id)
    """
    return obs_to_level_int(obs, base, raw_dim, level, all_subspace)[count_id]


def obs_to_level_int_sp(obs, base, raw_dim, level):
    """
    Convert raw obs to level_int and permutation permutation invariant level_int
    """
    equivalent_rep = [obs_to_level_int(obs, base, raw_dim, level)]
    obs_eq = copy.deepcopy(obs)
    obs_eq[:2] = obs[2:4]
    obs_eq[2:4] = obs[:2]
    equivalent_rep.append(obs_to_level_int(obs_eq, base, raw_dim, level))
    return equivalent_rep


def int_to_obs(int_rep, base, raw_dim):
    """

    :param int_rep: int or 1-d numpy array
    :param base:
    :param raw_dim:
    :return: np array (bz, raw_dim)
    """
    if not isinstance(int_rep, np.ndarray):
        int_rep = np.array([int_rep])
    bz = int_rep.size
    obs = np.zeros((bz, raw_dim), dtype=np.int64)
    q = int_rep
    for d in range(raw_dim):
        q, r = divmod(q, base)
        obs[:, - (d + 1)] = r.reshape(-1)
    if raw_dim == 5:
        obs_first_col = copy.deepcopy(obs[:, 0])
        obs[:, [0,1,2,3]] = obs[:, [1,2,3,4]]
        obs[:, -1] = obs_first_col
    return obs



def obs_to_int_pi(obs, base, raw_dim):
    if raw_dim == 5:
        return obs[4] * base**4 + obs[0] * base**3 + obs[1] * base**2 + obs[2] * base + obs[3], \
               obs[4] * base**4 + obs[2] * base**3 + obs[3] * base**2 + obs[0] * base + obs[1]
    elif raw_dim == 6:
        return obs[0] * base**5 + obs[1] * base**4 + obs[2] * base**3 + obs[3] * base**2 + obs[4] * base + obs[5], \
               obs[2] * base**5 + obs[3] * base**4 + obs[0] * base**3 + obs[1] * base**2 + obs[4] * base + obs[5]
    elif raw_dim == 7:
        return obs[0] * base ** 6 + obs[1] * base ** 5 + obs[2] * base ** 4 + \
               obs[3] * base ** 3 + obs[4] * base ** 2 + obs[5] * base + obs[6], \
               obs[2] * base ** 6 + obs[3] * base ** 5 + obs[0] * base ** 4 + \
               obs[1] * base ** 3 + obs[4] * base ** 2 + obs[5] * base + obs[6]


def s_to_sp(s, base, raw_dim):
    s = convert_to_base(s, base)
    s_p_str = str(s).zfill(raw_dim)
    if raw_dim == 5:
        s_p = int(s_p_str[0], base) * base**4 + int(s_p_str[3], base) * base**3 + int(s_p_str[4], base) * base**2 + int(s_p_str[1], base) * base + int(s_p_str[2], base)
    elif raw_dim == 6:
        s_p = int(s_p_str[2], base) * base**5 + int(s_p_str[3], base) * base**4 + int(s_p_str[0], base) * base**3 + int(s_p_str[1], base) * base**2 + int(s_p_str[4], base) * base + int(s_p_str[5], base)
    elif raw_dim == 7:
        s_p = int(s_p_str[2], base) * base**6 + int(s_p_str[3], base) * base**5 + int(s_p_str[0], base) * base**4 \
              + int(s_p_str[1], base) * base**3 + int(s_p_str[4], base) * base ** 2 + int(s_p_str[5], base) * base + int(s_p_str[6], base)
    return s_p


def seperate_s_o(s_o, state_dim, agent_obs_dim):
    s = s_o[:, :state_dim]
    o1 = s_o[:, state_dim: state_dim + agent_obs_dim]
    o2 = s_o[:, state_dim + agent_obs_dim:]
    return s, o1, o2

