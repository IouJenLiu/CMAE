DIGITS = '0123456789abcdefghijklmnopqrstuvwxyz'


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




