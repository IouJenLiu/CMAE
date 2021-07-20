from env.env_factory import get_env



def make_env(env_name, seed, rank):
    def _thunk():
        env = get_env(env_name)()
        env.seed(rank + seed)
        return env

    return _thunk