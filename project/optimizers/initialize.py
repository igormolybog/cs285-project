def default_reward_table(shape):
    '''
        call r_rable.cast(default_reward_table(shape))
    '''
    prod = lambda s: reduce(lambda a, b: a*b, s)
    list_to_rewtable = lambda x: np.array(x).reshape(shape)
    init_rew_list = [-0.1/prod(shape[:-1])]*(prod(shape)-shape[-1])+[1]*shape[-1]
    return list_to_rewtable(init_rew_list)
