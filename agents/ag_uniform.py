def agent(obs, conf):
    return obs.step % conf.banditCount
