import math, random


history = {
    "turn": 0,
    "cnts": [0] * 100,
    "ocnts": [0] * 100,
    "hits": [0] * 100,
    "la": -1,
}

def agent(observation, configuration):
    global history

    N = 100
    p = [63.9413, 0.000533149]
    ti = observation.step
    if ti == 0:
        pass
    else:
        la = history["la"]
        ola = sum(observation.lastActions) - la
        if sum(history["hits"]) < observation.reward:
            history["hits"][la] += 1 / pow(0.97, history["cnts"][la] + history["ocnts"][la])
        history["cnts"][la] += 1
        history["ocnts"][ola] += 1

    tau = p[0] / (ti + 1) + p[1]
    ea = [0] * N
    hits = history["hits"]
    cnts = history["cnts"]
    ocnts = history["ocnts"]
    for i in range(N):
        if cnts[i] == 0:
            ea[i] = math.exp(0.99 * pow(0.97, ocnts[i]) / tau)
        else:
            ea[i] = math.exp(hits[i] / cnts[i] * pow(0.97, cnts[i] + ocnts[i]) / tau)

    se = sum(ea)
    r = random.random() * se
    t = 0
    la = 99
    for i in range(N):
        t += ea[i]
        if t >= r:
            la = i
            break

    history["la"] = la
    return la