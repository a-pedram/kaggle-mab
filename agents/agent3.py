import math, random


history = {
    "turn": 0,
    "cnts": [0] * 100,
    "ocnts": [0] * 100,
    "hits": [0] * 100,
    "osteps": [0] * 100,
    "la": -1,
}

def agent(observation, configuration):
    global history

    N = 100
    p = [0.39918, 0.000138129, 1.23946]
    ti = observation.step
    if ti == 0:
        pass
    else:
        la = history["la"]
        ola = sum(observation.lastActions) - la
        history["osteps"][ola] = ti
        if sum(history["hits"]) < observation.reward:
            history["hits"][la] += 1 / pow(0.97, history["cnts"][la] + history["ocnts"][la])
        history["cnts"][la] += 1
        history["ocnts"][ola] += 1

    tau = p[0] / (ti + 1) + p[1]
    ea = [0] * N
    hits = history["hits"]
    cnts = history["cnts"]
    ocnts = history["ocnts"]
    osteps = history["osteps"]

    tv = sorted([(-ocnts[i], osteps[i], i) for i in range(N)])
    ot = [0] * N
    for i in range(N):
        ot[tv[i][2]] = 99 - i

    for i in range(N):
        if cnts[i] == 0:
            if ocnts[i] > 1:
                ea[i] = math.exp(min(500, ot[i] / 100 * pow(0.97, ocnts[i]) / tau))
            else:
                ea[i] = math.exp(min(500, 0.99 * pow(0.97, ocnts[i]) / tau))
        else:
            w = pow(cnts[i], p[2])
            wo = ocnts[i]
            if ocnts[i] < 2:
                wo = 0
            r = hits[i] / cnts[i]
            ro = ot[i] / 100
            ea[i] = math.exp(min(500, (r * w + ro * wo) / (w + wo) * pow(0.97, cnts[i] + ocnts[i]) / tau))

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