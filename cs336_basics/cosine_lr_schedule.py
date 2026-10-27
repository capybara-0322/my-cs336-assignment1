import math

def learning_rate_schedule(t, a_max, a_min, t_w, t_c):
    if t < t_w:
        return t * a_max / t_w
    if t >= t_w and t <= t_c:
        at = a_min + 0.5*(1+math.cos((t-t_w)*math.pi/(t_c-t_w))) * (a_max - a_min)
        return at
    else:
        at = a_min
        return at




