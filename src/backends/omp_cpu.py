import numpy as np

try:
    from numba import njit, prange
    NUMBA = True
except Exception:
    NUMBA = False
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap
    def prange(n): return range(n)

@njit(parallel=True, fastmath=True)
def update_agents(x, v, lane_len, dt, v_max, is_green, safe_gap, stops, at_stopline, headway):
    N = x.shape[0]
    for i in prange(N):
        # leader distance
        d_lead = headway[i]
        v_des = min(v[i] + 2.0*dt, v_max)  # simple accel 2 m/s^2
        # brake for leader
        if d_lead < 2*safe_gap:
            v_des = min(v_des, max(0.0, (d_lead - safe_gap)/dt))
        # brake for red signal if near stop line
        if (not is_green) and (x[i] < 5.0):  # 5m to stop line
            v_des = 0.0
            if not at_stopline[i]:
                stops[i] += 1
                at_stopline[i] = True
        else:
            at_stopline[i] = False
        x[i] = x[i] - v_des*dt
        v[i] = v_des
    return
