import argparse, math
import numpy as np
from model.roads import Topology
from model.signals import SignalPlan
from model.metrics import Metrics
from model.agents import Fleet, DIRECTIONS
from backends.omp_cpu import update_agents

def run(minutes=2, arrival_per_h=600, cycle=60, dt=0.2, seed=42):
    topo = Topology()
    plan = SignalPlan(cycle=cycle)
    rng = np.random.default_rng(seed)
    fleet = Fleet.empty(max_per_dir=2000)
    metrics = Metrics()
    T = int((minutes*60)/dt)

    lam = arrival_per_h/3600.0  # per second per approach
    p_spawn = lam*dt

    for step in range(T):
        t = step*dt
        # arrivals
        for d in DIRECTIONS:
            if rng.random() < p_spawn:
                # spawn only if head of queue is not too close to lane start
                fleet.spawn(d, topo.lane_length, t=t)

        # compute headways per direction
        head = {d: fleet.headways(d) for d in DIRECTIONS}

        # signal state
        green_ns = plan.is_green_ns(t)

        # update per direction
        for d in DIRECTIONS:
            n = fleet.active[d]
            if n == 0: 
                continue
            # EW is green when not NS
            is_green = (green_ns and d in ('N','S')) or ((not green_ns) and d in ('E','W'))
            update_agents(
                fleet.x[d][:n], fleet.v[d][:n],
                topo.lane_length, dt, topo.speed_limit,
                is_green, topo.safe_gap, fleet.stops[d][:n],
                fleet.at_stopline[d][:n], head[d]
            )

        # count departures & queues
        queues = {}
        for d in DIRECTIONS:
            departed_count = 0
            n = fleet.active[d]
            if n > 0:
                # usuwamy pojazdy, które wyjechały
                mask_departed = fleet.x[d][:n] <= 0.0
                for idx in np.where(mask_departed)[0]:
                    travel_time = t - fleet.birth_time[d][idx]
                    stops = fleet.stops[d][idx]
                    metrics.record_departure(travel_time, stops)
                    departed_count += 1
                if departed_count > 0:
                    # usuń ich fizycznie z tablic
                    fleet.remove_departed(d)
            # queue ~ number of cars within 25m of stop line with near-zero speed
            n = fleet.active[d]
            near = (fleet.x[d][:n] < 25.0) & (fleet.v[d][:n] < 0.1)
            queues[d] = int(near.sum())
        metrics.update_queue(queues)

    # finish: simple throughput estimate = departures per minute
    # We tracked departures implicitly via remove_departed, but didn't aggregate stops/travel times per-vehicle.
    # For the *showcase*, compute only throughput and max queues. Extend next time.
    sim_minutes = minutes
    summary = metrics.summary(sim_minutes)
    # Override departures with rough proxy: total cars that left the lane segments
    # (Our MVP didn't accumulate departures; set to 0 for now, throughput 0.)
    print("=== MVP metrics (to extend next lab) ===")
    print(summary)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=2.0)
    ap.add_argument('--arrival', type=float, default=600.0)
    ap.add_argument('--cycle', type=float, default=60.0)
    ap.add_argument('--dt', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    run(args.minutes, args.arrival, args.cycle, args.dt, args.seed)
