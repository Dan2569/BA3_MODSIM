from enum import Enum
from random import randint,sample

import circuit_template as cp
import metric as mt
import numpy as np
from scipy import stats


class ScenarioType(Enum):
    S1 = "Anneau de Sugiyama"
    S2 = "Balayage en densité"
    S3 = "Freinage d'urgence"




class Scenario:
    def __init__(self, scenario=ScenarioType.S1, single_lane=True):
        self.scenario      = scenario
        self.single_lane   = single_lane if scenario == ScenarioType.S1 else False
        self.n_lanes       = 1 if self.single_lane else 2*cp.N_LANES
        self.seeds         = list(range(2)) #TODO: Remettre à 20 une fois les tests validés
        self.v_equilibre   = 7
        self.t_end         = 600
        self.t_w           = 300
        self.dt            = 0.1
        self.params        = {"v0": self.v_equilibre,               # vitesse désirée (d'équilibre) (m/s)
                              "s0": 2.0,                            # distance pare-chocs (m)
                              "T": 1.5,                             # temps inter-véhiculaire (s)
                              "a_max": 3,                           # accélération max (m/s^2)
                              "b": 9,                               # décélération de confort (m/s^2)
                              "delta": 4.0,                         # Exposant d'accélération (loi IDM)
                              "t": 0.0,                             # temps de simulation (s)
                              "direction": +1,
                              "strategy": cp.Strategy.dampStopNGo,
                              "forced": {}}        
        match scenario:
            case ScenarioType.S1:
                # 1m de perturbations toujours a ajouter et le min gap est WRONG
                self.circ_stad = cp.make_circle(230.0)
                self.N         = 22
                self.veh       = []
                for seed in self.seeds:
                    veh        = cp.place_vehicles(self.circ_stad, 22, "uniform", 0, seed) #J'ai 0 de manière arbitraire, il est pas utilisé dans le cas de uniform                        
                    veh["v"][:]     = 0.0
                    rng             = np.random.default_rng(seed)
                    veh["s"]        = np.mod(veh["s"] + rng.uniform(-1, 1, size=len(veh["s"])), self.circ_stad.L)
                    self.veh.append(veh)
            case ScenarioType.S2:
                min_gap        = 2.0
                self.circ_stad = cp.make_stadium()
                max            = int(self.circ_stad.L // (cp.VEH_LENGTH + min_gap))
                self.N         = [randint(10, max)//(cp.N_LANES*2) for _ in range(20)]
                self.veh       = [cp.place_vehicles(self.circ_stad, self.N[i], "random", 2.0, self.seeds[i]) for i in range(len(self.seeds))]

            case ScenarioType.S3:
                self.N              = [40//(cp.N_LANES*2)]
                self.circ_stad      = cp.make_stadium()
                self.veh            = [cp.place_vehicles(self.circ_stad, self.N[0])]
                self.t_w            = 200
                self.veh[0]["v"][:] = self.v_equilibre

        self.play_scenario()


    def _simulation_loop(self, veh, times, positions, speeds, accelerations, gaps, collisions, strategy=cp.Strategy.dampStopNGo):
        '''Prends 1 seed de veh avec ses 4 voies et run la simulation, update les tableaux'''
        #_____MASK_____
        lanes = []
        if self.single_lane:
            # Seulement la voie 0 dans le sens direct
            idx = np.flatnonzero((veh["direction"] == +1) & (veh["lane"] == 0))
            lanes.append(idx)
        else:
            for direction in (+1, -1):
                for lane in range(cp.N_LANES):
                    idx = np.flatnonzero((veh["direction"] == direction) & (veh["lane"] == lane))
                    lanes.append(idx)

        for k, t in enumerate(times):
            for lane, indices in enumerate(lanes):
                #_____STEP_____
                if 200.0 <=t<= 203.0 and self.scenario == ScenarioType.S3:
                    self.params["forced"] = {0: -cp.BMAX}
                else:
                    self.params["forced"] = {}
                # passed the direction to step, needs to handle the direction in which to step (and add it in argument too)
                self.params["t"] = t
                self.params["direction"] = veh["direction"][indices[0]]
                s_new, v_new, infos = cp.step(veh["s"][indices], veh["v"][indices], self.dt, self.circ_stad, self.params)
                veh["s"][indices], veh["v"][indices] = s_new, v_new

                #_____UPDATING_THE_TABLES_____
                positions[lane, k]     = positions[lane, k - 1] + infos["dist"]
                speeds[lane][k]        = v_new
                accelerations[lane][k] = infos["a_eff"]
                gaps[lane][k]          = infos["gap"] # a ajouter dans step, pas ici de le faire
                if len(infos["impacts"]) > 0: collisions[lane].extend(infos["impacts"])  # fix the impact output in step to match dan version of trajectory

        return times, positions, speeds, accelerations, gaps, collisions


    def play_scenario(self, confidence_level=0.95, strategy=cp.Strategy.dampStopNGo):
        '''NP for now, probably can do better time complexity (not a dictionary pro) -> hehe fixed ;)'''

        simulations = [] # temp
        times = np.arange(0.0, self.t_end + self.dt, self.dt) #need to change this line, but thats how i pass times to the helop function for now
        for simulation in self.veh:

            # _____INIT TABLES_____
            # not up to date since now can overtake -> not fixed number of cars on one lane then
            times = np.arange(0.0, self.t_end + self.dt, self.dt)
            T = len(times)
            N = len(simulation["s"]) // 4

            positions     = np.zeros((self.n_lanes, T, N))
            speeds        = np.zeros((self.n_lanes, T, N))
            accelerations = np.zeros((self.n_lanes, T, N))
            gaps          = np.zeros((self.n_lanes, T, N))
            collisions    = [[] for _ in range(self.n_lanes)]

            # _____SIMULATION_____
            simulations.append(self._simulation_loop(simulation, times, positions, speeds, accelerations, gaps, collisions, strategy))

        self.results = self.compute_results(simulations, times, confidence_level)
        self.print_results(self.results)
        return self.results


    def compute_results(self, simulations, times, confidence_level):
        '''build the trajectories to extract the data from the simulations, then mean it to produce one piece of result'''
        #_____INIT_TABLES_____
        Q     = np.zeros((len(simulations), self.n_lanes))
        C     = np.zeros((len(simulations), self.n_lanes))
        D_max = np.zeros((len(simulations), self.n_lanes))
        D_99  = np.zeros((len(simulations), self.n_lanes))
        sat   = np.zeros((len(simulations), self.n_lanes))
        E_bar = np.zeros((len(simulations), self.n_lanes))

        #_____BUILD_TRAJECTORIES_____
        trajectories = [
            [mt.Trajectory(times, pos[lane], spd[lane], acc[lane], gap[lane], col[lane], self.circ_stad.L)
                for lane in range(self.n_lanes)]
            for times, pos, spd, acc, gap, col in simulations]

        #_____BUILD_DATA_FROM_TRAJECTORIES_____
        for i, row in enumerate(trajectories):
            for j, traj in enumerate(row):
                Q[i, j]     = traj.flow_Q(self.t_w, self.t_end)
                C[i, j]     = traj.collision_count()
                D_max[i, j] = traj.max_deceleration(self.t_w, self.t_end)
                D_99[i, j]  = traj.percentile_deceleration(self.t_w, self.t_end)
                sat[i, j]   = traj.saturation_fraction(self.t_w, self.t_end)
                E_bar[i, j] = traj.mean_equilibrium_distance(self.t_w, self.t_end)
                # convergence time for s3, need find epsilon (dunno)
                # E_barre/N pour S2, pas sur de comment faire pour celui ci, faire le plotlib dans cette methode i guess (probleme de demain anyway)

        return {"Q"    : self._mean_metric(Q, confidence_level),
                "C"    : self._mean_metric(C, confidence_level),
                "D_max": self._mean_metric(D_max, confidence_level),
                "D_99" : self._mean_metric(D_99, confidence_level),
                "sat"  : self._mean_metric(sat, confidence_level),
                "E_bar": self._mean_metric(E_bar, confidence_level)}


    def _mean_metric(self, metric, cl):
        n    = metric.shape[0]
        if n < 2:
            return metric[0], np.zeros(metric.shape[1])

        mean = np.mean(metric, axis=0)
        #ddof=1 -> Bessel's correction aaskip si ddof=0 -> underestimates the true variance
        std  = np.std(metric, axis=0, ddof=1)
        #stats.t.ppf to calculate the  the Z for a given confidence level with n-1 degrees of freedom (geeksforgeeks)
        error_margin = stats.t.ppf((1.0 + cl) / 2.0, df=n - 1) * (std / np.sqrt(n))

        return mean, error_margin


    def print_results(self, data):
        print(f"=== {self.scenario.value} ===")
        if self.single_lane:
            print("\n--- Voie unique (Direction +1, Lane 0) ---")
            for key, (mean, err) in data.items():
                m, e = mean[0], err[0]
                print(f"    {key:6s}: {m:8.3f} ± {e:7.3f}  [{m - e:8.3f}, {m + e:8.3f}]")
        else:
            idx = 0
            for direction in (+1, -1):
                print(f"\n--- Direction {direction:+d} ---")
                for lane in range(cp.N_LANES):
                    print(f"  Lane {lane}:")
                    for key, (mean, err) in data.items():
                        m, e = mean[idx], err[idx]
                        print(f"    {key:6s}: {m:8.3f} ± {e:7.3f}  [{m - e:8.3f}, {m + e:8.3f}]")
                    idx += 1
