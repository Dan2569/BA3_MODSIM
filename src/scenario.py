from enum import Enum
from random import randint

import circuit_template as cp
import metric as mt
import numpy as np
from scipy import stats


class ScenarioType(Enum):
    S1 = "Anneau de Sugiyama"
    S2 = "Balayage en densité"
    S3 = "Freinage d'urgence"

class Strategy(Enum):
    maxVelocity = {} # max accelleration -> full brakes
    leastBreaks = {} # keep good distance to leader


params = {
    "v0"        : 7,
    "T"         : 0.1,
    "Direction" : +1,
    "Strategy"  : Strategy.maxVelocity
}


class Scenario:
    def __init__(self, scenario=ScenarioType.S1):
        self.scenario      = scenario
        self.seeds         = [randint(0, 1000) for _ in range(20)]
        self.v_equilibre   = 7 # still needs to be calculated
        self.t_end         = 600
        self.t_w           = 300
        self.dt            = 0.1
        self.N             = 0


        match scenario:
            case ScenarioType.S1:
                # 1m de perturbations toujours a ajouter et le min gap est WRONG
                self.N         = [22//(cp.N_LANES*2)]
                self.circ_stad = cp.make_circle(230.0)
                self.veh       = [cp.place_vehicles(self.circ_stad, self.N[0], "uniform",
                                               self.N[0]/230 -1, self.seeds[i]) for i in range(len(self.seeds))]
                for seed_vehicles in self.veh:
                        seed_vehicles["v"][:] = 0

            case ScenarioType.S2:
                min_gap        = 2.0
                self.circ_stad = cp.make_stadium()
                max            = int(self.circ_stad.L // (cp.VEH_LENGTH + min_gap))
                self.N         = [randint(10, max)//(cp.N_LANES*2) for _ in range(20)]
                self.veh       = [cp.place_vehicles(self.circ_stad, self.N[i], "random", 2.0, self.seeds[i]) for i in range(len(self.seeds))]

            case ScenarioType.S3:
                self.N         = [40//(cp.N_LANES*2)]
                self.circ_stad = cp.make_stadium()
                self.veh       = [cp.place_vehicles(self.circ_stad, self.N[0])]
                self.t_w       = 200

                self.veh[0]["acceleration"][:] = 0
                self.veh[0]["v"][:]            = self.v_equilibre

        self.play_scenario()


    def _simulation_loop(self, veh, times, positions, speeds, accelerations, gaps, collisions, strategy=Strategy.maxVelocity):
        '''Prends 1 seed de veh avec ses 4 voies et run la simulation, update les tableaux'''
        #_____MASK_____
        lanes = []
        for direction in (+1, -1):
            for lane in range(cp.N_LANES):
                idx = np.flatnonzero((veh["direction"] == direction) & (veh["lane"] == lane))
                lanes.append(idx)

        for k, t in enumerate(times):
            for lane, indices in enumerate(lanes):
                #_____STEP_____
                if 200.0 <=t<= 203.0 and self.scenario == ScenarioType.S3:
                    accelerations[lane][0] = -cp.BMAX #maybe work actually
                # passed the dirsction to step, needs to handle the direction in wich to step (and add it in argument too)
                params["T"] = t
                params["direction"] = veh["direction"][indices[0]]
                s_new, v_new, infos = cp.step(veh["s"][indices], veh["v"][indices], self.dt, self.circ_stad, params)
                veh["s"][indices], veh["v"][indices] = s_new, v_new

                #_____UPDATING_THE_TABLES_____
                positions[lane, k]     = positions[lane, k - 1] + infos["dist"]
                speeds[lane][k]        = v_new
                accelerations[lane][k] = infos["a_eff"]
                gaps[lane][k]          = infos["gap"] # a ajouter dans step, pas ici de le faire
                collisions[lane].append(infos["impacts"]) # fix the impact output in step to match dan version of trajectory

        return times, positions, speeds, accelerations, gaps, collisions


    def play_scenario(self, confidence_level=0.95, strategy=Strategy.maxVelocity):
        '''NP for now, probably can do better time complexity (not a dictionary pro) -> hehe fixed ;)'''

        simulations = [] # temp
        times = np.arange(0.0, self.t_end + self.dt, self.dt) #need to change this line, but thats how i pass times to the helop function for now
        for simulation in self.veh:

            # _____INIT TABLES_____
            # not up to date since now can overtake -> not fixed number of cars on one lane then
            times = np.arange(0.0, self.t_end + self.dt, self.dt)
            T = len(times)
            N = len(simulation["s"]) // 4

            positions     = np.zeros((4, T, N))
            speeds        = np.zeros((4, T, N))
            accelerations = np.zeros((4, T, N))
            gaps          = np.zeros((4, T, N))
            collisions    = [[] for _ in range(4)]

            # _____SIMULATION_____
            simulations.append(self._simulation_loop(simulation, times, positions, speeds, accelerations, gaps, collisions, strategy))

        self.results = self.compute_results(simulations, times, confidence_level)
        self.print_results(self.results)
        return self.results


    def compute_results(self, simulations, times, confidence_level):
        '''build the trajectories to extract the data from the simulations, then mean it to produce one piece of result'''
        #_____INIT_TABLES_____
        Q     = np.zeros((len(simulations), 4))
        C     = np.zeros((len(simulations), 4))
        D_max = np.zeros((len(simulations), 4))
        D_99  = np.zeros((len(simulations), 4))
        sat   = np.zeros((len(simulations), 4))
        E_bar = np.zeros((len(simulations), 4))

        #_____BUILD_TRAJECTORIES_____
        trajectories = [
            [mt.Trajectory(times, pos[lane], spd[lane], acc[lane], gap[lane], col[lane], self.circ_stad.L)
                for lane in range(4)]
            for times, pos, spd, acc, gap, col in simulations]

        #_____BUILD_DATA_FROM_TRAJECTORIES_____
        for i, row in enumerate(trajectories):
            for j, traj in enumerate(row):
                Q[i, j]     = mt.flow_Q(traj, self.t_w, self.t_end)
                C[i, j]     = mt.collision_count(traj)
                D_max[i, j] = mt.max_deceleration(traj, self.t_w, self.t_end)
                D_99[i, j]  = mt.percentile_deceleration(traj, self.t_w, self.t_end)
                sat[i, j]   = mt.saturation_fraction(traj, self.t_w, self.t_end)
                E_bar[i, j] = mt.mean_equilibrium_distance(traj, self.t_w, self.t_end)
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
            return metric, 0.0

        mean = np.mean(metric, axis=0)
        #ddof=1 -> Bessel's correction aaskip si ddof=0 -> underestimates the true variance
        std  = np.std(metric, axis=0, ddof=1)
        #stats.t.ppf to calculate the  the Z for a given confidence level with n-1 degrees of freedom (geeksforgeeks)
        error_margin = stats.t.ppf((1.0 + cl) / 2.0, df=n - 1) * (std / np.sqrt(n))

        return mean, error_margin


    def print_results(self, data):
        for key, (mean, error_margin) in data.items():
            print(f"{key}: {mean} ± {error_margin} = [{mean - error_margin}, {mean + error_margin}] \n")
