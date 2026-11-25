from dataclasses import dataclass
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

params = {
    "v0": 15.0,
    "s0": 1.0,
    "T": 1.5
}

class Scenario:
    def __init__(self, interval = 0.99, scenario=ScenarioType.S1):
        self.scenario      = scenario
        self.conf_interval = interval
        self.seeds         = [randint(0, 1000) for _ in range(20)]
        self.v_equilibre   = 7 # still needs to be calculated
        self.t_end         = 600
        self.t_w           = 300
        self.dt            = 0.1
        self.N             =  0


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
                self.N              = [40//(cp.N_LANES*2)]
                self.circ_stad      = cp.make_stadium()
                self.veh            = [cp.place_vehicles(self.circ_stad, self.N[0])]
                self.veh[0]["v"][:] = self.v_equilibre
                self.t_w            = 200

        self.play_scenario()

    def _simulation_loop(self, veh, times, positions, speeds, accelerations, gaps, collisions):
        '''Prends 1 seed de veh avec ses 4 voies et run la simulation, update les tableaux'''
        #_____MASK_____
        lanes = [] # pas sur que la boucle step fonctionen corretement dans les 2 sens
        for direction in (+1, -1):
            for lane in range(cp.N_LANES):
                idx = np.flatnonzero((veh["direction"] == direction) & (veh["lane"] == lane))
                lanes.append(idx)

        for k, t in enumerate(times):
            for lane, indices in enumerate(lanes):
                #_____STEP_____
                s_new, v_new, infos = cp.step(veh["s"][indices], veh["v"][indices], self.dt, self.circ_stad, params)
                veh["s"][indices], veh["v"][indices] = s_new, v_new

                if 200.0 <=t<= 203.0 and self.scenario == ScenarioType.S3:
                    accelerations[lane][0] = -cp.BMAX #wont work i think, will fix later

                #_____UPDATING_THE_TABLES_____
                positions[lane, k]     = positions[lane, k - 1] + infos["dist"]
                speeds[lane][k]        = v_new
                accelerations[lane][k] = infos["a_eff"]
                gaps[lane][k]          = infos["gap"] # a ajouter dans step, pas ici de le faire
                collisions[lane].append(infos["impacts"]) # fix the impact output in step to match dan version of trajectory

        return times, positions, speeds, accelerations, gaps, collisions


    def play_scenario(self):
        '''NP for now, probably can do better time complexity (not a dictionary pro) -> hehe fixed ;)'''

        simulations = [] # temp
        times = np.arange(0.0, self.t_end + self.dt, self.dt) #need to change this line, but thats how i pass times to the helop function for now
        for simulation in self.veh:

            # _____INIT TABLES_____
            # not up to date since now can ivertake -> not fixed number of cars on one lane then
            times = np.arange(0.0, self.t_end + self.dt, self.dt) # nvm 0.1 in the jupyter
            T = len(times)
            N = len(simulation["s"])

            positions     = np.zeros((4, T, N))
            speeds        = np.zeros((4, T, N))
            accelerations = np.zeros((4, T, N))
            gaps          = np.zeros((4, T, N))
            collisions    = [[] for _ in range(4)]

            # _____SIMULATION_____
            simulations.append(self._simulation_loop(simulation, times, positions, speeds, accelerations, gaps, collisions))

        self.results = self.compute_results(simulations, times, self.conf_interval)
        self.print_results(self.results)
        return self.results


    def compute_results(self, simulations, times, ci=0.99):
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
                # convergence time for s3
                # E_barre/N pour S2, pas sur de comment faire pour celui ci, faire le plotlib dans cette methode i guess (probleme de demain anyway)

        return {"Q"    : self._mean_metric(Q, ci),
                "C"    : self._mean_metric(C, ci),
                "D_max": self._mean_metric(D_max, ci),
                "D_99" : self._mean_metric(D_99, ci),
                "sat"  : self._mean_metric(sat, ci),
                "E_bar": self._mean_metric(E_bar, ci)}


    def _mean_metric(self, metric, ci=0.99):
        #ohh still need to add the ci actually my bad
        n    = metric.shape[0]
        mean = np.mean(metric, axis=0)
        std  = np.std(metric, axis=0)
        # needs to use calculate the Zalpha/2
        # stats from scity to do so
        lower = mean - 1.960 * (std/np.sqrt(n))
        upper = mean + 1.960 * (std/np.sqrt(n))

        interval       = (metric>lower) & (metric<upper)
        confident_mean = np.mean(metric[interval])
        # hyper raciste, peut etre ca matrche, peut etre j'ai rien compris on verra
        return confident_mean

    def print_results(self, data):
        for key, value in data.items():
            print(f"{key}: {value}\n")
