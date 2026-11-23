from dataclasses import dataclass
from enum import Enum
import circuit_template as cp
import numpy as np
from random import randint
from metric import Trajectory

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
        self.scenario = scenario
        self.conf_interval = interval
        self.seeds = [randint(0, 1000) for _ in range(20)]
        self.v_equilibre = 7
        self.t_end = 600
        self.t_w = 300
        self.dt = 0.1


        match scenario:
            case ScenarioType.S1:
                self.N = 22
                self.circ_stad = cp.make_circle(230.0)
                self.veh = [cp.place_vehicles(self.circ_stad, self.N, "uniform",
                                               self.N/230 -1, self.seeds[i]) for i in range(len(self.seeds))]
                for seed_vehicles in self.veh:
                        seed_vehicles["v"][:] = 0

            case ScenarioType.S2:
                max = 999 # faut calculer la saturation zebi, je sais pas encore comment choper cette D
                self.N = [randint(10, max) for _ in range(20)]
                self.circ_stad = cp.make_stadium()
                self.veh = [cp.place_vehicles(self.circ_stad, self.N[i], "random", 2.0, self.seeds[i]) for i in range(len(self.seeds))]

            case ScenarioType.S3:
                self.N = 40
                self.circ_stad = cp.make_stadium()
                self.veh = [cp.place_vehicles(self.circ_stad, self.N)]
                self.veh[0]["v"][:] = self.v_equilibre
                self.t_w = 200

        self.play_scenario()

    def simulation_loop(self, veh, times, positions, speeds, accelerations, gaps, collisions):
        #_____MASK_____
        lanes = []
        for direction in (+1, -1):
            for lane in range(cp.N_LANES):
                idx = np.flatnonzero((veh["direction"] == direction) & (veh["lane"] == lane))
                lanes.append(idx)

        for k, t in enumerate(times):
            for lane, indices in enumerate(lanes):
                #_____STEP_____
                s_new, v_new, infos = cp.step(veh["s"][indices], veh["v"][indices], self.dt, self.circ_stad, params)
                veh["s"][indices], veh["v"][indices] = s_new, v_new

                #_____UPDATING_THE_TABLES_____
                positions[k][lane]
                speeds[k][lane] = s_new
                accelerations[k][lane] = infos["a_eff"]
                gaps[k][lane] = infos["dist"]
                collisions += infos["impacts"]

        return positions, speeds, accelerations, gaps, collisions


    def play_scenario(self):
        '''NP for now, probably can do better time complexity (not a dictionary pro)'''

        simulations = [] # temp
        for i in range(len(self.veh)):

            # _____INIT TABLES_____
            times = np.arange(0.0, self.t_end + self.dt, self.dt) # nvm 0.1 in the jupyter

            T = len(times)

            positions = np.zeros((4, T, self.N))
            speeds = np.zeros((4, T, self.N))
            accelerations = np.zeros((4, T, self.N))
            gaps = np.zeros((4, T, self.N))
            collisions = [[] for _ in range(4)]

            # _____SIMULATION_____
            # calculate the mean of the data instead of storing for now, well do it later
            veh = self.veh[i]
            simulations.append(self.simulation_loop(veh, times, positions, speeds, accelerations, gaps, collisions))


        return [Trajectory(times,
            positions[lane],
            speeds[lane],
            accelerations[lane],
            gapes[lane],
            collisions[lane],
            230.0) for lane in range(4)]
# need change the 230 put actual L
    def print_results(self):
        pass
