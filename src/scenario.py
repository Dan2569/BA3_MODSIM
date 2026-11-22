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
    

class Scenario:
    def __init__(self, interval = 0.99, scenario=ScenarioType.S1):
        self.scenario = scenario
        self.conf_interval = interval
        self.seeds = [randint(0, 1000) for _ in range(20)]
        self.v_equilibre = 7
        self.t_end = 600
        self.t_w = 300


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
                self.veh = cp.place_vehicles(self.circ_stad, self.N)
                self.veh["v"][:] = self.v_equilibre
                self.t_w = 200

        self.play_scenario()

    def simulation_loop(self, times, positions, speeds, accelerations, gaps, collisions):
        for dt in times:
            for direction in (+1, -1):
                for lane in range(cp.N_LANES):
                    mask = ((self.veh["direction"] == direction) & (self.veh["lane"] == lane))
                    indices = np.flatnonzero(mask)
                    s_new, v_new, info = cp.step(self.veh["s"][indices], self.veh["v"][indices], dt, self.circ_stad)
                    self.veh["s"][indices], self.veh["v"][indices] = s_new, v_new
        return positions, speeds, accelerations, gaps, collisions


    def play_scenario(self):
        '''NP for now, probably can do better time complexity (not a dictionary pro)'''
        
        simulations = [] # temp
        for i in range(len(self.veh)):
            # _____INIT TABLES_____
            times = np.arange(0.0, 600.0, 0.5) # step of time 0.5, dunno what is used in TPs
    
            T = len(times)
            N = self.N * 4
    
            positions = np.zeros((T, N))
            speeds = np.zeros((T, N))
            accelerations = np.zeros((T, N))
            gaps = np.zeros((T, N))
            collisions = []
            # _____SIMULATION_____
            # calculate the mean of the data instead of storing for now, well do it later
            simulations.append(self.simulation_loop(times, positions, speeds, accelerations, gaps, collisions))
        
        return Trajectory()

    def print_results(self):
        pass
    
        
        
