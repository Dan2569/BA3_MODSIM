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
                N = 22
                self.veh = [cp.place_vehicles(cp.make_circle(230.0), N, "uniform",
                                               N/230 -1, self.seeds[i]) for i in range(len(self.seeds))]
                for seed_vehicles in self.veh:
                        seed_vehicles["v"][:] = 0

            case ScenarioType.S2:
                max = 999 # faut calculer la saturation zebi, je sais pas encore comment choper cette D
                N = [randint(10, max) for _ in range(20)]
                self.veh = [cp.place_vehicles(cp.make_stadium(), N[i], "random", 2.0, self.seeds[i]) for i in range(len(self.seeds))]

            case ScenarioType.S3:
                N = 40
                self.veh = cp.place_vehicles(cp.make_stadium(), N)
                self.veh["v"][:] = self.v_equilibre
                self.t_w = 200

        self.play_scenario()

    def play_scenario(self):
        
        self.trajectory = Trajectory()
        pass

    def print_results(self):
        pass
    
        
        
