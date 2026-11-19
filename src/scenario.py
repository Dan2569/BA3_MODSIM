from dataclasses import dataclass
from enum import Enum
import circuit_template as cp
import numpy as np
from random import randint

class ScenarioType(Enum):
    S1 = "Anneau de Sugiyama"
    S2 = "Balayage en densité"
    S3 = "Freinage d'urgence"
    

class Scenario:
    def __init__(self, interval, scenario):
        self.scenario = scenario
        self.conf_interval = interval
        self.seeds = [randint(0, 1000) for _ in range(20)]

        match scenario:
            case ScenarioType.S1:
                N = 22
                self.veh = ["circle", cp.make_circle(), N, ]
            case ScenarioType.S2:
                N = ?
                self.veh = ["circle", cp.make_stadium(), N, ]
            case ScenarioType.S3:
                N = ?
                self.veh = ["circle", cp.make_stadium(), N, ]
    
        
        
