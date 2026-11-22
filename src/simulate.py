from dataclasses import dataclass
import circuit_template as cp
import numpy as np
from metric import Trajectory




def simulate(Circuit:cp.Circuit,N,layout,min_gap,seed,dt,t_w,t_end,params):
    n_steps = int(round(t_end/dt))
    s_hist = np.zeros((n_steps+1, N)) 
    v_hist = np.zeros((n_steps+1, N))
    a_hist = np.zeros((n_steps+1, N))
    gap_hist = np.zeros((n_steps+1, N))
    t_hist = np.linspace(0, t_end, n_steps + 1)
    collisions = list()
        
    dic = cp.place_vehicles(Circuit,N,layout,min_gap,seed)
    mask = (dic["direction"] == 1) & (dic["lane"] == 0)
    s_hist[0] = dic["s"][mask]
    v_hist[0] = dic["v"][mask]
    _, gap_hist[0], _, _ = cp.neighbours(dic,Circuit.L) 
    
    
    for i in range(n_steps+1):
        s_new, v_new, info = cp.step(s_hist[i],v_hist[i],dt,Circuit,params)
        s_hist[i+1] = s_new
        v_hist[i+1] = v_new
        a_hist[i+1] = info["a_eff"]
        collisions.append(info["impacts"])
        gap_hist[i+1] = info["gap"]
        
        
    return Trajectory(t_hist,s_hist,v_hist,a_hist,gap_hist,collisions,Circuit.L,cp.VEH_LENGTH)
        
        
        
"""
Demander à Stephh les position non repliées

"""