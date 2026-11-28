from dataclasses import dataclass
import circuit_template as cp
import numpy as np

A_MAX = 3.0   # limite d'accélération du simulateur [m/s²]
B_MAX = 9.0   # limite de décélération du simulateur [m/s²]

@dataclass
class Trajectory:
    t: np.ndarray          # (T,)
    x: np.ndarray          # (T, N) positions non repliées
    v: np.ndarray          # (T, N) vitesse
    a: np.ndarray          # (T, N) accélération réellement appliquée
    gap: np.ndarray        # (T, N) distances s_n pare-chocs à pare-chocs
    collisions: list       # [(t, n, v_rel), ...]
    L: float               # longueur du circuit
    l: float = cp.VEH_LENGTH   # longueur de véhicule

    def get_N(self):
        return self.x.shape[1]

    def get_s_e(self):
        return self.L / self.get_N() - self.l


    def _mask(self, t_w=None, t_end=None):
        t_w = self.t[0] if t_w is None else t_w
        t_end = self.t[-1] if t_end is None else t_end
        return (self.t >= t_w) & (self.t <= t_end)


    def flow_Q(traj, t_w, t_end):
        i0 = np.argmin(np.abs(traj.t - t_w))
        i1 = np.argmin(np.abs(traj.t - t_end))
        dx = traj.x[i1] - traj.x[i0]
        return 3600 / (traj.L * (traj.t[i1] - traj.t[i0])) * dx.sum()

    def collision_count(self):
        return len(self.collisions)


    def max_deceleration(self, t_w=None, t_end=None):
        m = self._mask(t_w, t_end)
        return np.max(np.maximum(0, -self.a[m]))


    def percentile_deceleration(self, t_w=None, t_end=None, q=99):
        m = self._mask(t_w, t_end)
        return np.percentile(np.maximum(0, -self.a[m]), q)


    def saturation_fraction(self, t_w=None, t_end=None):
        m = self._mask(t_w, t_end)
        return np.mean(np.isclose(self.a[m], -B_MAX))


    def equilibrium_distance(self):
        s_e = self.get_s_e()
        v_bar = self.v.mean(axis=1, keepdims=True)                  # (T, 1)
        A = (self.gap - s_e) / s_e                                  # (T, N)
        B = (self.v - v_bar) / np.maximum(v_bar, 1.0)               # (T, N)
        C = self.a / A_MAX                                          # (T, N), signe gardé

        return np.sqrt((A**2 + B**2 + C**2).mean(axis=1))           # (T,)


    def mean_equilibrium_distance(self, t_w, t_end):
        E = self.equilibrium_distance()
        mask = (self.t >= t_w) & (self.t <= t_end)
        return E[mask].mean()

    def impact_speeds(traj):
        return np.array([c[2] for c in traj.collisions])

    #made by AI
    def convergence_time(self, eps=1e-9,t_start=None):
        m = self._mask(t_start, None)
        E = self.equilibrium_distance()
        E = E[m]
        t = self.t[m]
        above = np.where(E >= eps)[0]       # indices où E est encore trop grand
        if len(above) == 0:
            return t[0]           # sous eps dès le départ
        if above[-1] == len(E) - 1:
            return np.inf                   # jamais stabilisé
        return t[above[-1] + 1]        # instant suivant le dernier dépassement

# ---------------- Tests ----------------
#---------------- made by AI ------------
if __name__ == "__main__":
    # --- Test 1 : débit ---
    # 4 véhicules, L = 100 m, tous avancent de 10 m en 1 s
    # Q = 3600 / (100 * 1) * (4 * 10) = 1440
    flow_traj = Trajectory(
        t=np.array([0., 1.]),
        x=np.array([[0., 0., 0., 0.],
                    [10., 10., 10., 10.]]),
        v=np.full((2, 4), 10.), #[[10., 10., 10., 10.],[10., 10., 10., 10.]]

        a=np.zeros((2, 4)),
        gap=np.full((2, 4), 20.),
        collisions=[],
        L=100.0,
    )
    q = flow_traj.flow_Q(0, 1)
    print("Q =", q, "(attendu 1440)")
    assert np.isclose(q, 1440.0)

    # --- Test 2 : débit avec une grille de temps plus fine ---
    # t = 0, 0.5, ..., 2 ; vitesse 10 m/s ; fenêtre [1, 2] -> 10 m par véhicule
    t = np.linspace(0, 2, 5)
    x = np.outer(t * 10, np.ones(4))
    flow_traj2 = Trajectory(
        t=t, x=x, v=np.full((5, 4), 10.), a=np.zeros((5, 4)),
        gap=np.full((5, 4), 20.), collisions=[], L=100.0,
    )
    assert np.isclose(flow_traj2.flow_Q(1, 2), 1440.0)

    # --- Test 3 : équilibre parfait -> E = 0 ---
    eq = Trajectory(
        t=np.array([0., 1.]), x=np.zeros((2, 4)),
        v=np.full((2, 4), 10.), a=np.zeros((2, 4)),
        gap=np.full((2, 4), 20.), collisions=[], L=100.0,
    )
    assert eq.get_s_e() == 20.0
    assert np.allclose(eq.equilibrium_distance(), 0.0)

    # --- Test 4 : collisions et vitesses d'impact ---
    eq.collisions = [(0.5, 1, 4.0), (0.8, 2, 2.0)]
    assert eq.collision_count() == 2
    assert np.allclose(eq.impact_speeds(), [4.0, 2.0])

    # --- Test 5 : décélération et saturation ---
    dec = Trajectory(
        t=np.array([0., 1., 2.]), x=np.zeros((3, 2)),
        v=np.zeros((3, 2)),
        a=np.array([[0., 0.], [-9., -3.], [1., 0.]]),
        gap=np.full((3, 2), 20.), collisions=[], L=40.0,
    )
    assert dec.max_deceleration() == 9.0
    assert np.isclose(dec.saturation_fraction(), 1 / 6)   # 1 case sur 6
    assert dec.max_deceleration(2, 2) == 0.0            # fenêtre sur le dernier pas

    # --- Test 6 : temps de convergence ---
    conv = Trajectory(
        t=np.arange(5.), x=np.zeros((5, 2)),
        v=np.array([[10., 6.], [10., 8.], [10., 10.], [10., 10.], [10., 10.]]),
        a=np.zeros((5, 2)), gap=np.full((5, 2), 20.), collisions=[], L=50.0,
    )
    # s_e = 50/2 - 5 = 20 ; E est > 0 aux pas 0 et 1, puis nul
    assert conv.convergence_time(eps=0.01) == 2.0

    print("Tous les tests passent")

#Etape 2:
#voir comment ça marche dans ipynb

#Etape 3:
#Comprendre cles fonction acceleration, gap, le moteur
#Comprendre les scénarios et les simulations

#Etape 4:
#Comprendre comment trouver les fonctions pour la simulation
