import numpy as np
import matplotlib.pyplot as plt 

class waveformGenerator():
    def __init__(self) -> None:
        pass

    def pulses_constantVd(self, steps, Vd, Vw, tw, Vr, tr, td, Vh = 0, t_rise = 100e-9):
        steps = int(steps)
        drain = np.ones(steps) * Vd
        t_total = 10 * t_rise + 6 * td + 2 * tw + 3 * tr
        gate = np.zeros(steps)
        dt = t_total / steps
        tprev, tnext = 0, 0 + td
        gate[int(tprev//dt): int(tnext//dt)] = np.ones(int(tnext//dt - tprev//dt)) * Vh
        tprev, tnext = tnext, tnext + t_rise
        gate[int(tprev//dt): int(tnext//dt)] = np.linspace(Vh, -Vw, int(tnext//dt - tprev//dt))
        tprev, tnext = tnext, tnext + tw
        gate[int(tprev//dt): int(tnext//dt)] = np.ones(int(tnext//dt - tprev//dt)) * -Vw
        tprev, tnext = tnext, tnext + t_rise
        gate[int(tprev//dt): int(tnext//dt)] = np.linspace(-Vw, Vh, int(tnext//dt - tprev//dt))
        tprev, tnext = tnext, tnext + td
        gate[int(tprev//dt): int(tnext//dt)] = np.ones(int(tnext//dt - tprev//dt)) * Vh
        tprev, tnext = tnext, tnext + t_rise
        gate[int(tprev//dt): int(tnext//dt)] = np.linspace(Vh, Vr, int(tnext//dt - tprev//dt))
        tprev, tnext = tnext, tnext + tr
        gate[int(tprev//dt): int(tnext//dt)] = np.ones(int(tnext//dt - tprev//dt)) * Vr
        tprev, tnext = tnext, tnext + t_rise
        gate[int(tprev//dt): int(tnext//dt)] = np.linspace(Vr, Vh, int(tnext//dt - tprev//dt))
        tprev, tnext = tnext, tnext + td
        gate[int(tprev//dt): int(tnext//dt)] = np.ones(int(tnext//dt - tprev//dt)) * Vh
        tprev, tnext = tnext, tnext + t_rise
        gate[int(tprev//dt): int(tnext//dt)] = np.linspace(Vh, -Vw, int(tnext//dt - tprev//dt))
        tprev, tnext = tnext, tnext + tr
        gate[int(tprev//dt): int(tnext//dt)] = np.ones(int(tnext//dt - tprev//dt)) * -Vw
        tprev, tnext = tnext, tnext + t_rise
        gate[int(tprev//dt): int(tnext//dt)] = np.linspace(-Vw, Vh, int(tnext//dt - tprev//dt))
        tprev, tnext = tnext, tnext + td
        gate[int(tprev//dt): int(tnext//dt)] = np.ones(int(tnext//dt - tprev//dt)) * Vh
        tprev, tnext = tnext, tnext + tw
        gate[int(tprev//dt): int(tnext//dt)] = np.ones(int(tnext//dt - tprev//dt)) * Vw
        tprev, tnext = tnext, tnext + t_rise
        gate[int(tprev//dt): int(tnext//dt)] = np.linspace(Vw, Vh, int(tnext//dt - tprev//dt))
        tprev, tnext = tnext, tnext + td
        gate[int(tprev//dt): int(tnext//dt)] = np.ones(int(tnext//dt - tprev//dt)) * Vh
        tprev, tnext = tnext, tnext + t_rise
        gate[int(tprev//dt): int(tnext//dt)] = np.linspace(Vh, Vr, int(tnext//dt - tprev//dt))
        tprev, tnext = tnext, tnext + tr
        gate[int(tprev//dt): int(tnext//dt)] = np.ones(int(tnext//dt - tprev//dt)) * Vr
        tprev, tnext = tnext, tnext + t_rise
        gate[int(tprev//dt): int(tnext//dt)] = np.linspace(Vr, Vh, int(tnext//dt - tprev//dt))
        tprev, tnext = tnext, tnext + td
        gate[int(tprev//dt): int(tnext//dt)] = np.ones(int(tnext//dt - tprev//dt)) * Vh

        return gate, drain, t_total

# gen = waveformGenerator()
# g, d, t = gen.pulses_constantVd(1e5, 0.05, 2.5, 1e-6, 2, 3e-6, 2e-6)
# t = np.linspace(0, t, int(1e5))
# plt.plot(t, g)
# plt.plot(t, d)
# plt.show()