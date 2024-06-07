#import
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import stats
import scipy.signal
np.random.seed(1000000)

path = "./plots/test"
iter = 0

#Fundamental Constants
q = 1.602177e-19           #Charge of electron in C
k_B = 1.380649e-23         #Boltzmann constant in J/K
e0 = 8.854e-12             #Vacuum permittivity in As/Vm
T = 300                    #absolute temperature in K

#Semiconductor parameters
n_i = 1e10*1e6             # intrinsic carrier concentration in m^-3
N_A = 2e17*1e6             # Acceptor concentration in m^-3
N_D = 0                    # Donor concentration in m^-3
mu_n = 10*1e-4             # electron mobility in m^2/Vs
eps_s = e0*11.9            # Permittivitsy of silicon in As/Vm

#MOSFET parameters
L = 1e-6                 # Gate length in m
W = 1e-6                   # Gate width in m
EOT = 1e-9                 # Equivalent gate oxide thickness in m
VFB = -0.3                 # Flatband voltage in V
VDS = 0.05                 # Drain-Source voltage in V
iG0 = 10e4                 # MOS leakage current in A/m2 at VG = 1 V

#MFM parameters
A_MFM = (1e-6)**2          # Total MFM capacitor area in m^s2
d = 4.5e-9                 # Ferroelectric thickness in m
Pr = 0.025                 # Remanent polarization in C/m2
Ec = 0.9e8                 # Coercive field in V/m
s = 0.2                     # Grain variation sigma/mu in %
rho_0 = 400                # Single domain internal resistance in Ohm m
iL0 = 1.8e3                # MFM leakage current at VF = 0V in A/m2
VL0 = 0.5                  # MFM leakage normalization voltage in V 
Psign = -1                 # Initial polarization direction
N = 200                    # Number of grains
A = A_MFM/N                # Area of each grain in m^2

Idatadf = pd.read_csv("_10x10um_cell3-13_IgVg_1e0.csv", header = 256)
keys = [k for k in Idatadf]
IGdatadf = np.array(Idatadf[keys[2]])
Idata = IGdatadf*1e4*3
VG = np.array([Idatadf[keys[1]]])
VG.shape = (302,)


K, J = 10, 10
IDmean1 = np.zeros((K,J), dtype=np.longdouble)
IDmean2 = np.zeros((K,J), dtype=np.longdouble)
Qfinal1 = np.zeros((K,J), dtype=np.longdouble)
Qfinal2 = np.zeros((K,J), dtype=np.longdouble)
V_read = np.zeros((K,1), dtype=np.longdouble)
T_read = np.zeros((J,1), dtype=np.longdouble)

#Grain parameter mean values 
mu_rho = np.ones((1,N))*rho_0*d/A_MFM*N                  #Internal resistance in Ohm
mu_Vbias = np.ones((1,N))*0                              #Internal bias voltage in V
mu_a1 = np.ones((1,N))*(-3*np.sqrt(3)*Ec/4/Pr)*1.0       #1st Landau constant in m/F
mu_a11 = (-1*mu_a1/2/Pr**2)*1.0                          #2nd Landau constant in m^5/(F*C^2)
mu_a111 = np.ones((1,N))*0                               #3rd Landau constant in m^9/(F*C^4)

#Grain parameter standard deviations
sigma_rho = np.ones((1,N))*rho_0*N*d/A_MFM*s    #Internal resistance in Ohm
sigma_Vbias = np.ones((1,N))*1*s                #Internal bias voltage in V
sigma_a1 = np.abs(mu_a1)*s                      #1st Landau constant in m/F
sigma_a11 = mu_a11*s                            #2nd Landau constant in m^5/(F*C^2)
sigma_a111 = np.ones((1,N))*0   

#Create Normal Distributed Fitting Parameters
rho = np.abs(np.random.normal(mu_rho,sigma_rho))
Vbias = np.random.normal(mu_Vbias,sigma_Vbias)
a1 = -np.abs(np.random.normal(mu_a1,sigma_a1))
a11 = np.abs(np.random.normal(mu_a11,sigma_a11))
a111 = np.abs(np.random.normal(mu_a111,sigma_a111))

#Normalize Landau parameters to grain geometry
alpha = d*a1/A              #1st Landau constant in V/C
beta  = d*a11/(A**3)        #2nd Landau constant in V/C^3
gamma = d*a111/(A**5)       #3rd Landau constant in V/C^5


for k in range(0, K):
    print(k)
    for j in range(0, J):
        #Voltage Waveform
        Vread = 0.2 * (k + 1)                #Read voltage in V
        Vprg = 2                    #PRG voltage in V
        Twait = 20e-6               #Wait time between pulses in s
        Trf = 5e-8                  #Rise/fall times in s
        Tprg = 10e-6                #PRG pulse width in s    
        Tread = 1e-7 * (2) ** ((j + 1) -1)      #Read pulse width in s
        T_total = 4*Twait + 6*Trf + Tprg + 2*Tread
        #T_total = 3e-6

        #Simulation parameters
        Cp = 0e-12                     #Parasitic capacitance in F
        pts = 30000                     #Number of simulation points
        dt = T_total/pts                #Simulation time steps in s
        #Useful parameters
        A_MOS = L*W                                        # Total Gate area in m^2
        b = q/k_B/T                                        # Inverse thermal voltage in V^-1
        Psi_B = 1/b*np.log(N_A/n_i)                           # Onset of strong inversion in V
        p_p0 = ((N_A-N_D)+np.sqrt((N_A-N_D)**2+4*n_i**2))/2     # Equilibrium hole concentration in m^-3
        n_p0 = n_i**2/p_p0                                  # Equilibrium electron concentration in m^-3
        L_D = np.sqrt(eps_s/(q*p_p0*b))                       # Extrinsic Debye-length for holes in m                 
        Cox = 3.9*8.854e-12/EOT                            # Gate oxide capacitance in F/m2
        VT = VFB + 2*Psi_B + np.sqrt(2*eps_s*q*N_A*2*Psi_B)/Cox   # threshold voltage in V

        #Build voltage waveform
        V = np.zeros((pts,1))
        t = np.arange(0, (pts - 1)*dt + dt, dt)
        
        try:
            V[int(Twait/dt):int((Twait+Trf)/dt)] = np.arange(0, Vread, Vread/(Trf/dt)).reshape((-1, 1))
        except:
            V[int(Twait/dt):int((Twait+Trf)/dt)+1] = np.arange(0, Vread, Vread/(Trf/dt)).reshape((-1, 1))
        V[int((Twait+Trf)/dt):int((Twait+Trf+Tread)/dt)] = Vread
        try:
            V[int((Twait+Trf+Tread)/dt)+1:int((Twait+2*Trf+Tread)/dt)] = np.arange(Vread, Vread/(Trf/dt), -Vread/(Trf/dt)).reshape(-1, 1)
        except:
            V[int((Twait+Trf+Tread)/dt)+1:int((Twait+2*Trf+Tread)/dt)+1] = np.arange(Vread, Vread/(Trf/dt), -Vread/(Trf/dt)).reshape(-1, 1)
        try:
            V[int((2*Twait+2*Trf+Tread)/dt):int((2*Twait+3*Trf+Tread)/dt)] = np.arange(0, Vprg, Vprg/(Trf/dt)).reshape((-1, 1))
        except:
            V[int((2*Twait+2*Trf+Tread)/dt):int((2*Twait+3*Trf+Tread)/dt)+1] = np.arange(0, Vprg, Vprg/(Trf/dt)).reshape((-1, 1))
        V[int((2*Twait+3*Trf+Tread)/dt):int((2*Twait+3*Trf+Tread+Tprg)/dt)] = Vprg
        try:
            V[int((2*Twait+3*Trf+Tread+Tprg)/dt):int((2*Twait+4*Trf+Tread+Tprg)/dt)] = np.arange(Vprg, 0, -Vprg/(Trf/dt)).reshape((-1, 1))
        except:
            V[int((2*Twait+3*Trf+Tread+Tprg)/dt):int((2*Twait+4*Trf+Tread+Tprg)/dt)+1] = np.arange(Vprg, 0, -Vprg/(Trf/dt)).reshape((-1, 1))
        try:
            V[int((3*Twait+4*Trf+Tread+Tprg)/dt):int((3*Twait+5*Trf+Tread+Tprg)/dt)] = np.arange(0, Vread, Vread/(Trf/dt)).reshape((-1, 1))
        except:
            V[int((3*Twait+4*Trf+Tread+Tprg)/dt):int((3*Twait+5*Trf+Tread+Tprg)/dt)+1] = np.arange(0, Vread, Vread/(Trf/dt)).reshape((-1, 1))
        V[int((3*Twait+5*Trf+Tread+Tprg)/dt):int((3*Twait+5*Trf+2*Tread+Tprg)/dt)] = Vread
        try:
            V[int((3*Twait+5*Trf+2*Tread+Tprg)/dt):int((3*Twait+6*Trf+2*Tread+Tprg)/dt)] = np.arange(Vread, 0, -Vread/(Trf/dt)).reshape((-1, 1))  
        except:
            V[int((3*Twait+5*Trf+2*Tread+Tprg)/dt):int((3*Twait+6*Trf+2*Tread+Tprg)/dt)+1] = np.arange(Vread, 0, -Vread/(Trf/dt)).reshape((-1, 1))  

        #Switching
        # V[:4750] = np.zeros((4750, 1))
        # V[4750:5000] = np.linspace(0, 3, 250).reshape(250, 1)
        # V[5000: 20000] = 3 * np.ones((15000, 1))
        # V[20000:20250] = np.linspace(3, 0, 250).reshape(250, 1)
        # V[20250:] = np.zeros((9750, 1))
        
        #Read Waveform
        # V[:5000] = np.zeros((5000, 1))
        # V[5000: 10000] = 1.5 * np.ones((5000, 1))
        # V[15000:] = np.zeros((15000, 1))

        #PWD Waveform
        # V[:5000] = np.zeros((5000, 1))
        # tp = 0.33e-6
        # V[5000: 5000 + int(tp/dt)] = 2.5*np.ones((int(tp/dt), 1))
        # V[5000 + int(tp/dt):] = np.zeros((25000 - int(tp/dt), 1))

        #Sweep
        #V = np.linspace(-5, 5, 30000)

        #Sawtooth
        #V = 2 * scipy.signal.sawtooth(2 * np.pi * t * 1e6, 0.5)

        if k == 0 and j == iter:
            plt.figure()
            plt.plot(t, V)
            plt.title("Voltage waveform")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.savefig(path + "/Voltage Wavevorm")
            plt.close()
            # plt.show()


        #Reference semiconductor charge vs. surface potential
        Psi_S = np.arange(-2, 2 + 0.001, 0.001)                #Semiconductor surface potential in V
        F = np.sqrt(np.abs((np.exp(-b*Psi_S)+b*Psi_S-1)+n_p0/p_p0*np.exp(-b*VDS)*(np.exp(b*Psi_S)-b*Psi_S*np.exp(b*VDS)-1)))*np.sign(Psi_S)
        Qs = np.real(np.sqrt(2)*eps_s/b/L_D*F)       # Total charge at the surface of the semiconductor in C/m^2
        Vox = Qs/Cox                           # Voltage drop across the gate oxide in V
        Qp = (Psi_S+Vox)*Cp                    # Parasitic charge parallel to Gate capacitance in C
        Qps = Qp + Qs*A_MOS                    # Sum of parasitic and Gate charge in C
        C_ox = np.diff(Qs)/np.diff(Psi_S+Vox+VFB)   # Theoretical Gate capacitance in F/m2



        # Initializing simulation variables
        QF = np.zeros((pts,N), dtype=np.longdouble)
        QL = np.zeros((pts,1), dtype=np.longdouble)
        QFall = np.zeros((pts,1), dtype=np.longdouble)
        QS = np.zeros((pts,1), dtype=np.longdouble)
        Vgs = np.zeros((pts,1), dtype=np.longdouble)
        VF = np.zeros((pts,1), dtype=np.longdouble)
        iF = np.zeros((pts,N), dtype=np.longdouble)          # Current through each grain in A
        Vint = np.zeros((pts,N), dtype=np.longdouble)        # Internal voltage of each grain in V
        iL = np.zeros((pts,1), dtype=np.longdouble)          # Parasitic leakage current in A
        iG = np.zeros((pts,1), dtype=np.longdouble)          # Gate leakage current in A
        it = np.zeros((pts,1), dtype=np.longdouble)          # Total current in A

        # Initial charge conditions
        Q0 = np.sqrt(-alpha/beta/2)
        QF[0,:] = Psign*Q0
        QS[0] = A_MOS*np.real(np.sqrt(2)*eps_s/b/L_D*np.sqrt((np.exp(-b*0)+b*0-1)+n_p0/p_p0*np.exp(-b*VDS)*(np.exp(b*0)-b*0*np.exp(b*VDS)-1)))
        QFall[0] = np.sum(QF[0,:])
        QL[0] = QS[0] - QFall[0]
        for i in range(1, pts):
            iF[i-1,:] = (VF[i-1, :]*np.ones([1,N]) - Vint[i-1,:])/rho
            QF[i,:] = QF[i-1,:] + iF[i-1,:]*dt
            QFall[i] = np.sum(QF[i,:])
            Vint[i,:] = (2*alpha.T[0,:] + 4*beta.T[0,:]*(QF[i,:]**2))*QF[i,:] + Vbias.T[0,:]
            #iL[i-1] = np.sign(VF[i-1])*scipy.interpolate.interp1d(VG, Idata, fill_value="extrapolate")(VF[i-1])*A_MFM
            iL[i-1] = np.sign(VF[i-1])*iL0*np.exp(np.abs(VF[i-1])/VL0)*A_MFM
            iG[i-1] = iG0*Vgs[i-1]**2*np.sign(Vgs[i-1])*A_MOS
            QL[i] = QL[i-1] + (iL[i-1]-iG[i-1])*dt
            it[i-1] = np.sum(iF[i-1,:]) + iL[i-1]
            QS[i] = QS[i-1] + (it[i-1]-iG[i-1])*dt
            if np.abs(scipy.interpolate.interp1d(Qps,Psi_S+Vox+VFB)(QS[i])) > np.abs(V[i]) or np.isnan(np.abs(scipy.interpolate.interp1d(Qps,Psi_S+Vox+VFB)(QS[i]))):
                Vgs[i] = V[i]
            else:
                Vgs[i] = scipy.interpolate.interp1d(Qps,Psi_S+Vox+VFB)(QS[i])
            VF[i] = V[i] - Vgs[i]
        

        ID = W/L*mu_n*Cox*((Vgs-VFB-2*Psi_B-VDS/2)*VDS-2/3*np.sqrt(2*eps_s*q*N_A)/Cox*((VDS+2*Psi_B)**1.5-(2*Psi_B)**1.5))
        ID = ID * (ID > 0)
        VOX = QS/Cox/A_MOS
        PSI_S = Vgs - VFB - VOX
        ID_diff = 100e-9*W/L*np.exp(b*(Vgs-VT))*(1-np.exp(-b*VDS))
        ID_diff = ID_diff * (ID_diff < 100e-9 * W / L)
        ID = ID + ID_diff
        iFall = it - iL
        iMOS = it - iG
        Qp = Cp*Vgs

        #Semiconductor charge vs. surface potential (Slide 1)
        plt.figure()
        plt.semilogy(Psi_S, np.abs(Qs) * 100)
        plt.xlabel(r"$Psi_S$ (V)")
        plt.ylabel(r"$|Q_S|$ ($\frac{\mu C}{cm^2}$)")
        plt.title("Semiconductor Charge")
        #plt.show()
        plt.close()
        #Voltages over time     
        if j == iter and k == iter:
            plt.figure()
            plt.plot(t, V, label = "V")
            plt.plot(t, VF, label = r"$V_{F}$")
            plt.plot(t, Vgs, label = r"$V_{GS}$")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.title("Voltages over time")
            plt.legend()
            #plt.show()
            plt.savefig(path + "/voltage")
            plt.close()

            #Currents over time
            plt.figure()
            plt.plot(t, iMOS/A_MOS * 1e-12, label = r"$I_{MOS}$")
            # plt.plot(t, iF[:, 0], label = r"$I_F$")
            plt.plot(t, iFall/A_MFM * 1e-12, label = r"$I_F$")
            plt.plot(t, iL / A_MFM * 1e-12, label = r"$I_L$")
            plt.plot(t, iG / A_MOS * 1e-12, label = r"$I_G$")
            plt.xlabel("Time (s)")
            plt.ylabel(r"Current $(A$ $\mu m^{-2})$")
            plt.title("Currents over time")
            plt.legend()
            #plt.show()
            plt.savefig(path + "/current")
            plt.close()

            plt.figure()
            plt.plot(t, ID / A_MOS * 1e-12, label = r"I_D")
            plt.xlabel("Time (s)")
            plt.ylabel(r"Current $(A$ $\mu m^{-2})$")
            plt.title("Drain Current")
            plt.legend()
            #plt.show()
            plt.savefig(path + "/IDvst")
            plt.close()

            #Charge over time
            # plt.plot(t, QS, label = r"$Q_S$")
            # plt.plot(t, QF[:, 0], label = r"$Q_F$")
            # plt.plot(t, QL, label = r"$Q_L$")
            plt.figure()
            plt.plot(t, (QS-Qp)/A_MOS * 0.0001, label = r"$Q_S$")
            plt.plot(t, QFall/A_MFM * 0.0001, label = r"$Q_F$")
            plt.plot(t, QL/A_MFM * 0.0001, label = r"$Q_L$")
            plt.xlabel("Time (s)")
            plt.ylabel(r"Charge $(C$ $cm^{-2})$")
            plt.title("Charge over time")
            plt.legend()
            #plt.show()
            plt.savefig(path + "/charge")
            plt.close()

            #Id-VG Curve
            plt.figure()    
            plt.semilogy(V[ID > 0], ID[ID > 0] / A_MOS * 1e-12)
            plt.xlabel("Voltage (V)")
            plt.ylabel(r"log(Current) $(A$ $\mu m^{-2})$")
            plt.title("Id-Vg Curve")
            #plt.show()
            plt.savefig(path + "/IdVg")
            plt.close()

        # plt.plot(t, Vgs)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Voltage (V)")
        # plt.title("Vgs P up")
        # plt.show()

            for i in range(iF.shape[1]):
                plt.plot(t, iF[:, i])
            plt.xlabel("Time (s)")
            plt.ylabel("Current (A)")
            plt.title("All iF")
            plt.savefig(path + "/alliF")
            plt.close()
            #plt.show()

            for i in range(QF.shape[1]):
                plt.plot(t, QF[:, i])
            plt.xlabel("Time (s)")
            plt.ylabel("Charge (C)")
            plt.title("All QF")
            plt.savefig(path + "/allQF")
            plt.close()

            plt.plot(rho)
            plt.savefig(path + "/rand")
            plt.close()

            df = pd.DataFrame(rho)
            df.to_csv("data.csv")

            
        #plt.show()

        Qfinal1[k,j] = QFall[int(np.round((2*Twait+2*Trf+Tread)/dt))]/A_MFM
        Qfinal2[k,j] = QFall[int(np.round((3*Twait+4*Trf+Tread+Tprg)/dt))]/A_MFM
        IDmean1[k,j] = np.mean(ID[int(np.round((Twait)/dt)):int(round((Twait+2*Trf+Tread)/dt))])
        IDmean2[k,j] = np.mean(ID[int(round((3*Twait+4*Trf+Tread+Tprg)/dt)):int(round((3*Twait+6*Trf+2*Tread+Tprg)/dt))])
        V_read[k] = Vread
        T_read[j] = Tread

    # plt.figure()
    # plt.plot(t, ID)
    # plt.xlabel("Time (s)")
    # plt.ylabel(r"$I_D$ (A)")
    # plt.show()


        
        

ON_OFF_diff = IDmean1-IDmean2
ON_OFF_ratio = IDmean1/IDmean2

# plt.figure()
# plt.plot(T_read, V_read, label = "V_read")
# plt.xlabel("T_pulse (s)")
# plt.ylabel("V_read (V)")
# plt.legend()
# plt.show()

data = ON_OFF_diff
plt.figure(figsize=(8, 6))
cax = plt.imshow(data, interpolation='nearest', cmap='viridis')
cbar = plt.colorbar(cax)
cbar.set_label('Value')
x_start, x_end = T_read[0], T_read[len(T_read) - 1]
y_start, y_end = V_read[0], V_read[len(V_read) - 1]
x_labels = np.linspace(x_start, x_end, num=10)
y_labels = np.linspace(y_start, y_end, num=10)
plt.xlabel(r'T_read ($\mu s$)')
plt.ylabel('V_pulse (V)')
plt.xticks(ticks=np.arange(10), labels=["{:.5f}".format(float(x)*1e6)[:3] for x in x_labels])
plt.yticks(ticks=np.arange(10), labels=["{:.5f}".format(float(y))[:3] for y in y_labels])
plt.title(r'$I_{ON}$ vs. $I_{OFF}$')
#plt.show()
plt.savefig(path + "/ON_OFF_Diff")
plt.close()

data = np.log(ON_OFF_ratio)
plt.figure(figsize=(8, 6))
cax = plt.imshow(data, interpolation='nearest', cmap='viridis')
cbar = plt.colorbar(cax)
cbar.set_label('Value')
x_start, x_end = T_read[0], T_read[len(T_read) - 1]
y_start, y_end = V_read[0], V_read[len(V_read) - 1]
x_labels = np.linspace(x_start, x_end, num=10)
y_labels = np.linspace(y_start, y_end, num=10)
plt.xlabel(r'T_read ($\mu s$)')
plt.ylabel('V_pulse (V)')
plt.xticks(ticks=np.arange(10), labels=["{:.5f}".format(float(x)*1e6)[:3] for x in x_labels])
plt.yticks(ticks=np.arange(10), labels=["{:.5f}".format(float(y))[:3] for y in y_labels])
plt.title(r'$I_{ON}$/$I_{OFF}$')
#plt.show()
plt.savefig(path + "/ON_OFF_ratio")
plt.close()


data = Qfinal1
plt.figure(figsize=(8, 6))
cax = plt.imshow(data, interpolation='nearest', cmap='viridis')
cbar = plt.colorbar(cax)
cbar.set_label('Value')
x_start, x_end = T_read[0], T_read[len(T_read) - 1]
y_start, y_end = V_read[0], V_read[len(V_read) - 1]
x_labels = np.linspace(x_start, x_end, num=10)
y_labels = np.linspace(y_start, y_end, num=10)
plt.xlabel(r'T_read ($\mu s$)')
plt.ylabel('V_pulse (V)')
plt.xticks(ticks=np.arange(10), labels=["{:.5f}".format(float(x)*1e6)[:3] for x in x_labels])
plt.yticks(ticks=np.arange(10), labels=["{:.5f}".format(float(y))[:3] for y in y_labels])
plt.title(r'$Charge$')
plt.savefig(path + "/Q2d")
plt.close()
#plt.show()
