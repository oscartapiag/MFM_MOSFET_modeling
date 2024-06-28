#Credit to Junhao Chen for developing this code

import numpy as np 
import matplotlib.pyplot as plt 
import scipy
import scipy.signal
from waveforms import waveformGenerator

path = "./plots/threshold_voltage/0.8V/"

e_charge = scipy.constants.e    #C
k_B = scipy.constants.k  # J/K
T_0 = 300   # K
beta_e = e_charge/(k_B*T_0)
epsilon_0 = scipy.constants.epsilon_0   # C/Vm

#MOS Cap Parameters

# Silicon Channel
n_i = 1e10*1e6             # intrinsic carrier concentration in m^-3
N_A = 2e17*1e6             # Acceptor concentration in m^-3
N_D = 0                    # Donor concentration in m^-3
mu_n = 10*1e-4             # electron mobility in m^2/Vs
epsilon_Si = epsilon_0*11.9            # Permittivity of silicon in As/Vm

# MOSFET
L = 1e-6 # Gate length in m
W = 1e-6 # Gate width in m
EOT = 1e-9 # Oxide thickness in m
V_FB = -0.3 # flat band voltage in V
VDS = 0.05 # drain-source voltage in V

A_MOS = L*W # MOS cap area
Psi_B = 1/beta_e * np.log(N_A/n_i) # Onset of strong inversion in V
p_p0 = ( (N_A-N_D)+ np.sqrt((N_A-N_D)**2 + 4*n_i**2) )/2 # Equilibrium hole concentration in m^-3
n_p0 = n_i**2/p_p0 # Equilibrium electron concentration in m^-3
L_D = np.sqrt(epsilon_Si/(e_charge * p_p0 * beta_e)) # % Extrinsic Debye-length for holes in m
C_ox = 3.9*epsilon_0/EOT # Gate oxide capacitance per area in F/m^2
#V_TH = V_FB + 2*Psi_B + np.sqrt(2*epsilon_Si* e_charge * N_A *2*Psi_B)/(C_ox) # threshold voltage in V
V_TH = 0.8

def F_func(Psi_S, V_ds = VDS):
    return np.sqrt((np.exp(-beta_e *Psi_S)+ beta_e*Psi_S -1) + n_p0/p_p0*np.exp(-beta_e *V_ds) *
                   (np.exp( beta_e *Psi_S)- beta_e*Psi_S* np.exp(beta_e*V_ds)-1))*np.sign(Psi_S)
def Func_Q_MOS(Psi_S, V_ds=VDS): # Total charge density at the surface of the semiconductor in C/m^2
    return np.real(np.sqrt(2)* epsilon_Si /beta_e /L_D * F_func(Psi_S, V_ds))

plt.figure()
Psi_Si_list = np.linspace(-0.7,2,500)
Q_MOS_list = Func_Q_MOS(Psi_S=Psi_Si_list)

# Gate leakage current density:
def Gate_leak(V_GS):
    i_G0 = 10e5  # MOS leakage current density in A/m2 at VG = 1 V
    return i_G0 * V_GS**2 * np.sign(V_GS)
ax1 = plt.subplot(111)

def Drain_current(V_GS_list, V_DS_list):
    I_D_list = []
    m = 1.05
    for i, (V_GS, V_DS) in enumerate(zip(V_GS_list,V_DS_list)):
        if V_GS > V_TH:
            if V_DS > (V_GS - V_TH)/m:
                V_DS = (V_GS - V_TH)/m
            I_D = W/L * mu_n * C_ox * ((V_GS - V_TH - m*V_DS/2)*V_DS)
        else:
            # I_D = 0
            I_D = W/L * mu_n * C_ox * (1/beta_e)**2 * (m-1) * (1-np.exp(-V_DS/beta_e)) * \
                  np.exp( (V_GS-V_TH)/ beta_e / m) 
        I_D_list.append(I_D)
    return I_D_list

#Ferroelectric Layer and Grain Parameters

A_FE = (5e-6)*(0.75e-6)    # m^2
d_FE = 4.5e-9   # m
P_r = 2.5 * 1e-2 # remanent polarization C/m^2
E_c = 0.9 * 1e6 / 1e-2    # coercive field V/m

def FE_leak(V):
    i_FE0 = 1.8e3 # MFM leakage current density at VF = 0V in A/m2
    V_FE0 = 0.5 # MFM leakage normalization voltage in V
    return i_FE0* np.sign(V) * np.exp(np.abs(V)/ V_FE0 ) 

N_domain = 10
rho_0 = 400 # Single domain internal resistivitsy in Ohm*m
alpha_0 = -3 * np.sqrt(3)*E_c/4/P_r *d_FE*N_domain/A_FE
beta_0 = -1 * alpha_0 / 2 / P_r**2 *(N_domain/A_FE)**2  # 1st Landau constant in m/F

std_domain = 0.2
alpha = alpha_0 * np.abs(np.random.normal(1, std_domain, N_domain))
beta = beta_0 * np.abs(np.random.normal(1, std_domain, N_domain))
rho = rho_0 * np.abs(np.random.normal(1, 0, N_domain)) * d_FE / A_FE * N_domain
def func_V_Q(Q):
    return 2*alpha*Q + 4*beta*Q**3

Sim_steps = int(1e5)
gen = waveformGenerator()
t_rise = 0.1
t_write = 1 
t_delay = 1
t_read = 3
V_write = 2
V_read = 2
V_hold = 0.0
Waveform_time = 1e-6 * np.cumsum([0, 1, t_rise, t_write, t_rise, t_delay])
Waveform_voltage = np.array([0, 0, V_write, V_write, V_hold, V_hold])
dt = max(Waveform_time) / Sim_steps
# Sim_time = np.linspace(0, max(Waveform_time), Sim_steps)
print(dt)
# V_input = np.interp(x = Sim_time, xp = Waveform_time, fp= Waveform_voltage)
V_input, V_DS, t_total = gen.pulses_constantVd(Sim_steps, VDS, 2.5, 1e-6, 2, 3e-6, 2e-6)
Sim_time = np.linspace(0, t_total, Sim_steps)
# dt = t_total / Sim_steps
# print(dt)
plt.figure()
#plt.plot(Waveform_time, Waveform_voltage, label = 'V_GS')
plt.plot(Sim_time, V_input, label = 'V_GS')
#plt.plot(Waveform_time, V_DS*np.ones(Waveform_voltage.shape), label = 'V_DS')
plt.plot(Sim_time, V_DS, label = "V_DS")
plt.legend()
plt.savefig(path+"Vgs+Vds")
plt.close()

V_GS = np.zeros(Sim_steps)
Psi_Si_list = np.linspace(-0.4,1.7,500)
Q_MOS_list = Func_Q_MOS(Psi_S=Psi_Si_list)
V_MOS_list = Psi_Si_list + Q_MOS_list/C_ox + V_FB

I = np.zeros(Sim_steps)
I_Leak = np.zeros(Sim_steps)
I_Gate_leak = np.zeros(Sim_steps)
V_FE = np.zeros(Sim_steps)
Q_FE_tot = np.zeros(Sim_steps)
Q_MOS = np.zeros(Sim_steps)
Q_Leak = np.zeros(Sim_steps)
Q_FE = np.zeros((Sim_steps, N_domain))
I_FE =np.zeros((Sim_steps, N_domain))
V_int = np.zeros((Sim_steps, N_domain)) # internal voltages

#Initial Conditions
# initial conditions
Q_0 = np.sqrt(-alpha/beta/2) # charge on every domain interface
P_sign = -1
Q_FE[0, :] = Q_0 * P_sign
Q_FE_tot[0] = np.sum(Q_FE[0, :])
Q_MOS[0] = np.interp(x = 0, xp = V_MOS_list, fp= Q_MOS_list)* A_MOS
Q_Leak[0] = Q_MOS[0] - Q_FE_tot[0]

for i in range(1, Sim_steps):
    I_FE[i-1, :] = (V_FE[i-1] * np.ones((1, N_domain)) - V_int[i-1, :]) / rho
    Q_FE[i, :] = Q_FE[i-1, :] + I_FE[i-1, :] * dt
    Q_FE_tot[i] = np.sum(Q_FE[i, :])
    V_int[i, :] = func_V_Q(Q_FE[i, :])
    I_Leak[i-1] = FE_leak(V_FE[i-1])* A_FE
    I_Gate_leak[i-1] = Gate_leak(V_GS[i-1]) * A_MOS
    Q_Leak[i] = Q_Leak[i-1] + (I_Leak[i-1] - I_Gate_leak[i-1]) *dt
    I[i-1] = np.sum(I_FE[i-1,:]) + I_Leak[i-1]
    Q_MOS[i] = Q_MOS[i-1] + (I[i-1] - I_Gate_leak[i-1])*dt
    V_GS[i] = np.interp(x = Q_MOS[i]/A_MOS, xp = Q_MOS_list, fp= V_MOS_list)
    V_FE[i] = V_input[i] - V_GS[i]  
#MOSFET Drain Current Calculation
# plt.figure()
# #plt.plot(scipy.signal.stft(I_Leak)[2], color = "red")
# #plt.plot(scipy.signal.stft(I - I_Gate_leak)[2], color = "blue")
# plt.plot(scipy.signal.stft(I - I_Leak)[2], color = "green")
# plt.show()
# plt.close()
I_D = np.array(Drain_current(V_GS,V_DS))

#Plots
plt.figure(figsize=(14,9))
plt.subplot(2, 2, 1)
plt.plot(Sim_time, V_input, '-', label = 'V_input')
plt.plot(Sim_time, V_FE, '-', label = 'V_FE')
plt.plot(Sim_time, V_GS, '-', label = 'V_GS')
plt.hlines(V_TH, xmin=Sim_time[0], xmax=Sim_time[-1], linestyles='--', colors='r', label='V_TH')
plt.plot(Sim_time, V_DS, "--", label = "V_DS", color = "yellow")
plt.xlabel('Time(s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(Sim_time, Q_MOS/A_MOS, '--', label = 'Q_MOS')
plt.plot(Sim_time, Q_FE_tot/A_FE, '-', label = 'Q_FE')
plt.plot(Sim_time, Q_Leak/A_FE, '--', label = 'Q_Leak')
plt.xlabel('Time(s)')
plt.ylabel('Charge (C/m^2)')
plt.legend()
ax1 = plt.subplot(2, 2, 3)
ax1.plot(Sim_time, V_input, '--g',label = "V_G")
ax1.legend(loc=4)
ax1.set_ylabel('Voltage(V)')
ax1.set_xlabel('Time(s)')
ax2 = ax1.twinx() # this is the important function
ax2.plot(Sim_time, np.abs(I_D),'-r',label="|I_D|")
ax1.plot(Sim_time, V_DS, "--", color = "yellow", label = "V_DS")
ax2.legend(loc=3)
# ax2.set_xlim([0, np.e])
ax2.set_ylabel('Current(A)')
plt.subplot(2, 2, 4)
plt.plot(Sim_time, I - I_Gate_leak, '-', label = 'I_MOS')
plt.plot(Sim_time, I_Gate_leak, '-', label = 'I_Gate_leak')
plt.plot(Sim_time, I - I_Leak, '-', label = 'I_FE') # = np.sum(I_FE, axis=1)
plt.plot(Sim_time, I_Leak, '-', label = 'I_FE_leak')
plt.xlabel('Time(s)')
#plt.ylabel('Current (A)')
plt.legend()
plt.savefig(path + "charge-voltage-current-draincurrent")
plt.close()

plt.figure(figsize=(8,6))
plt.semilogy(V_input[I_D > 0], I_D[I_D > 0])
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.savefig(path + "IDVG")
plt.title("ID-VG")
plt.close()

plt.figure(figsize=(8,6))
plt.plot(Sim_time, Q_FE/(A_FE/N_domain), '--')
plt.xlabel('Time(s)')
plt.ylabel('Charge (C/m^2)')
plt.legend([f'Domain {i+1}' for i in range(N_domain)])
plt.savefig(path + "domain-charges")
plt.close()

#2D Plots

K, J = 10, 10
IDmean1 = np.zeros((K,J), dtype=np.longdouble)
IDmean2 = np.zeros((K,J), dtype=np.longdouble)
Qfinal1 = np.zeros((K,J), dtype=np.longdouble)
Qfinal2 = np.zeros((K,J), dtype=np.longdouble)
V_read = np.zeros((K,1), dtype=np.longdouble)
T_read = np.zeros((J,1), dtype=np.longdouble)

for k in range(0, K):
    print(k)
    for j in range(0, J):
        #Voltage Waveform
        Vread = 0.2 * (k + 1)       #Read voltage in V
        Vprg = 2                    #PRG voltage in V
        Twait = 20e-6               #Wait time between pulses in s
        Trf = 5e-8                  #Rise/fall times in s
        Tprg = 10e-6                #PRG pulse width in s    
        Tread = 1e-7 * (2) ** ((j + 1) -1)      #Read pulse width in s
        T_total = 4*Twait + 6*Trf + Tprg + 2*Tread
        #T_total = 3e-6

        #Simulation parameters
        pts = Sim_steps                     #Number of simulation points
        dt = T_total/pts                #Simulation time steps in s

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
        
        V_GS = np.zeros(pts)
        Psi_Si_list = np.linspace(-0.4,1.7,500)
        Q_MOS_list = Func_Q_MOS(Psi_S=Psi_Si_list)
        V_MOS_list = Psi_Si_list + Q_MOS_list/C_ox + V_FB
        
        I = np.zeros(pts)
        I_Leak = np.zeros(pts)
        I_Gate_leak = np.zeros(pts)
        V_FE = np.zeros(pts)
        
        Q_FE_tot = np.zeros(pts)
        Q_MOS = np.zeros(pts)
        Q_Leak = np.zeros(pts)
        Q_FE = np.zeros((pts, N_domain))
        
        I_FE =np.zeros((pts, N_domain))
        V_int = np.zeros((pts, N_domain)) # internal voltages

        Q_0 = np.sqrt(-alpha/beta/2) # charge on every domain interface
        
        P_sign = -1
        Q_FE[0, :] = Q_0 * P_sign
        Q_FE_tot[0] = np.sum(Q_FE[0, :])
        
        Q_MOS[0] = np.interp(x = 0, xp = V_MOS_list, fp= Q_MOS_list)* A_MOS
        Q_Leak[0] = Q_MOS[0] - Q_FE_tot[0]
        
        for i in range(1, pts):
            I_FE[i-1, :] = (V_FE[i-1] * np.ones((1, N_domain)) - V_int[i-1, :]) / rho
            Q_FE[i, :] = Q_FE[i-1, :] + I_FE[i-1, :] * dt
            Q_FE_tot[i] = np.sum(Q_FE[i, :])
        
            V_int[i, :] = func_V_Q(Q_FE[i, :])
            I_Leak[i-1] = FE_leak(V_FE[i-1])* A_FE
            I_Gate_leak[i-1] = Gate_leak(V_GS[i-1]) * A_MOS
        
            Q_Leak[i] = Q_Leak[i-1] + (I_Leak[i-1] - I_Gate_leak[i-1]) *dt
            I[i-1] = np.sum(I_FE[i-1,:]) + I_Leak[i-1]
            Q_MOS[i] = Q_MOS[i-1] + (I[i-1] - I_Gate_leak[i-1])*dt
            if np.interp(x = Q_MOS[i]/A_MOS, xp = Q_MOS_list, fp= V_MOS_list) > np.max(V):
                V_GS[i] = V[i]
            else:
                V_GS[i] = np.interp(x = Q_MOS[i]/A_MOS, xp = Q_MOS_list, fp= V_MOS_list)
            V_FE[i] = V[i] - V_GS[i]
        I_D = Drain_current(V_GS,V_DS*np.ones(V_GS.shape))
        Qfinal1[k,j] = Q_FE_tot[int(np.round((2*Twait+2*Trf+Tread)/dt))]/A_FE
        Qfinal2[k,j] = Q_FE_tot[int(np.round((3*Twait+4*Trf+Tread+Tprg)/dt))]/A_FE
        IDmean1[k,j] = np.mean(I_D[int(np.round((Twait)/dt)):int(round((Twait+2*Trf+Tread)/dt))])
        IDmean2[k,j] = np.mean(I_D[int(round((3*Twait+4*Trf+Tread+Tprg)/dt)):int(round((3*Twait+6*Trf+2*Tread+Tprg)/dt))])
        V_read[k] = Vread
        T_read[j] = Tread

ON_OFF_diff = IDmean1-IDmean2
ON_OFF_ratio = IDmean1/IDmean2

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
plt.title(r'$I_{ON}$ - $I_{OFF}$')
#plt.show()
plt.savefig(path + "ON_OFF_Diff")
plt.close()

data = np.log10(ON_OFF_ratio)
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
plt.savefig(path + "ON_OFF_ratio")
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
plt.savefig(path + "Q2d")
#plt.show()
plt.close()

