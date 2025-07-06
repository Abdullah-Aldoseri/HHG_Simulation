

# ---------------- Time scale figure ----------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

fig, ax = plt.subplots(figsize=(15, 1.8), dpi=300)

start_time = 1e-18
end_time = 1e18

num_points = 1000
time_values = np.logspace(np.log10(start_time), np.log10(end_time),
                          num=num_points)

comment_times = [1e-18, 1e-14, 1e-9, 1e-3, 60, 2.4e6, 1.4e13, 1e17]
comment_labels = [
    'S-P states electron\nmotion in H atom', 'fs laser\npulse', 
    'State-of-the-art CPU\nincrement', 'Camera flash\nduration', 
    '1 minute', 'Moon orbit\ncycle', 'First human\nrecorded existence', 
    'Age of\nuniverse'
]

cmap_colors = plt.cm.viridis(np.linspace(0, 1, num_points))
custom_cmap = LinearSegmentedColormap.from_list("CustomCmap", cmap_colors)

for t, color in zip(time_values, cmap_colors):
    ax.plot([t, t], [0, 1], color=color, linewidth=5)

for t, label in zip(comment_times, comment_labels):
    ax.text(t, 1.15, label, ha='center', va='bottom', fontsize=12,
            fontweight='bold', color='black')

ax.set_xscale('log')
ax.set_xlim(start_time, end_time)
ax.set_ylim(0, 1.5)
ax.set_yticks([])
ax.set_xlabel('Time (seconds)', fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.show()

# ---------------- Qualitative Coulomb potential ----------------

import matplotlib.pyplot as plt
from scipy.constants import electron_volt as eV
import numpy as np

Z = 1
e = 1.602176634e-19
eps0 = 8.854187817e-12
f0 = 3.75e14
elc = 1.602e-19
f0 = 3.75e14
m_e = 9.109e-31
w0 = f0 * 2 * np.pi

phi_array = np.linspace(0, np.pi / 2, num=100, endpoint=False)
phi = 0
I = 2e18
C = 299792458
eps0 = 8.85e-12
r_ind = 1.0
E0 = np.sqrt((2 * I) / (C * eps0 * r_ind))
t = 2e-15

Modification_amount = 1

r = np.linspace(0.5e-12, 1e-11, 1000)

V = - (Z * e ** 2) / (4 * np.pi * eps0 * r)  # Potential equation

E_field = E0 *(np.cos((w0*t) + phi)*Modification_amount) #Modifying Laser Field
V_mod = V + r * E_field
V_mod2 = V + r * -E_field

# Plotting
plt.figure(figsize=(10, 6), dpi=300)

r_angst = r * 1e10
plt.plot(r_angst, V / eV, "--", label='Coulomb Potential', color='b')
plt.plot(-r_angst, V / eV, "--", color='b')

t_fs = t * 1e15

plt.plot(-r_angst, V_mod / eV, 
         label=f'Modified Coulomb Potential at t={t_fs} fs', color='r')
plt.plot(r_angst, V_mod2 / eV, color='r')
plt.axhline(y=-1500, xmin=0, xmax=10, color='black', linestyle='-', 
            label='Ionisation potential')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

# To omit the ticks on the x and y axis
plt.xticks([], [])
plt.yticks([], [])

plt.xlabel('Distance from nucleus (Å)')
plt.ylabel('Potential (eV)')
plt.title('Coulomb Potential Modified by Laser Field')
plt.legend()
plt.show()




# ----------------- The Gaussian and ADK plot ---------------------

import scipy.optimize as sop
import numpy as np
import matplotlib.pyplot as plt
import math 

# Constants and Parameters
n = 1               # Principal quantum number
l = 1               # Orbital quantum number
m = 0               # Magnetic quantum number
r_ind = 1.          # Refractive index
z = 1               # Degree of ionisation, charge
C = 299792458       # Speed of light in m/s
l2 = 0.8            # Effective orbital quantum number (ADK)
n2 = 0.93           # Effective principal quantum number (ADK)
Ei = (z**2 / (2 * n2**2))  # Binding energy of argon (in a.u.)

eps0 = 8.85e-12     # Permittivity of vacuum
m_e = 9.109e-31     # Electron rest mass
elc = 1.602e-19     # Elementary charge

I = 2e18            # Laser intensity in W/m^2
phi1 = 0            # Phase
Eh_eV = 27.211      # Hartree energy in eV
a0 = 0.528e-10      # Bohr radius
Eh_SI = 27.211 * elc  # Hartree energy in Joules
E_in_au = Eh_SI / elc / a0  # Electric field in atomic units

# Time definitions
t = np.linspace(-50e-15, 50e-15, 1000, endpoint=False) 
T = np.linspace(0, 4e-15, 1000)
T2 = np.linspace(0, 2e-14, 1000)

t_fwhm = 2.5e-14   # FWHM of the pulse period
f0 = 3.75e14       # Central frequency of driving laser
w0 = f0 * 2 * np.pi
sig = t_fwhm / (2 * np.sqrt(2 * np.log(2)))  # Width, standard deviation
I = 2e18           # Laser intensity
E0 = np.sqrt((2 * I) / (C * eps0 * r_ind))  # Peak electric field strength

# Gaussian Wave Packet Function
def Gauss_WP(t, sig, t_fwhm, w0, phi):
    term1 = np.exp(-t**2 / (2 * (sig**2)))
    term2 = np.cos((w0 * t) + phi1)
    return term1 * term2 * E0 / E_in_au

E = Gauss_WP(t, sig, t_fwhm, w0, phi1)  # Gaussian electric field in a.u.

# ADK Ionization Rate Function

# ADK Ionization Rate Function
def ionz_rate(E):
    term1 = np.sqrt(3 / (np.pi**3))
    term2 = (((2 * l) + 1) * math.factorial(l + abs(m))) / \
            (math.factorial(abs(m)) * math.factorial(l - abs(m)))
    term3 = (elc / (((n2**2) - (l2**2))**0.5))**(abs(m) + 0.5)
    term4 = ((n2 + l2) / (n2 - l2))**l2
    term5 = (z**2 / n2**3) * ((4 * np.e * z**3) / (E * (n2**3) * \
            (((n2**2) - (l2**2))**0.5)))**((2 * n2 - abs(m) - 1.5))
    w1 = term1 * term2 * term3 * term4 * term5 * \
         np.exp(-(2 * z**3) / (3 * E * n2**4))
    return w1

w1 = ionz_rate(E)  # ADK ionization rate

# Plotting
plt.figure(figsize=(10, 6), dpi=300)
t_fs = t * 1e15
E_envelope = np.abs(E0 * np.exp(-t**2 / (2 * (sig ** 2))) / E_in_au)

# ADK Ionization Probability Plot
ax1 = plt.gca()
ax1.plot(t_fs, w1, 'b-', linewidth=4, label='ADK ionization probability')
ax1.set_xlabel("Time (fs)")
ax1.set_ylabel("ADK ionization probability", color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.legend(loc='upper left')

# Gaussian Wave Packet and Pulse Envelope Plot
ax2 = ax1.twinx()
c_freq = f0 / 1e12  # Convert to THz
ax2.plot(t_fs,E,'r-',label=f'G. wave P.,\nf0={c_freq}THz,\n$\lambda$ = 400 nm')
ax2.plot(t_fs, E_envelope, 'g--', label='Pulse envelope')  # Upper envelope
ax2.plot(t_fs, -E_envelope, 'g--')  # Lower envelope

# Annotating FWHM
half_max = max(E) / 2
fwhm_start = -12.5  # FWHM start
fwhm_end = 12.5     # FWHM end
ax2.axvline(fwhm_start, color='k', linestyle='--', label='FWHM')
ax2.axvline(fwhm_end, color='k', linestyle='--')
ax2.annotate('FWHM\n= 25 fs', xy=(0, half_max), xytext=(15, half_max),
             fontsize=12, color='black')

ax2.set_ylabel("Amplitude of Gaussian", color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.legend(loc='upper right')

plt.show()





phi_array = np.linspace(0,np.pi/2,num=100, endpoint=False) # different phases 


E0 = np.sqrt((2*I)/(C*eps0*r_ind))                       

def func1 (T2,phi,w):
    return np.cos((w*T2)+phi)-np.cos(phi)

def func2 (T2,phi,w):    
    return w * -np.sin(phi)*T2

def Velocity(arrival_times,phi_array):
    return -((elc * E0) / (m_e * w0)) *\
        (np.sin((w0*arrival_times)+phi_array)-np.sin(phi_array)) 

def Pondro_E (elc,E0, m_e, w):
    return (elc**2 * E0**2)/(4*m_e*w**2)
                
Up = Pondro_E (elc,E0, m_e, w0) #in joules(c2(F/c)2)/(Kg*1/s2)= (m2 Kg/s2)









phi_array =np.linspace(0, np.pi/2, num=20,
                       endpoint=False)# More points for smoother curve

plt.figure(dpi=300)  # High resolution

T_fs = T * 1e15  # Convert time from seconds to femtoseconds

comment_phis = np.linspace(0, len(phi_array) - 1, num=10, dtype=int)

for pp, phi in enumerate(phi_array):
    plt.plot(T_fs, func1(T, phi, w0) - func2(T, phi, w0), label=str(phi))
    plt.plot(T_fs, 0 * T, color='k', linestyle='--')
    
    #  phi to degrees
    phi_degrees = (phi / np.pi) * 180
    
    # Add comments only for specific phi values
    if pp in comment_phis:
        x_comment = T_fs[-1] * 1.05 
        y_comment = func1(T[-1], phi, w0) - func2(T[-1], phi, w0)
        
        # Check if phi equal to 18
        if phi_degrees == 18:
            comment = f'$\phi$: {phi_degrees:.2f}°' #Format
            plt.text(x_comment, y_comment, comment, fontsize=10, color='red',
                     weight='bold', verticalalignment='center',
                     horizontalalignment='left')
        else:
            comment = f'$\phi$: {phi_degrees:.2f}°'
            plt.text(x_comment, y_comment, comment, fontsize=10,
                     verticalalignment='center', horizontalalignment='left')

plt.xlabel("Time (fs)")
plt.ylabel("Trajectories as a function of birth time")

plt.show()





phi_array = np.linspace(0*np.pi, 2*np.pi, num=100, endpoint=False)



 # counter for trajectories that cross zero once twice or more 
cross_zero_count = 0
non_cross_zero_count = 0
multi_cross_count = 0 

offset_factor = 0
ax1 = plt.gca()


for (pp, phi) in enumerate(phi_array):
    y_values = func1(T, phi, w0) - func2(T, phi, w0) + pp * offset_factor
    crosses_zero = [y > pp * offset_factor for y in y_values[1:]]
    
    cross_count = sum(1 for i in range(1, len(crosses_zero))\
                      if crosses_zero[i] != crosses_zero[i-1])

    phi_deg = np.degrees(phi)

    if cross_count >= 2:
        ax1.plot(t_fs, y_values, color='green')
        multi_cross_count += 1 
    elif np.isclose(phi_deg, 18) or np.isclose(phi_deg, 198):
        ax1.text(0.05, pp * offset_factor, f"{phi_deg:.0f}°", color='black')

        ax1.plot(t_fs, y_values, color='blue')
    elif cross_count == 1:
        ax1.plot(t_fs, y_values, color='red')
        cross_zero_count += 1
    else:
        ax1.plot(t_fs, y_values, color='black')
        non_cross_zero_count += 1

zero_line = np.linspace(0, (len(phi_array) - 1) * offset_factor, len(T))
ax1.plot(t_fs, zero_line, color='k', linestyle='--')



ax1.set_xlabel("Time (fs)")
ax1.set_ylabel("Trajectories as a function of birth time")

from matplotlib.lines import Line2D


# To arrange the legend
custom_lines = [Line2D([0], [0], color='black' , lw=2),
                Line2D([0], [0], color='red'   , lw=2),
                Line2D([0], [0], color='green' , lw=2),
                Line2D([0], [0], color='blue'  , lw=2)]

plt.legend(custom_lines, [f'Do not cross zero ({non_cross_zero_count})',
                          f'Cross zero once ({cross_zero_count})',
                          f'Multi-cross ({multi_cross_count})',
                          'Cut-off (2)'], loc='upper right')





phi_ticks = np.linspace(0, (len(phi_array) - 1) * offset_factor, 20)
phi_tick_labels = np.degrees(np.linspace(0, 2 * np.pi, num=20, endpoint=False))

ax1.set_yticks(phi_ticks)
ax1.set_yticklabels([f"{phi:.0f}" for phi in phi_tick_labels])
ax1.set_ylabel("Trajectories approx. $\phi^°$ vals. with offest")

plt.show()
















# Adding the offeset to the curves

phi_array = np.linspace(0*np.pi, 2*np.pi, num=100, endpoint=False)

cross_zero_count = 0
non_cross_zero_count = 0
multi_cross_count = 0 

offset_factor = 0.6
ax1 = plt.gca()


for (pp, phi) in enumerate(phi_array):
    y_values = func1(T, phi, w0) - func2(T, phi, w0) + pp * offset_factor
    crosses_zero = [y > pp * offset_factor for y in y_values[1:]]
    
    cross_count = sum(1 for i in range(1, len(crosses_zero))\
                      if crosses_zero[i] != crosses_zero[i-1])

    phi_deg = np.degrees(phi)

    if cross_count >= 2:
        ax1.plot(t_fs, y_values, color='green')
        multi_cross_count += 1 
    elif np.isclose(phi_deg, 18) or np.isclose(phi_deg, 198):
        ax1.text(0.05, pp * offset_factor, f"{phi_deg:.0f}°", color='black')

        ax1.plot(t_fs, y_values, color='blue')
    elif cross_count == 1:
        ax1.plot(t_fs, y_values, color='red')
        cross_zero_count += 1
    else:
        ax1.plot(t_fs, y_values, color='black')
        non_cross_zero_count += 1

zero_line = np.linspace(0, (len(phi_array) - 1) * offset_factor, len(T))
ax1.plot(t_fs, zero_line, color='k', linestyle='--')



ax1.set_xlabel("Time (fs)")
ax1.set_ylabel("Trajectories as a function of birth time")

from matplotlib.lines import Line2D


# To arrange the legend
custom_lines = [Line2D([0], [0], color='black' , lw=2),
                Line2D([0], [0], color='red'   , lw=2),
                Line2D([0], [0], color='green' , lw=2),
                Line2D([0], [0], color='blue'  , lw=2)]

plt.legend(custom_lines, [f'Do not cross zero ({non_cross_zero_count})',
                          f'Cross zero once ({cross_zero_count})',
                          f'Multi-cross ({multi_cross_count})',
                          'Cut-off (2)'], loc='upper right')





phi_ticks = np.linspace(0, (len(phi_array) - 1) * offset_factor, 20)
phi_tick_labels = np.degrees(np.linspace(0, 2 * np.pi, num=20, endpoint=False))

ax1.set_yticks(phi_ticks)
ax1.set_yticklabels([f"{phi:.0f}" for phi in phi_tick_labels])
ax1.set_ylabel("Trajectories approx. $\phi^°$ vals. with offest")

plt.show()














''''----------------Smooth cut off energy------------------------'''





plt.figure(dpi=300)

phi_array = np.linspace(0, np.pi/2,
                        num=1000,
                        endpoint=False)  # More points for smoother curve

T = np.linspace(0, 4e-15, 1000)  # More points for smoother curve

phi_array = np.linspace(0,np.pi/2,num=50, endpoint=False) 

arrival_times = []                   

for (pp, phi) in enumerate(phi_array):
    value = func1(T,phi,w0)-func2(T,phi,w0)
    for (ii,val) in enumerate(value):
        if val > 0:                  
                               
            break
    arrival_times.append(T[ii])     

velocity_array = Velocity(np.array(arrival_times),phi_array)

kinetic_energy_array =  (1/2) * m_e *((velocity_array)**2)/Up


#  kinetic energy array with the smooth curve
kinetic_energy_array_smooth = (1/2) * m_e * ((Velocity(np.array(arrival_times),
                                                       phi_array))**2) / Up

plt.plot((phi_array/np.pi)*180,
         kinetic_energy_array_smooth)  #Plots K.E. vs phi
plt.xlabel("$\phi°$")
plt.ylabel("E$_{k}$ / U$_p$ ratio")
plt.grid()



# Draw a vertical line at phi=17.999°
vertical_line_angle = 17.999
plt.axvline(x=vertical_line_angle, color='red', linestyle='--',
            label=f'$\phi°$ = {vertical_line_angle:.0f}°')

horizontal_line_ratio = 3.17
plt.axhline(y=horizontal_line_ratio, color='blue', linestyle='--',
            label = f'$E_k$/$U_p$ = {horizontal_line_ratio:.2f}') 

x_comment_before = (phi_array[0]/np.pi)*180
y_comment_before = horizontal_line_ratio * 0.9
comment_before = 'Long\ntrajectories\ncontribution'
plt.text(x_comment_before, y_comment_before, comment_before, fontsize=10,
         verticalalignment='center', horizontalalignment='left')

x_comment_after = (phi_array[-1]/np.pi)*60
y_comment_after = horizontal_line_ratio * 0.9
comment_after = 'Short\ntrajectories\ncontribution'
plt.text(x_comment_after, y_comment_after, comment_after, fontsize=10,
         verticalalignment='center', horizontalalignment='left')

plt.legend()
plt.show()











#------------------------------------------------
#------------------------------------------------


#%%%%



'------------------- Qualitative few cycles gaussian pulse ------------------'


import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-1, 1, 5000)  


frequency = 9 / (max(t) - min(t))

# Gaussian envelope
gaussian_envelope = np.exp(-((t)**2) / (2 * (0.2)**2))

# Gaussian equation - term of oscillation
gaussian_pulse = np.cos(2 * np.pi * frequency * t) * gaussian_envelope

# Calculate the phase in radians for all points
phase_rad = 2 * np.pi * frequency * t

max_idx = np.argmax(gaussian_envelope)

# Convert 18 degrees to radians
target_phase_rad = np.deg2rad(18+180)  
target_idx = np.argmin(np.abs(phase_rad - target_phase_rad))

target_phase_rad2 = np.deg2rad(18)
target_idx2 = np.argmin(np.abs(phase_rad - target_phase_rad2))




plt.figure(figsize=(10, 6),dpi=300)

plt.axis('off')
plt.gca().set_frame_on(False)

plt.plot(t, gaussian_pulse, 
         linewidth=1)  # Increased linewidth for better visibility
plt.title("Gaussian Pulse with HHG Cutoff Phase at 18°")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.axhline(0, color='gray', linestyle='--')  # zero line
plt.scatter(t[target_idx], gaussian_pulse[target_idx], color='red') # MarkPoint
plt.annotate(' Cutoff phase = 198° ', (t[target_idx],
                                       gaussian_pulse[target_idx]),
             textcoords="offset points", xytext=(19,-20), ha='center')

plt.annotate('Cutoff\n phase = 18°', (t[target_idx2],
                                      gaussian_pulse[target_idx2]),
             textcoords="offset points", xytext=(19,-20), ha='center')
plt.scatter(t[target_idx2], gaussian_pulse[target_idx2],
            color='red') # Mark point


plt.show()




#%%%%

'-------Transition Dipole Moment vs Photon energy --------'  





N_t     = 8500                    # Number of points in time.

Ip      = 15.76 / 27.2114         # Atom ionization potential for argon.


'''

I = 2e15 w/m^-2 = 0.516 E in au.
I = 2e16 W/m-^2 = 1.632
'''

E0      = 0.105    #For 800nm (The E0 is 0.105 a.u)  
                               
omega0  =   0.057 # In a.u.
A0 	    = E0 / omega0            
T 	    = 2.0 * np.pi / omega0 

#------------------------------------------------
#------------------------------------------------



def E(phir):                # Electric field in scaled coordinates.
	
	return np.cos(phir)

def A(phir):                # Vector potential in scaled coordinates.
	
	return - np.sin(phir)

def intA(phir): 	        # Integrated vector potential in scaled coordinates.

	return np.cos(phir)


# phib : Phase of birth (ionization)
# phir : Phase of recombination
# psr : Momentum divided A0

def saddlePointEquations (x0, phi3r):

 	eqs = np.zeros(2, dtype=float)

 	psr, phi1r = x0

 	Ab    = A(phi1r)
 	#Ar    = A(phi3r)

 	intAb = intA(phi1r)
 	intAr = intA(phi3r)

 	# Ionization equation
 	eqs[0] = np.power(psr + Ab, 2.0)

 	# Propagation equation

 	eqs[1] = intAr - intAb + (phi3r - phi1r) * psr

 	return eqs

def solveSystem (phi3r):

	N     = np.size(phi3r)
	phi1r = np.zeros(N)
	psr   = np.zeros(N)

# For all times in range, use least squares method to solve saddle-point system.
# Initial guess for first round is 0 and 0 for both psr and phi1r.

	for k in range(N):

		if (k == 0):
			res = sop.least_squares (saddlePointEquations, \
							  		 x0=(0, 0), \
							  		 args=(phi3r[k],))
		else:
			res = sop.least_squares (saddlePointEquations, \
							  x0=(psr[k - 1], phi1r[k - 1]), \
							  args=(phi3r[k],))

		psr[k]   = res.x[0]
		phi1r[k] = res.x[1]

	return phi1r, psr

def getSolutions (N_t, PHI_MIN, PHI_MAX):

# Recombination phase phi3r is defined in units of laser cycle.
# Spans 0 to 1.
# Only parts within range defined by PHI_MIN and PHI_MAX are calculated for.

	phi3r = np.linspace(0.0, 1.0, N_t)

# idxMIN/MAX are the min. & max. array indices within which is the recollision
# window defined by PHI_MIN/MAX

	idxMIN = np.argmin(np.absolute(phi3r - PHI_MIN))
	idxMAX = np.argmin(np.absolute(phi3r - PHI_MAX))

	# scale phi3r to units of 2 * pi
	phi3r *= 2.0 * np.pi

	# Define phase of ionization and canonical momentum vectors.
	phi1r = np.zeros(N_t)
	psr   = np.zeros(N_t)

	# Solve system
	phi1r[idxMIN : idxMAX], psr[idxMIN : idxMAX] = solveSystem\
        (phi3r[idxMIN : idxMAX])

	idx = np.array([idxMIN, idxMAX])

	# Return all this stuff.

	return phi1r, phi3r, psr, idx

#------------------------------------------------
#------------------------------------------------


# Recollision window of time (in units of laser cycle).
# Choose PHI_MIN = 0.30, PHI_MAX = 0.7 for short trajectories.
# Choose PHI_MIN = 0.60, PHI_MAX = 1.0 for long trajectories.
# Choose PHI_MIN = 0.30, PHI_MAX = 1.0 for both trajectories.
PHI_MIN = 0.3
PHI_MAX = 0.75

# Solve saddle-point equations in scaled coordinates.

# phib : Phase of birth (ionization).
# phir : Phase of recombination.
# psr : Momentum divided A0.

phib, phir, psr, idx = getSolutions (N_t, PHI_MIN, PHI_MAX)

k  = A0 * np.copy(psr)      # Canonical momentum
tb = np.copy(phib) / omega0 # Time of birth (ionization)
tr = np.copy(phir) / omega0 # Time of recombination
dt = tr[1] - tr[0]

filt = np.zeros_like(phir)
tr[0 : idx[0]] = np.nan
tr[idx[1] : N_t] = np.nan

KE = np.power(k + A0 * A(phir), 2.0) / 2.0


# With v = k + A(t)
v = (k + A0 * A(phir))[idx[0] : idx[1]]
arec  = v / np.power(Ip + np.power(v, 2.0), 3.0)


fig, ax = plt.subplots(1, 1, figsize=(4, 3),dpi=300)
ax.semilogy(27.2113 * KE[idx[0] : idx[1]], arec)
ax.set_ylabel(r'$|D(\Omega)|^2$')
ax.set_xlabel("Energy (eV)")


plt.show()

#------------------------------------------------

#%%%


'-------Spectrum viewing--------'  




import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
from scipy.signal import find_peaks, peak_widths


c=2.997e8






#  paste the pathe name of the image  (bd_220819_Power1.73_press35.bmp)
M = plt.imread('bd_220819_Power1.73_press35.bmp',format = None)






M = np.array(M)*1.0                        
rM = rotate(M, -7.0)
spec = np.mean(rM[560:610,400:1200],axis=0)    # spectrum
bgnd = np.mean(rM[640:695,400:1200],axis=0)    # background noise
diff = spec - bgnd                      # filtering out the  noise



u = np.arange(401,1201,1) # The desired span of the screen 


peaks,_= find_peaks(diff, height = 3,width=5, distance=10)
print ("Peaks details are:",_['peak_heights']) 


plt.figure(dpi=300)

#comments
peak_comments = ["x","h27","h29","h31","h33","h35","h37","..","h41","..","h45"]


for i, peak_index in enumerate(peaks):
    if i < len(peak_comments):
        comment = peak_comments[i]  
        comment_size = 10 
        plt.annotate(f'{comment}', xy=(u[peak_index], diff[peak_index]),
                     xytext=(u[peak_index], diff[peak_index] + 2),
                      arrowprops=dict(facecolor='black', arrowstyle='->'),
                      fontsize=comment_size)
        

plt.plot(u,spec, label='Spectrum')
plt.plot(u,bgnd, label='Background')

plt.plot(u,diff,label='Difference between them')
print(u[peaks])
plt.plot(u[peaks], diff[peaks],".")
plt.title("Power=1.73 W, pressure 30 Torr")

plt.xlabel("Pixels of the MCP-phosphor detector ")
plt.ylabel("Mean values of pixels - Intensity (arb. units)")
plt.grid()

legend = plt.legend(loc='center left')

legend_font_size = 8  
for text in legend.get_texts():
    text.set_fontsize(legend_font_size)


plt.plot(u,np.zeros_like(diff) , "--", color = "gray")



plt.show()


plt.figure(dpi=300)
plt.imshow(rM, cmap='gray') 
plt.title('Original Image')
plt.colorbar()

plt.axhline(y=560, color='r', linestyle='--', label='Start of Spectrum')
plt.axhline(y=610, color='r', linestyle='--', label='End of Spectrum')
plt.axhline(y=640, color='b', linestyle='--', label='Start of Background')
plt.axhline(y=695, color='b', linestyle='--', label='End of Background')
plt.legend()

plt.show()


#%%%
'-------Curve fitting algorithm--------'  





import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
from scipy.signal import find_peaks


c=2.997e8


# Paste the path name of the image  (bd_220819_Power1.73_press35.bmp)
M = plt.imread('bd_220819_Power1.73_press35.bmp',format = None)



M = np.array(M)*1.0                        
rM = rotate(M, -7.0)
spec = np.mean(rM[560:610,400:1200],axis=0)    # spectrum
bgnd = np.mean(rM[640:695,400:1200],axis=0)    # background noise
diff = spec - bgnd                      # filtering out the  noise



u = np.arange(401,1201,1) # the desired span of the screen

plt.figure(dpi=300)
peaks,_= find_peaks(diff, height = 3,width=5, distance=10)
print ("Peaks details are:",_['peak_heights']) 



# comments for each peak
peak_comments = ["21","23","25","h27","h29","h31","h33","h35","h37"]

for i, peak_index in enumerate(peaks):
    if i < len(peak_comments):
        comment = peak_comments[i]   
        comment_size = 10  
        plt.annotate(f'{comment}', xy=(u[peak_index], diff[peak_index]),
                     xytext=(u[peak_index], diff[peak_index] + 2),
                      arrowprops=dict(facecolor='black', arrowstyle='->'),
                      fontsize=comment_size)
        

plt.plot(u,spec, label='Spectrum')
plt.plot(u,bgnd, label='Background')

plt.plot(u,diff,label='Difference between them')
print(u[peaks])
plt.plot(u[peaks], diff[peaks],".")
plt.title("Power=1.73 W, pressure 30 Torr")

plt.xlabel("Pixels of the MCP-phosphor detector ")
plt.ylabel("Mean values of pixels - Intensity (arb. units)")
plt.grid()


legend = plt.legend(loc='center left')


legend_font_size = 8


for text in legend.get_texts():
    text.set_fontsize(legend_font_size)


plt.plot(u,np.zeros_like(diff) , "--", color = "gray")


plt.show()


plt.figure()
plt.imshow(rM, cmap='gray') 
plt.title('Original Image')
plt.colorbar() 
plt.axhline(y=560, color='r', linestyle='--', label='Start of Spectrum')
plt.axhline(y=610, color='r', linestyle='--', label='End of Spectrum')
plt.axhline(y=640, color='b', linestyle='--', label='Start of Background')
plt.axhline(y=695, color='b', linestyle='--', label='End of Background')
plt.legend()

plt.show()


'-------Curve fitting procdure--------'  



# positions ot the peaks:
    
 # --- > peak pixels of 0.84 mJ _ 35 Torr ->>> Also copy the pathe name of the
       # corresponding image in M 'Above' as well!
#peak_pixels = np.array([914,  959, 998, 1031, 1061, 1086, 1108])

    

 # ------ > peak pixels of 1.73 mJ _ 35 Torr 
peak_pixels = np.array([ 873, 928, 971, 1008, 1041, 1066, 1094, 1116])




def func(u,u0,A): # calculates y(x)
    Lambda = 100e-9*np.sin(np.arctan((u-u0)/A) + 30*np.pi/180)
    return Lambda

Lam_0 = 795e-9

N = 21 

wave_length = np.array([1/N,1/(N+2),1/(N+4),1/(N+6),
                        1/(N+8),1/(N+10),1/(N+12),1/(N+14)])*795e-9

u = np.arange(401,1201,1) # the entire span of the screen



#Curve fitting
popt, pcov = curve_fit(func,peak_pixels,wave_length,
                       p0=[ 541.229006  , -1645.32998136] ,maxfev=200000) 


'''
'The value of A is quite big, for it makes the (tan) yields very
 small value in order to prevent a rapid change.'
'''
plt.plot(peak_pixels,(func(peak_pixels,*popt))*\
         10**(9),"X--",
         label="Curve-fitted experimental harmonics peaks $\lambda_N$",
         color='r' )



plt.plot(peak_pixels, (wave_length)*10**(9),"o--",
         label="The expected (theoretical) harmonics $\lambda_N$")

wave_length_labels = [f"$N_1$={N}", f"$N_2$={N+2}", f"$N_3$={N+4}",
                      f"$N_4$={N+6}", f"$N_5$={N+8}", f"$N_6$={N+10}",
                      f"$N_7$={N+12}",f"$N_8$={N+14}",
                      "...","..."]  # to Add comments

for i, (x, y) in enumerate(zip(peak_pixels, wave_length * 10 ** 9)):
    if i < len(wave_length_labels):
        comment = wave_length_labels[i]
        plt.annotate(f'{comment}', xy=(x, y), xytext=(x + 5, y),
                     arrowprops=dict(facecolor='black', arrowstyle='->'))


plt.xlabel("Pixels ")
plt.ylabel("$\lambda$($u_m$) [nm]")
plt.title(f"Theoretical harmonic order $N_1$={N}, P= 1.73 W, Pr=35 Torr")
plt.grid()
plt.legend()
plt.show()



residuals= (func(peak_pixels,*popt))-wave_length

ssr = np.sum(residuals**2)

plt.plot(peak_pixels, residuals, 'o-',
         label=f"Residuals, ($\\mathbf{{SSR: {ssr:.4}}}$)")
plt.axhline(0, color='red', linestyle='--')

plt.xlabel("x")
plt.ylabel("Residuals")
plt.title(f"Residuals results for the prediction ($N_1$: {N})")
plt.grid(True)
plt.legend()
plt.show()





u_values = np.arange(401, 1201, 1) 

wavelengthspan = func(u_values, *popt)  # Calculate wavelengths using func()




plt.plot(wavelengthspan*1e9,diff,label='Difference between them')

plt.xlabel("Wave length [nm]")
plt.ylabel("Intensity (a.u.)")
plt.grid()


peak_comments = ["21","23","25","h27","h29","h31","h33","h35","h37"]
peaks, _ = find_peaks(diff, height=3, width=5, distance=10) 

#   comments  
for i, peak_index in enumerate(peaks):
    if i < len(peak_comments):
        comment = peak_comments[i]
        comment_size = 10  
        plt.annotate(f'{comment}', xy=(wavelengthspan[peak_index] *\
                                       1e9, diff[peak_index]),
                     xytext=(wavelengthspan[peak_index] *\
                             1e9, diff[peak_index] + 2),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=comment_size)

plt.xlim(wavelengthspan[0] * 1e9,
         wavelengthspan[-1] * 1e9) # to flip the wave length

plt.show()




wavelengthspan = func(u_values, *popt)  # Calculate wavelengths using func()

omega = (2*np.pi* c) / wavelengthspan
hbar=6.6e-34*2*np.pi
e = 1.6e-19# electron charge

Energy_eV=(omega*1.05457182e-34)/e
width = 8  
height = 5  


plt.figure(dpi=300)

plt.plot(Energy_eV,diff)

peaks, _ = find_peaks(diff, height=3, width=5, distance=10)  


plt.xlabel("Photon energy [eV]")
plt.ylabel("Intensity (a.u.)")
plt.grid()

peak_comments = ["21","23","25","h27","h29","h31","h33","h35","h37"]

for i, peak_index in enumerate(peaks):
    if i < len(peak_comments):
        comment = peak_comments[i]  
        comment_size = 10  
        plt.annotate(f'{comment}', xy=(Energy_eV[peak_index],
                                       diff[peak_index]),
                     xytext=(Energy_eV[peak_index], diff[peak_index] + 2),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=comment_size)
plt.show()


# two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),dpi=300)

ax1.plot(peak_pixels, (func(peak_pixels, *popt)) * 10**(9), "X--",
         label="Curve-fitted experimental harmonics peaks $\lambda_N$",
         color='r')

ax1.plot(peak_pixels, (wave_length) * 10**(9), "o--",
         label="The expected (theoretical) harmonics $\lambda_N$")

wave_length_labels = [f"$N_1$={N}", f"$N_2$={N+2}", f"$N_3$={N+4}",
                      f"$N_4$={N+6}", f"$N_5$={N+8}", f"$N_6$={N+10}",
                      f"$N_7$={N+12}",f"$N_8$={N+14}", "...","..."]

for i, (x, y) in enumerate(zip(peak_pixels, wave_length * 10 ** 9)):
    if i < len(wave_length_labels):
        comment = wave_length_labels[i]
        ax1.annotate(f'{comment}', xy=(x, y), xytext=(x + 5, y),
                     arrowprops=dict(facecolor='black', arrowstyle='->'))

ax1.set_xlabel("Horizontal axis of the detector")
ax1.set_ylabel("$\lambda$($u_m$) [nm]")

ax1.grid()
ax1.legend()



residuals = (func(peak_pixels, *popt)) - wave_length
ssr = np.sum(residuals**2)

ax2.plot(peak_pixels, residuals, 'o-',
         label=f"Residuals, ($\\mathbf{{SSR: {ssr:.4}}}$)")

ax2.axhline(0, color='red', linestyle='--')

ax2.set_xlabel("Horizontal axis of the detector ")
ax2.set_ylabel("Residuals")
ax2.grid()
ax2.legend()

plt.tight_layout()

plt.show()



fig, ax1 = plt.subplots(figsize=(7, 5),dpi=300)

ax1.plot(peak_pixels, (func(peak_pixels, *popt)) * 10**(9), "X--",
         label="Curve-fitted experimental harmonics peaks $\lambda_N$",
         color='r')
ax1.plot(peak_pixels, (wave_length) * 10**(9), "o--",
         label="The expected (theoretical) harmonics $\lambda_N$")

wave_length_labels = [f"$N_1$={N}", f"$N_2$={N+2}", f"$N_3$={N+4}",
                      f"$N_4$={N+6}", f"$N_5$={N+8}", f"$N_6$={N+10}",
                      f"$N_7$={N+12}",f"$N_8$={N+14}",
                      "...","..."]  # to add comments

for i, (x, y) in enumerate(zip(peak_pixels, wave_length * 10 ** 9)):
    if i < len(wave_length_labels):
        comment = wave_length_labels[i]
        ax1.annotate(f'{comment}', xy=(x, y), xytext=(x + 5, y),
                     arrowprops=dict(facecolor='black', arrowstyle='->'))

ax1.set_xlabel("Horizontal axis of the detector")
ax1.set_ylabel("$\lambda$($u_m$) [nm]")
ax1.grid()
ax1.legend()

ax2 = ax1.twinx()

residuals = (func(peak_pixels, *popt)) - wave_length
ssr = np.sum(residuals**2)

ax2.plot(peak_pixels, residuals, 'o--',
         label=f"Residuals, ($\\mathbf{{SSR: {ssr:.4}}}$)", color="g")

ax2.axhline(0, color='red', linestyle='--')

ax2.set_ylabel("Residuals")

ax2.legend(loc="lower left")

plt.tight_layout()

plt.show()




#%%%

'-----------------------1st 800 nm Phase matching Maps -----------------------'


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap


# Paste the path of the intensities named:
excel_file = 'Intensities_ 800nm(1)_0.84_1.28_1.73.xlsx'

xldata = pd.read_excel(excel_file,
                       sheet_name = "P173",
                       index_col = 0)

xl_clean = xldata.replace(np.NaN, 0)


print(xl_clean)

heatmap_data = xl_clean
plt.figure(dpi=300)

sns.heatmap(heatmap_data, cbar_kws={'label': 'Intesity (arb. units)'},
            annot=False, vmin=0, vmax=35)

plt.tight_layout()

plt.xlabel("Harmonic number ")
plt.ylabel("Pressure [Torr] ")
plt.title("Driving laser pulse energy 1.73 mJ")


plt.show()

#----------------------------------------------------------
#----------------------------------------------------------




# Please add the path name of the file: P_0.84_1.28_1.73.xlsx
excel_file = 'Path of the file: Peak sheet P_0.84_1.28_1.73.xlsx'

xldata = pd.read_excel(excel_file,
                       sheet_name = "P128",
                       index_col = 0)

xl_clean2 = xldata.replace(np.NaN, 0)


print(xl_clean)

heatmap_data = xl_clean2
plt.figure(dpi=300)

sns.heatmap(heatmap_data, cbar_kws={'label': 'Intesity (arb. units)'},
            annot=False, vmin=0, vmax=35)

plt.tight_layout()

plt.xlabel("Harmonic number ")
plt.ylabel("Pressure [Torr] ")
plt.title("Driving laser pulse energy 1.28 mJ")


plt.show()


#----------------------------------------------------------
#----------------------------------------------------------


# Please add the path name of the file: P_0.84_1.28_1.73.xlsx
excel_file = 'Path of the file: Peak sheet P_0.84_1.28_1.73.xlsx'

xldata = pd.read_excel(excel_file,
                       sheet_name = "P0,84_2",
                       index_col = 0)

xl_clean3 = xldata.replace(np.NaN, 0)


print(xl_clean)

heatmap_data = xl_clean3
plt.figure(dpi=300)

sns.heatmap(heatmap_data,  cbar_kws={'label': 'Intesity (arb. units)'},
            annot=False,vmin=0, vmax=35)


plt.tight_layout()


plt.xlabel("Harmonic number ")
plt.ylabel("Pressure [Torr] ")
plt.title("Driving laser pulse energy 0.84 mJ")


plt.show()


#----------------------------------------------------------
#----------------------------------------------------------


# Please add the path name of the file: P_0.84_1.28_1.73.xlsx
excel_file = 'Path of the file: Peak sheet P_0.84_1.28_1.73.xlsx'

xldata = pd.read_excel(excel_file,
                       sheet_name = "P0,4",
                       index_col = 0)

xl_clean2 = xldata.replace(np.NaN, 0)


print(xl_clean)

heatmap_data = xl_clean2
plt.figure(dpi=300)

sns.heatmap(heatmap_data, cbar_kws={'label': 'Intesity (arb. units)'},
            annot=False, vmin=0, vmax=35)

plt.tight_layout()

plt.xlabel("Harmonic number ")
plt.ylabel("Pressure [Torr] ")
plt.title("Driving laser pulse energy 0.4 mJ")


plt.show()










#%%





'-----------------------Image Averiging-----------------------'

' This code is used to average both the 800 nm shots and the 400 nm shots '






import os
import numpy as np
from PIL import Image

# Specify the folder path where the images are located
folder_path = " path of the foler where the single shots are!"

# Specify the path to the destination folder where the
# averaged images will be stored
destination_folder = "The path of the output folder"

os.chdir(folder_path)

all_files = os.listdir()
image_files = [filename for filename in all_files if\
               filename.endswith((".png", ".PNG"))]

image_groups = {}

for image_file in image_files:
    # identify the hour-minute name from the file name
    hour_minute = image_file.split("_")[2][:4] 
    
    if hour_minute in image_groups:
        image_groups[hour_minute].append(image_file)
    else:
        image_groups[hour_minute] = [image_file]

os.makedirs(destination_folder, exist_ok=True)

for group_name, group_files in image_groups.items():
    image_path = group_files[0]
    image = Image.open(image_path)
    width, height = image.size
    num_images = len(group_files)

    average_image = np.zeros((height, width, 3), dtype=np.float32)

    for image_file in group_files:
        try:
            image = Image.open(image_file)
            image_array = np.array(image, dtype=np.float32)
            expected_shape = (height, width, 3)
            if image_array.shape != expected_shape:
                # Check if image has only one channel
                if image_array.ndim == 2:
                    image_array = np.stack((image_array,) * 3, axis=-1)
                else:
                    raise ValueError(f"Shape {image_array.shape} does not\
                                     match the expected dimensions\
                                         {expected_shape}.")
                                         
            average_image += image_array
        except (IOError, ValueError) as e:
            print(f"Skipping {image_file}. Error: {str(e)}")

    average_image /= num_images

    average_image = np.round(average_image).astype(np.uint8)

    result_image = Image.fromarray(average_image, mode="RGB")

    result_image_path = os.path.join(destination_folder,
                                     f"average_{group_name}.png")
    
    result_image.save(result_image_path)





#%%
# ----------------Curve fitting of the 800_2 ----------------





import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
from scipy.signal import find_peaks



c=2.997e8



# Paste the path name of the image  (average_1155.png)
M = plt.imread('average_1155.png',format = None)



M = np.array(M)*1.0                        
rM = rotate(M, 0.0)



spec_800_2 = np.mean(rM[290:410, 400:1200],axis=0)    # spectrum
bgnd_800_2 = np.mean(rM[420:460, 400:1200],axis=0)    # background noise


reduced_spec_800_2 = spec_800_2[:, 0]*255
reduced_bgnd_800_2 = bgnd_800_2[:, 0]*255

diff_800_2 = reduced_spec_800_2 - reduced_bgnd_800_2  # filtering out the noise



u = np.arange(400,1200,1) # the desired span of the screen


peaks800_2,_= find_peaks(diff_800_2, height =3,width=6, distance=18)




rounded_peak_heights = [round(val, 4) for val in _['peak_heights']]
print("Peaks details are:", rounded_peak_heights)






# comments 
plt.figure(dpi=300)

peak_comments = ["h21","h23","h25","h27","h29"]

for i, peak_index in enumerate(peaks800_2):
    if i < len(peak_comments):
        comment = peak_comments[i]  
        comment_size = 10  
        plt.annotate(f'{comment}', xy=(u[peak_index], diff_800_2[peak_index]),
                     xytext=(u[peak_index], diff_800_2[peak_index] + 4),
                      arrowprops=dict(facecolor='black', arrowstyle='-'),
                      fontsize=comment_size)
        

plt.plot(u,reduced_spec_800_2, label='Spectrum')
plt.plot(u,reduced_bgnd_800_2, label='Background')

plt.plot(u,diff_800_2,label='Difference between them')
plt.plot(u[peaks800_2], diff_800_2[peaks800_2],".")

plt.xlabel("Pixels of the MCP-phosphor detector ")
plt.ylabel("Mean values of pixels - Intensity (arb. units)")
plt.grid()

legend = plt.legend(loc='center left')

legend_font_size = 8

for text in legend.get_texts():
    text.set_fontsize(legend_font_size)


plt.plot(u,np.zeros_like(diff_800_2) , "--", color = "gray")

plt.show()


plt.figure(dpi=300)
plt.imshow(rM, cmap='gray')  
plt.colorbar()  
plt.axhline(y=280, color='r', linestyle='--', label='Spectrum')
plt.axhline(y=418, color='r', linestyle='--')
plt.axhline(y=420, color='b', linestyle='--')
plt.axhline(y=460, color='b', linestyle='--', label='Background')
legend = plt.legend(loc='lower left')

plt.legend()

plt.show()







### -------------------------------> The curve fit of the 2nd 800nm --> 1.8 mJ 

  
peak_pixels_800_2 = np.array ([840, 884, 923, 956, 986])#800nm_2 




def func(u,u0,A): # calculates y(x)
    Lambda = 100e-9*np.sin(np.arctan((u-u0)/A) + 30*np.pi/180)
    return Lambda

Lam_0 = 795e-9

N = 21 # The starting number of the Harmonic sequence

wave_length_800_2 = np.array([1/N,1/(N+2),1/(N+4),1/(N+6),1/(N+8)])*795e-9
u = np.arange(401,1201,1) # the desired span of the screen

popt_800_2, pcov = curve_fit(func,peak_pixels_800_2 ,wave_length_800_2,
                             p0=[ 670.01296871, -1253.31351353] ,maxfev=200000) 



# Array of pixel positions for the desired span of the screen
u_values = np.arange(401, 1201, 1)

 # Calculate wavelengths using func()
wavelengthspan_800_2 = func(u_values, *popt_800_2) 



plt.xlim(wavelengthspan_800_2[0] *\
         1e9, wavelengthspan_800_2[-1] * 1e9) # to flip the wave length

plt.plot(wavelengthspan_800_2*1e9,diff_800_2,label='Difference between them')

plt.xlim(50,20)
plt.xlabel("Wave length [nm]")
plt.ylabel("Intensity (a.u.)")
plt.grid()


peak_comments = ["h21","h23","h25","h27","h29"]
peaks, _ = find_peaks(diff_800_2, height=3, width=6, distance=18)

#   comments  
for i, peak_index in enumerate(peaks):
    if i < len(peak_comments):
        comment = peak_comments[i] 
        comment_size = 10  
        plt.annotate(f'{comment}', xy=(wavelengthspan_800_2[peak_index] * 1e9,
                                       diff_800_2[peak_index]),
                     xytext=(wavelengthspan_800_2[peak_index] * 1e9,
                             diff_800_2[peak_index] + 2),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=comment_size)

plt.show()



# Calculate wavelengths using func()
wavelengthspan_800_2 = func(u_values, *popt_800_2) 

omega_800_2 = (2*np.pi* c) / wavelengthspan_800_2
hbar=6.6e-34*2*np.pi
e = 1.6e-19

Energy_eV_800_2=(omega_800_2*1.05457182e-34)/e
width = 8  # Width in inches
height = 5  # Height in inches

figure_size = (8, 6) 

plt.figure(figsize=figure_size,dpi=300)

plt.plot(Energy_eV_800_2,diff_800_2)

peaks, _ = find_peaks(diff_800_2, height=3, width=6, distance=18) 




peak_comments = ["h21","h23","h25","h27","h29"]

for i, peak_index in enumerate(peaks):
    if i < len(peak_comments):
        comment = peak_comments[i]  
        comment_size = 10  
        plt.annotate(f'{comment}', xy=(Energy_eV_800_2[peak_index],
                                       diff_800_2[peak_index]),
                     xytext=(Energy_eV_800_2[peak_index],
                             diff_800_2[peak_index] + 2),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=comment_size)


plt.xlim(25,60)
plt.xlabel("Photon energy [eV]")
plt.ylabel("Intensity (a.u.)")
plt.grid()
plt.show()









figure_size = (8, 6)  

plt.figure(figsize=figure_size,dpi=300)

# to flip the wave length
plt.xlim(wavelengthspan_800_2[0] * 1e9, wavelengthspan_800_2[-1] * 1e9) 




# Calculate wavelengths using func()
wavelengthspan_800_2 = func(u_values, *popt_800_2)  

omega_800_2 = (2*np.pi* c) / wavelengthspan_800_2
hbar=6.6e-34*2*np.pi
e = 1.6e-19

Energy_eV_800_2=(omega_800_2*1.05457182e-34)/e
width = 8  # Width in inches
height = 5  # Height in inches



plt.plot(Energy_eV_800_2,diff_800_2)

peaks,_= find_peaks(diff_800_2, height =3,width=6, distance=18)


plt.xlabel("Photon energy [eV]")
plt.ylabel("Intensity (a.u.)")
plt.grid()
plt.xlim(20,50)

peak_comments = ["h21","h23","h25","h27","h29"]

for i, peak_index in enumerate(peaks):
    if i < len(peak_comments):
        comment = peak_comments[i] 
        comment_size = 10
        plt.annotate(f'{comment}', xy=(Energy_eV_800_2[peak_index],
                                       diff_800_2[peak_index]),
                     xytext=(Energy_eV_800_2[peak_index],
                             diff_800_2[peak_index] + 2),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=comment_size)
plt.show()







u_values = np.arange(401, 1201, 1) 

# Calculate wavelengths using func()
wavelengthspan_800_2 = func(u_values, *popt_800_2)  



plt.figure(dpi=300)

plt.plot(Lam_0/wavelengthspan_800_2,diff_800_2,label='800_2 vs orders')



plt.xlim(15,35)

plt.legend()
plt.xlabel("Harmonics orders")
plt.ylabel("Intensity (a.u.)")
plt.grid()


#%%


'-------Heat maps 800_2-------'  

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap





# Paste the path of the intensities named:
excel_file = 'Intensities_800nm(2)_1.44_1.8_2.16.xlsx'


xldata = pd.read_excel(excel_file,
                       sheet_name = "1.8Watt",
                       index_col = 0)

xl_clean = xldata.replace(np.NaN, 0)


print(xl_clean)

heatmap_data = xl_clean
plt.figure(dpi=300)

sns.heatmap(heatmap_data, cbar_kws={'label': 'Intesity (arb. units)'},
            annot=False, vmin=0, vmax=45)

plt.tight_layout()

plt.xlabel("Harmonic number ")
plt.ylabel("Pressure [Torr] ")
plt.title("Driving laser pulse energy  1.8 mJ")



#----------------------------------------------------------
#----------------------------------------------------------






xldata = pd.read_excel(excel_file,
                       sheet_name = "1.4Watt",
                       index_col = 0)

xl_clean2 = xldata.replace(np.NaN, 0)


print(xl_clean)

heatmap_data = xl_clean2
plt.figure(dpi=300)

sns.heatmap(heatmap_data, cbar_kws={'label': 'Intesity (arb. units)'},
            annot=False, vmin=0, vmax=45)

plt.tight_layout()

plt.xlabel("Harmonic number ")
plt.ylabel("Pressure [Torr] ")
plt.title("Driving laser pulse energy  1.44 mJ")




#----------------------------------------------------------
#----------------------------------------------------------



xldata = pd.read_excel(excel_file,
                       sheet_name = "2.16Watt",
                       index_col = 0)

xl_clean3 = xldata.replace(np.NaN, 0)


print(xl_clean)

heatmap_data = xl_clean3
plt.figure(dpi=300)

sns.heatmap(heatmap_data,  cbar_kws={'label': 'Intesity (arb. units)'},
            annot=False,vmin=0, vmax=45)


plt.tight_layout()


plt.xlabel("Harmonic number ")
plt.ylabel("Pressure [Torr] ")
plt.title("Driving laser pulse energy 2.16 mJ")


plt.show()

#%%%


'--------------- 400 nm interpolation -------------------'






import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


c=2.997e8



# Paste the path name of the image  (average_1912.png)
M_400_1912 = plt.imread('Path name of the  ... average_1912.png',format = None)


rM_400_1912 = rotate(M_400_1912, 0)

print(np.max(M_400_1912))






u = np.arange(1,1281,1) # the entire span of the screen


spec_400_1912 = np.mean(rM_400_1912[280:420, 0:1281], axis=0)# spectrum
bgnd_400_1912 = np.mean(rM_400_1912[420:460, 0:1281], axis=0)# background noise

reduced_spec_400_1912 = spec_400_1912[:,0]
reduced_bgnd_400_1912 = bgnd_400_1912[:,0]

# filtering out the noise
diff_400_1912 = reduced_spec_400_1912 - reduced_bgnd_400_1912    

plt.figure(dpi=300)



plt.plot(u, reduced_spec_400_1912*255, label='Spectrum')
plt.plot(u, reduced_bgnd_400_1912*255, label='Background')
plt.plot(u, diff_400_1912*255, label='Difference')
plt.ylabel("Intensity a.u.")
plt.xlabel("Horisental axis of the CCD camera")

plt.legend()
plt.grid()
plt.show()




plt.figure(dpi=300)
plt.imshow(rM_400_1912, cmap='gray')
plt.colorbar() 
plt.axhline(y=280, color='r', linestyle='--', label='Spectrum')
plt.axhline(y=418, color='r', linestyle='--')
plt.axhline(y=420, color='b', linestyle='--')
plt.axhline(y=460, color='b', linestyle='--', label='Background')
plt.legend()

plt.show()



#%%%


'-------------- Interpolation of the HHG from the 400 nm curves---------------'

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
from scipy.signal import find_peaks
from scipy.interpolate import interp1d 
import os




# Paste the path name of the folder that contained the averaged 400 nm images
folder = 'Path name of the folder that contains the 400 nm averaged images'

all_files = os.listdir(folder)

# Filter for images
average_img_files = [f for f in all_files if f.startswith("average_") and\
                     f.endswith(".png")]

average_img_files = sorted(average_img_files, key=lambda\
                           x: int(x.split('_')[1].split('.png')[0]))
    
average_s = []

for img_file in average_img_files:
    img_path = os.path.join(folder, img_file)
    img = plt.imread(img_path, format=None)
    
    img_s = np.mean(img[280:420, :], axis=0)
    img_s =  img_s[:, 0]
    average_s.append(img_s - np.mean(img_s[1020:1080]))



average_b = average_s[38].copy()
average_b[558:591] = np.interp(np.arange(558,591),[558,591],
                               [average_s[38][558],average_s[38][591]])

average_b[742:767] = np.interp(np.arange(742,767),[742,767],
                               [average_s[38][742],average_s[38][767]])
b_mean = np.mean(average_b[460:520])



average_spec=[]
for s in average_s:
 spec = s - average_b*np.mean(s[460:520])/b_mean
 average_spec.append(spec)



plt.figure(dpi=300)
plt.ylabel("Inetnsity a.u.")
plt.xlabel("Horisental axis of the CCD camera")
plt.grid()
for s in average_s:
    plt.plot(s*255)
    
    
    
plt.figure(dpi=300,)       ## All the spectra against the 

plt.xlabel("Horizontal axis of the CCD camera")
plt.ylabel("Inetnsity a.u.")


for s in average_spec:
    plt.plot(s*255)
plt.grid()    
    




# Paste the path name of the image  (average_1155.png)
M_800_2 = plt.imread('Path name of the  ... average_1155.png',format = None)




M_800_2 = np.array(M_800_2)*1.0                        
rM_800_2 = rotate(M_800_2, 0.0)



spec_800_2 = np.mean(rM_800_2[290:410, 0:1280],axis=0)    # spectrum
bgnd_800_2 = np.mean(rM_800_2[420:460, 0:1280],axis=0)    # background noise

reduced_spec = spec_800_2[:, 0]
reduced_bgnd = bgnd_800_2[:, 0]

diff_800_2 = reduced_spec - reduced_bgnd 






peak_pixels_800_2 = np.array ([840,884,923,956,986])#800nm_2 


def func_800_2(u,u0,A): # calculates y(x)
    Lambda = 100e-9*np.sin(np.arctan((u-u0)/A) + 30*np.pi/180)
    return Lambda

Lam_0 = 795e-9

N = 21 # The starting number of the Harmonic sequence

wave_length_800_2 = np.array([1/N,1/(N+2),1/(N+4),1/(N+6),1/(N+8)])*795e-9


popt_800_2, pcov = curve_fit(func_800_2,peak_pixels_800_2,wave_length_800_2,
                             p0=[   670.01296871, -1253.31351353] 
                             ,maxfev=200000) 




u = np.arange(1,1281,1) # the desired span of the screen


# Calculate wavelengths using func()
wavelengthspan_800_2 = func_800_2(u, *popt_800_2)  



plt.grid()

plt.figure(dpi=300)
plt.xlim(13, 25)


for s in average_spec:
    #s1=s1+s
    plt.plot(Lam_0/wavelengthspan_800_2, s * 255)

    
plt.xlabel("Harmonic order")
plt.ylabel("Intensity a.u. ")
plt.xlim(17.7,18.7)

plt.ylim(-3,10)

plt.grid()
plt.show()




plt.figure(dpi=300)
plt.xlim(13, 25)


for s in average_spec:
    plt.plot(Lam_0/wavelengthspan_800_2, s * 255) 

    
plt.xlabel("Harmonic order")
plt.ylabel("Intensity a.u. ")

plt.grid()
plt.show()













file_name = 'average_1912.png'

if file_name in average_img_files:
    file_number = average_img_files.index(file_name)
else:
    print("File not found.")
    exit()

peaks, properties = find_peaks(average_spec[file_number] * 255, height=1,
                               width=1, distance=10)

print("Peak heights are------------------->>>>",  properties['peak_heights'])



plt.figure(dpi=300)

plt.plot(Lam_0 / wavelengthspan_800_2, average_spec[file_number] * 255,
         label="$\lambda_{pulse}$ = 400 nm\nE$_{pulse}$ = 0.46 mJ\nPressure\
             = 20 Torr\nGDD = -300\nTOD = 10000")  #f"{file_name}"
             
plt.legend()

plt.xlabel("Harmonic order")
plt.ylabel("Intensity a.u. ")
plt.xlim(11,45)

plt.grid()
plt.show()






plt.figure(dpi=300)
plt.grid()
plt.ylabel("Intensity a.u.")

plt.xlabel("Horizontal axis of the CCD camera")

plt.plot(average_b*255,label="Created background",zorder =2)
plt.plot(average_s[38]*255, label="Origional spectrum",zorder=1) 
plt.plot((average_s[38]-average_b)*255,label="Difference")

plt.legend(loc="lower center")




plt.figure(dpi=300)

for s in average_s:
    plt.plot(s*255)
    

plt.ylabel("Inetnsity a.u.")
plt.xlabel("Horisental axis of the CCD camera")
plt.grid()



interpolated_backgrounds = []

for average in average_s:
    average_b = average.copy()
    average_b[558:591] = np.interp(np.arange(558, 591), [558, 591],
                                   [average[558], average[591]])
    
    average_b[742:767] = np.interp(np.arange(742, 767), [742, 767],
                                   [average[742], average[767]])
    b_mean = np.mean(average_b[460:520])

    interpolated_backgrounds.append(average_b)

plt.figure(dpi=300)
plt.ylabel("Intensity a.u.")
plt.xlabel("Horizontal axis of the CCD camera")
plt.grid()
for bg in interpolated_backgrounds:
    plt.plot(bg * 255)
plt.show()






'-------ALL THE SPECTRA, 800_1 , 800_2, 400 nm-------'  

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import rotate
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d 
import os


#### 1.73 ###'


# Paste the path name of the image  (Abd_220819_Power1.73_press35.bmp)
M_173 = plt.imread('Abd_220819_Power1.73_press35.bmp',format = None)




M_173 = np.array(M_173)*1.0                        
rM_173 = rotate(M_173, -7.0)
spec_173 = np.mean(rM_173[560:610,0:1280],axis=0)    # spectrum
bgnd_173 = np.mean(rM_173[650:695,0:1280],axis=0)    # background noise
diff_173 = spec_173 - bgnd_173                      # filtering out the  noise



u = np.arange(1,1281,1) # the desired span of the screen

def func_173(u,u0,A):
    Lambda = 100e-9*np.sin(np.arctan((u-u0)/A) + 30*np.pi/180)
    return Lambda

Lam_0 = 795e-9


N = 21 # The starting number of the Harmonic sequence

peak_pixels_173 = np.array([ 873 , 928,  971, 1008, 1041, 1066, 1094, 1116])

wave_length_173 = np.array([1/N,1/(N+2),1/(N+4),1/(N+6),1/(N+8),1/(N+10),
                            1/(N+12),1/(N+14)])*795e-9


popt_173, pcov = curve_fit(func_173,peak_pixels_173,wave_length_173,
                           p0=[  657.95177175, -1486.51659083] ,maxfev=200000) 




u_values = np.arange(1, 1281, 1) 

# Calculate wavelengths using func()
wavelengthspan_173 = func_173(u_values, *popt_173)  




plt.figure(dpi=300)





##### 800 nm 1.28 mJ _ 35 Torr ######


c=2.997e8


# Paste the path name of the image  (Abd_220819_Power1.28_press35.bmp)
M_128 = plt.imread('Abd_220819_Power1.28_press35.bmp',format = None)


M_128 = np.array(M_128)*1.0                        
rM_128 = rotate(M_128, -7.0)


spec_128 = np.mean(rM_128[560:610,0:1280],axis=0)    # spectrum
bgnd_128 = np.mean(rM_128[640:695,0:1280],axis=0)    # background noise
diff_128 = spec_128 - bgnd_128  



peak_pixels_128 = np.array([869, 925, 969, 1005, 1038, 1065, 1092]) 


def func_128(u,u0,A): # calculates y(x)
    Lambda = 100e-9*np.sin(np.arctan((u-u0)/A) + 30*np.pi/180)
    return Lambda

Lam_0 = 795e-9

N = 21 # The starting number of the Harmonic sequence

wave_length_128 = np.array([1/N,1/(N+2),1/(N+4),1/(N+6),1/(N+8),1/(N+10),
                            1/(N+12)])*795e-9


popt_128, pcov = curve_fit(func_128,peak_pixels_128,wave_length_128,
                           p0=[ -3.40548495e+03, -1.47202285e+10] ,
                           maxfev=50000000) 

# Calculate wavelengths using func()
wavelengthspan_128 = func_128(u_values, *popt_128)  






######### 0.84 ########


# Paste the path name of the image  (Abd_220819_Power0.84_press35.bmp)
M_084 = plt.imread('Abd_220819_Power0.84_press35.bmp',format = None)







M_084 = np.array(M_084)*1.0                        
rM_084 = rotate(M_084, -7.0)
spec_084 = np.mean(rM_084[560:610,0:1280],axis=0)    # spectrum
bgnd_084 = np.mean(rM_084[640:695,0:1280],axis=0)    # background noise
diff_084 = spec_084 - bgnd_084  

# ------ > 0.84 _ 35 Torr
peak_pixels_084 = np.array([916,  960, 1000, 1032, 1061, 1087, 1107]) 


def func_084(u,u0,A): # calculates y(x)
    Lambda = 100e-9*np.sin(np.arctan((u-u0)/A) + 30*np.pi/180)
    return Lambda

Lam_0 = 795e-9

N = 23 #The starting number of the Harmonic sequence

wave_length_084 = np.array([1/N,1/(N+2),1/(N+4),1/(N+6),1/(N+8),1/(N+10),
                            1/(N+12)])*795e-9


popt_084, pcov = curve_fit(func_084,peak_pixels_084,wave_length_084,
                           p0=[  -3.40548495e+03, -1.47202285e+12] ,
                           maxfev=200000) #fit the difference in harmonics to 2


# Calculate wavelengths using func()
wavelengthspan_084 = func_084(u_values, *popt_084)  










Lam_0/wavelengthspan_173,

plt.plot(Lam_0/wavelengthspan_173,diff_173,label='800 nm, 1.73 mJ, 35 Torr')

plt.plot(Lam_0/wavelengthspan_173,diff_128,label='800 nm, 1.28 mJ, 35 Torr')

#plt.plot(Lam_0/wavelengthspan_173,diff_084,label='800 nm, 0.84 mJ, 35 Torr')









#### --------------------------800 _ 2 _ 1155







# Paste the path name of the image  (average_1155.png)

M_800_2 = plt.imread('average_1155.png', format=None) # 800nm_2


M_800_2 = np.array(M_800_2)*1.0                        
rM_800_2 = rotate(M_800_2, 0.0)



spec_800_2 = np.mean(rM_800_2[290:410, 0:1280],axis=0)    # spectrum
bgnd_800_2 = np.mean(rM_800_2[420:460, 0:1280],axis=0)    # background noise

reduced_spec = spec_800_2[:, 0]
reduced_bgnd = bgnd_800_2[:, 0]

diff_800_2 = reduced_spec - reduced_bgnd 





# spec_800_2 = np.mean(rM_800_2[280:420, 0:1280],axis=0)    # spectrum
# bgnd_800_2 = np.mean(rM_800_2[420:460, 0:1280],axis=0)    # background noise

# diff_800_2 = spec_800_2 - bgnd_800_2  


peak_pixels_800_2 = np.array ([840,884,923,956,986])#800nm_2 


def func_800_2(u,u0,A): # calculates y(x)
    Lambda = 100e-9*np.sin(np.arctan((u-u0)/A) + 30*np.pi/180)
    return Lambda

Lam_0 = 795e-9

N = 21 #The starting number of the Harmonic sequence

wave_length_800_2 = np.array([1/N,1/(N+2),1/(N+4),1/(N+6),1/(N+8)])*795e-9


popt_800_2, pcov = curve_fit(func_800_2,peak_pixels_800_2,wave_length_800_2,
                             p0=[670.01296871, -1253.31351353] ,maxfev=200000) 


# Calculate wavelengths using func()
wavelengthspan_800_2 = func_800_2(u_values, *popt_800_2) 





# Paste the path name of the folder  (310523_averaged images of 270423_correct)

folder = '310523_averaged images of 270423_correct'

all_files = os.listdir(folder)

# Filter for images (assuming they are png files; adjust as needed)
average_img_files = [f for f in all_files if f.startswith("average_") and\
                     f.endswith(".png")]

average_img_files = sorted(average_img_files, key=lambda\
                           x: int(x.split('_')[1].split('.png')[0]))
average_s = []

for img_file in average_img_files:
    img_path = os.path.join(folder, img_file)
    img = plt.imread(img_path, format=None)
    
    img_s = np.mean(img[280:420, :], axis=0)
    img_s =  img_s[:, 0]
    average_s.append(img_s - np.mean(img_s[1020:1080]))



average_b = average_s[38].copy()
average_b[558:591] = np.interp(np.arange(558,591),[558,591],
                               [average_s[38][558],average_s[38][591]])

average_b[742:767] = np.interp(np.arange(742,767),[742,767],
                               [average_s[38][742],average_s[38][767]])
b_mean = np.mean(average_b[460:520])



average_spec=[]
for s in average_s:
 spec = s - average_b*np.mean(s[460:520])/b_mean #- average_b0
 average_spec.append(spec)





M_084 = np.array(M_084)*1.0                        
rM_084 = rotate(M_084, -7.0)
spec_084 = np.mean(rM_084[560:610,0:1280],axis=0)    # spectrum
bgnd_084 = np.mean(rM_084[640:695,0:1280],axis=0)    # background noise
diff_084 = spec_084 - bgnd_084  







file_name = 'average_1913.png'

# Find the index of the file_name in average_img_files
if file_name in average_img_files:
    file_number = average_img_files.index(file_name)
else:
    print("File not found.")
    exit()

# Find peaks

# Print peak heights
#print("Peak heights are-------------------->>>>",  properties['peak_heights'])



#Nfit(np.arange(0,1280)*1.1-100,*p)+p[2]





peak_pixels_400_1912 = np.array([583, 764]) # ------ >400_1912Torr


def func_400_1912(u,u0,A): # calculates y(x)
    Lambda = 100e-9*np.sin(np.arctan((u-u0)/A) + 30*np.pi/180)
    return Lambda

Lam_0 = 795e-9


N = 14 #The starting number of the Harmonic sequence


wave_length_400_1912 = (np.array([1/N,1/(N+4)])*795e-9)/2


popt_400_1912, pcov = curve_fit(func_400_1912,peak_pixels_400_1912,
                                wave_length_400_1912,
                                p0=[665.78698895, -1271.77440922] ,
                                maxfev=2000000000) 




# Calculate wavelengths using func()
wavelengthspan_400_1912 = func_400_1912(u_values, *popt_400_1912)  


file_name = 'average_1914.png'

if file_name in average_img_files:
    file_number = average_img_files.index(file_name)
else:
    print("File not found.")
    exit()



# Plot peaks









#plt.figure(dpi=300)


plt.plot(Lam_0/wavelengthspan_800_2,diff_800_2*255,
         label='800 nm, 1.8 mJ, 21 Torr')

plt.plot(Lam_0/wavelengthspan_800_2,average_spec[file_number] * 255,
         label='400 nm, 0.46 mJ, 10 Torr ' ) #label=f"{file_name}"







#plt.xlabel("CCD pixels")


plt.xlabel("High harmonic order")

#plt.xlabel("Pixels of the CCD")

plt.ylabel("Intensity (a.u.)")

plt.grid()
plt.legend(loc='upper right', fontsize=8)
plt.xlim(10, 40)

plt.show()























omega_173 = (2*np.pi* c) / wavelengthspan_173
hbar=6.6e-34*2*np.pi#     1.05457182e-34 #j.s
e = 1.6e-19# electron charge

Energy_eV_173=(omega_173*1.05457182e-34)/e


plt.figure(dpi=300)


plt.plot(Energy_eV_173,diff_173,label='800 nm, 1.73 mJ, 35 Torr')

#plt.plot(Energy_eV_173,diff_128,label='800 nm, 1.28 mJ, 35 Torr')

#plt.plot(Energy_eV_173,diff_084,label='800 nm, 0.84 mJ, 35 Torr')
















#--------------------------

omega_800_2 = (2*np.pi* c) / wavelengthspan_800_2
hbar=6.6e-34*2*np.pi#     1.05457182e-34 #j.s
e = 1.6e-19# electron charge
Energy_eV_800_2=(omega_800_2*1.05457182e-34)/e




omega_400 = (2*np.pi* c) / (wavelengthspan_400_1912)
hbar=6.6e-34*2*np.pi#     1.05457182e-34 #j.s
e = 1.6e-19# electron charge
Energy_eV_400=(omega_400*1.05457182e-34)/e







plt.plot(Energy_eV_800_2,average_spec[file_number] * 255,'b' ,
         label='400 nm, 0.46 mJ, 10 Torr ' ) #label=f"{file_name}"

plt.plot(Energy_eV_800_2,diff_800_2* 255, 'r',label='800 nm, 1.8 mJ, 21 Torr' )







# Create the figure with the specified size




# Create the figure with the specified size
#figure_size = (8, 6)  # Adjust the width and height as needed

#figure_size = (5, 3)  # Adjust the width and height as needed

plt.xlabel("Photon energy")
plt.xlim(15,80)

#plt.ylim(-3,)
#plt.xlabel("Pixels of the CCD")

plt.ylabel("Intensity (a.u.)")

plt.grid()
plt.legend(loc='upper right', fontsize=8)

plt.show()






plt.figure(dpi=300)
#figure_size = (5, 1)  # Adjust the width and height as needed

plt.plot(Energy_eV_800_2,average_spec[file_number] * 255,'b' ,
         label='400 nm, 0.46 mJ, 10 Torr ' ) #label=f"{file_name}"


plt.xlabel("Photon energy")
#plt.xlim(25,80)

#plt.ylim(-3,)
#plt.xlabel("Pixels of the CCD")

plt.ylabel("Intensity (a.u.)")

plt.grid()
plt.legend(loc='upper right', fontsize=8)

plt.show()





#%%

'-------Heat maps 400nm-------'  



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Please paste the path name of the excel file :
excel_file = 'Path name of the ... Intensities_400nm_h9_h11.xlsx'

xldata = pd.read_excel(excel_file,
                       sheet_name = "TOD_tot",
                       index_col = 0)

xl_clean = xldata.replace(np.NaN, 0)


print(xl_clean)

heatmap_data = xl_clean
xl_clean = xldata.fillna(0)

plt.figure(figsize=(7, 8),dpi=300) 

ax=sns.heatmap(heatmap_data, cbar_kws={'label': 'Intesity (arb. units)'},
               annot=False, vmin=0, vmax=22)


plt.tight_layout()


xldata = pd.read_excel(excel_file, sheet_name="TOD_tot", index_col=0)
xl_clean = xldata.fillna(0)


plt.xlabel("Harmonic order ", labelpad=30)
plt.ylabel("Pressure [Torr] ", labelpad=50)



plt.show()

