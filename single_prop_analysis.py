import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#import seaborn as sns
#sns.set_style("whitegrid")
#sns.set_palette("colorblind")

_HERE = os.path.dirname(os.path.abspath(__file__))
_V1DATA = os.path.join(_HERE, "UIUC-propDB", "UIUC-propDB", "volume-1", "data")
path = os.path.join(_HERE, "ance_8.5x6_2849cm_4000.txt")
file_name = os.path.basename(path)
stem, _ = os.path.splitext(file_name)
parts = stem.split("_")

brand = parts[0].upper()
diameter_in = float(parts[1].split("x")[0])
diameter = diameter_in * 0.0254 # convert inches to meters
rpm = int(parts[3])

print(f"brand={brand}, diameter_m={diameter}, rpm={rpm}")
J, CT, CP, eta = np.loadtxt(path, skiprows=1, unpack=True)

fig, ax = plt.subplots(figsize=(6.0531*2, 3.74110012361*2))
line_eta = ax.plot(J, eta, marker='^', linewidth=2)
line_ct = ax.plot(J, CT, marker='o', linewidth=2)
line_cp = ax.plot(J, CP, marker='s', linewidth=2)


ct_color = line_ct[0].get_color()
cp_color = line_cp[0].get_color()
eta_color = line_eta[0].get_color()

for i in range(len(J)):
    ax.text(J[i], eta[i], str(i+1), fontsize=14, ha='center', va='bottom')
    ax.text(J[i], CT[i], str(i+1), fontsize=14, ha='center', va='bottom')
    ax.text(J[i], CP[i], str(i+1), fontsize=14, ha='center', va='bottom')
    

ax.set_xlabel(r'Advance Ratio ($J = \frac{V_{\mathrm{fs}}}{\Omega D}$)', fontsize=14)
ax.set_ylabel(r'[-]', fontsize=14)
ax.set_title(brand+f' Propeller  (Diameter: {diameter_in} in, RPM: {rpm})', fontsize=14, fontweight='bold')

from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker='o', color=ct_color, linestyle='-', markersize=8, linewidth=2, label=r'$C_T$'),
    Line2D([0], [0], marker='s', color=cp_color, linestyle='-', markersize=8, linewidth=2, label=r'$C_P$'),
    Line2D([0], [0], marker='^', color=eta_color, linestyle='-', markersize=8, linewidth=2, label=r'$\eta$')
]
ax.legend(handles=handles, fontsize=14)
ax.tick_params(labelsize=14)
try:
    plt.savefig("propeller_performance_original.pdf", bbox_inches="tight")
except PermissionError:
    plt.savefig("propeller_performance_original_backup.svg", bbox_inches="tight")
plt.close(fig)


Thrust = CT *  1.2 * (rpm / 60)**2 * (diameter)**4
print('Thrust: (N)', Thrust)
Vfs = J * (rpm / 60) * diameter 
print('Vfs:', Vfs)

Power = CP * (rpm / 60) ** 3 * (diameter) ** 5 * 1.2
print('Power:', Power)

Area = np.pi * (diameter / 2) ** 2
print('Area: (m^2)', Area)
#Area = diameter**2

# Define the equation to solve: f(Vj) = 0
def equation(Vj, Vfs_val, eta_val):
    epstemp = Vfs_val / Vj
    etatemp = 2/(1 + 1/epstemp)
    return np.divide(2 * Vfs_val * (Vj - Vfs_val), (Vj**2 - Vfs_val**2)) - etatemp

# Solve for Vj for each eta value using iterative convergence
Vj = np.zeros_like(eta)
for i in range(len(eta)):
    # Initial guess from fsolve
    solution = fsolve(equation, x0=Vfs[i], args=(Vfs[i], eta[i]))
    #print(solution)
    print(f"Initial guess for Vj (iteration {i+1}): {solution[0]:.6f} m/s")
    print('epstemp:', Vfs[i] / solution[0])
    Vj[i] = solution[0]
    
    # Iterative convergence loop
    tolerance = 1e-6
    max_iterations = 100
    for iteration in range(max_iterations):
        solution = fsolve(equation, x0=Vj[i], args=(Vfs[i], eta[i]))
        Vj[i] = solution[0]
        Vj_old = Vj[i]
        
        # Calculate V1 from average of Vfs and Vj
        V1_temp = 0.5 * (Vfs[i] + Vj[i])
        
        # Recalculate mdot
        mdot_temp = 1.2 * Area * V1_temp
        print(mdot_temp)
        
        # Recalculate Vj from thrust balance
        Vj[i] = Thrust[i] / mdot_temp + Vfs[i]
        print('epstemp:', Vfs[i] / Vj[i])
        print(f"Iteration {iteration+1}: Vj = {Vj[i]:.6f} m/s")
        # Check convergence
        if abs(Vj[i] - Vj_old) < tolerance:
            break



# Final calculations after convergence
V1 = 0.5 * (Vfs + Vj)
mdot = 1.2 * Area * V1
print('v1:', V1)
print('Vj (converged):', Vj)

print(np.divide(V1, Vj))


Epsilon = Vfs / Vj
Chi = Vfs / V1
Sigma = V1 / Vj
print(Sigma)
Utip = (rpm / 60) * (diameter / 2) 
Phitip = V1 / Utip
Jaero = Vfs / Utip #define aerodynamic advance ratio


sigma = np.linspace(0.00, 2.0, 100)#
chi = np.linspace(0.00, 2.0, 100)
CHI, SIGMA = np.meshgrid(chi, sigma)
EPS = CHI*SIGMA

plt.figure(figsize=(6.0531*2, 3.74110012361*2))
contour = plt.contourf(SIGMA, CHI, EPS, levels=np.linspace(0, 1, 256), cmap='viridis', antialiased=False)

plt.ylabel(r"$\chi = \frac{V_{freestream}}{V_{fan}}$", fontsize=14)
plt.xlabel(r"$\sigma = \frac{V_{fan}}{V_{jet}}$", fontsize=14)
#plt.title(r"Contour of $\epsilon = \chi \cdot \sigma$")
cbar = plt.colorbar(contour, label=r"$\epsilon = \frac{V_{freestream}}{V_{jet}}$")
cbar.set_label(r"$\epsilon = \frac{V_{freestream}}{V_{jet}}$", fontsize=14)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.ax.tick_params(labelsize=14)
mask = EPS > 1
plt.contourf(SIGMA, CHI, mask, levels=[0.5, 1.5], colors='gray',  hatches=['x'])
plt.contour(SIGMA, CHI, EPS, levels=[1.0], colors='black', linewidths=4)
plt.xlim(0, 2)
plt.ylim(0, 2)
contour_lines = plt.contour(SIGMA, CHI, EPS, levels=np.linspace(0.1, 0.9, 5), alpha = 0.8, colors='white', linewidths=1)
plt.clabel(contour_lines, inline=True, fontsize=14, fmt='%.1f')

#plt.savefig("chi_sigma_eps_contour.svg", format="svg", bbox_inches='tight', dpi=100)
#plt.scatter(Sigma, Chi, marker='x', color='red', s=100, zorder=10)

# Plot the line sigma = 0.5*(chi + 1)
sigma_line = np.linspace(0.5, 1, 100)
chi_line = 2 - 1/sigma_line
plt.plot(sigma_line, chi_line, '--', linewidth=3, color='white',  label=r'$\chi = 2 - 1/\sigma$')

plt.contour(SIGMA, CHI,EPS, levels=[1.0], colors='black', linewidths=3)


plt.scatter(Sigma, Chi, marker='x', color='red', s=100, zorder=10, label='Experiment')
for i in range(len(Chi)):
    plt.text(Sigma[i], Chi[i], str(i+1), fontsize=14, ha='center', va='bottom')
#plt.xlabel(r'$\sigma = V_1/V_j$', fontsize=12)
#plt.ylabel(r'$\chi = V_{\mathrm{fs}}/V_1$', fontsize=12)
plt.title(brand+f' Propeller (Diameter: {diameter_in} in, RPM: {rpm})', fontsize=14, fontweight='bold')
plt.legend(fontsize=14, loc='lower left')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
try:
    plt.savefig("chi_sigma_eps_contour.png", bbox_inches="tight", dpi=300)
    print("Saved chi_sigma_eps_contour.pdf successfully")
except PermissionError:
    print("Warning: Could not save PDF (file may be open). Saving as SVG instead.")
    plt.savefig("chi_sigma_eps_contour_backup.svg", bbox_inches="tight")





#cbar.set_clim(0, 1)
 
# Add gray region where EPS > 1



Jrange = np.linspace(0.01, 2, 100)
Phirange = np.linspace(0.01, 2, 100)
J_mesh, Phi_mesh = np.meshgrid(Jrange, Phirange)
Chirange = np.divide(J_mesh, Phi_mesh)
# Requested meshgrids for psi and phi in [0.1, 1.0]

#Chirange = np.minimum(Chirange, 1)

fig4, ax4 = plt.subplots(figsize=(6.0531*2, 3.74110012361*2))
contourf = ax4.contourf(J_mesh, Phi_mesh, np.minimum(Chirange, 3), levels=50, cmap='viridis', vmin=0, vmax=3.1, extend='max')
cbar = fig4.colorbar(contourf, ax=ax4, label=r'$\chi$', extend='max')
cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
ax4.plot(Jaero, Phitip, marker='x', color='red', linestyle='-', markersize=10, markeredgewidth=2)
for i in range(len(Jaero)):
    ax4.text(Jaero[i], Phitip[i], str(i+1), fontsize=10, ha='center', va='bottom')
ax4.set_xlabel(r'$J_{\mathrm{aero}} = V_{\mathrm{fs}}/U_{\mathrm{tip}}$', fontsize=12)
ax4.set_ylabel(r'$\Phi_{\mathrm{tip}} = V_1/U_{\mathrm{tip}}$', fontsize=12)
ax4.set_title(brand+f' Propeller  (Diameter: {diameter_in} in, RPM: {rpm})', fontsize=12, fontweight='bold')
try:
    fig4.savefig("jaero_phitip_contour.pdf", bbox_inches="tight", dpi=300)
except PermissionError:
    fig4.savefig("jaero_phitip_contour_backup.svg", bbox_inches="tight")




print('Jaero:', Jaero)
print('Phitip:', Phitip)
print('Chi:', Chi)
print('chicalc:', Jaero / Phitip)

fig3, ax3 = plt.subplots(figsize=(6.0531*2, 3.74110012361*2))
line_epsilon = ax3.plot(J, Epsilon, marker='^', linewidth=2)
line_chi = ax3.plot(J, Chi, marker='o', linewidth=2)
line_sigma = ax3.plot(J, Sigma, marker='s', linewidth=2)
#line_v1_calc = ax3.plot(J, V1 / (0.5*(Vfs + Vj)), marker='d', linewidth=2)

epsilon_color = line_epsilon[0].get_color()
chi_color = line_chi[0].get_color()
sigma_color = line_sigma[0].get_color()
#v1_calc_color = line_v1_calc[0].get_color()


for i in range(len(J)):
    ax3.text(J[i], Epsilon[i], str(i+1), fontsize=8, ha='center', va='bottom')
    ax3.text(J[i], Chi[i], str(i+1), fontsize=8, ha='center', va='bottom')
    ax3.text(J[i], Sigma[i], str(i+1), fontsize=8, ha='center', va='bottom')
    #ax3.text(J[i], V1[i] / (0.5*(Vfs[i] + Vj[i])), str(i+1), fontsize=8, ha='center', va='bottom')
    

ax3.set_xlabel(r'Advance Ratio ($J = \frac{V_{\mathrm{fs}}}{\Omega D}$)', fontsize=12)
ax3.set_ylabel(r'[-]', fontsize=12)
ax3.set_title(brand+f' Propeller (Diameter: {diameter_in} in, RPM: {rpm})', fontsize=12, fontweight='bold')

from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker='o', color=chi_color, linestyle='-', markersize=8, linewidth=2, label=r'$\chi = V_{\mathrm{fs}}/V_1$'),
    Line2D([0], [0], marker='s', color=sigma_color, linestyle='-', markersize=8, linewidth=2, label=r'$\sigma = V_1/V_j$'),
    Line2D([0], [0], marker='^', color=epsilon_color, linestyle='-', markersize=8, linewidth=2, label=r'$\varepsilon = V_{\mathrm{fs}}/V_j$'),
    #Line2D([0], [0], marker='d', color=v1_calc_color, linestyle='-', markersize=8, linewidth=2, label=r'$V_1 / (0.5(V_{\mathrm{fs}} + V_j))$')
]
ax3.legend(handles=handles)
ax3.set_ylim(0, 1.5)
try:
    fig3.savefig("eps_performance_original.pdf", bbox_inches="tight")
except PermissionError:
    fig3.savefig("eps_performance_original_backup.svg", bbox_inches="tight")
plt.close(fig3)




Phi = J / Chi
print('Chi:', Chi)
print('Sigma:', Sigma)
print('Chi * Sigma:', Chi * Sigma)
print(Vfs / Vj)



FanEfficiency = mdot/2 * (Vj**2 - Vfs**2) / Power
print('Fan Efficiency:', FanEfficiency)
ETAPROP = eta / FanEfficiency




CTHRUST = Thrust / (mdot * Vj)
#CPOWER = Power / ( mdot * Vj**2)
CPOWER = mdot/2 * (Vj**2 - Vfs**2) / ( mdot * Vj**2)
etatheory = 2/(1 + 1/np.linspace(0.01, 1, len(eta)))
CTtheory = 1 - np.linspace(0.01, 1, len(eta))
CPtheory = 0.5*(1 - np.linspace(0.01, 1, len(eta))**2)


#etatheory = 2/(1 + np.linspace(1, 10, len(eta)))
#CTtheory =  np.linspace(1, 10, len(eta)) - 1
#CPtheory = 0.5*( np.linspace(1, 10, len(eta))**2 - 1)


fig2, ax2 = plt.subplots(figsize=(6.0531*2, 3.74110012361*2))

theory_x = np.linspace(0, 1, len(eta))
line_eta = ax2.plot(theory_x, etatheory, linestyle="--", linewidth=2)
#line_etafan = ax2.plot(theory_x, EtaFan, linestyle="--", linewidth=2)
line_ct = ax2.plot(theory_x, CTtheory, linestyle="--", linewidth=2)
line_cp = ax2.plot(theory_x, CPtheory, linestyle="--", linewidth=2)
line_faneta = ax2.plot(np.linspace(0, 1, len(eta)),np.linspace(5, 10, len(eta)), linewidth=2, zorder=10)

eta_color = line_eta[0].get_color()
ct_color = line_ct[0].get_color()
cp_color = line_cp[0].get_color()
faneta_color = line_faneta[0].get_color() 

ax2.scatter(Epsilon, FanEfficiency, marker="s", s=60, zorder=10, color = faneta_color, label=r'$\eta_{\mathrm{fan}}$')
ax2.scatter(Epsilon, CTHRUST, marker="s", s=60, zorder=10, color=ct_color)
ax2.scatter(Epsilon, CPOWER, marker="s", s=60, zorder=10, color=cp_color, )
#ax2.scatter(Epsilon, ETAPROP, marker="s", s=60, zorder=10, color=eta_color)
ax2.scatter(Epsilon, eta / FanEfficiency, marker="s", s=60, zorder=10, color=eta_color)


for i in range(len(Epsilon)):
    ax2.text(Epsilon[i], CTHRUST[i], str(i+1), fontsize=14, ha='center', va='bottom')
    ax2.text(Epsilon[i], CPOWER[i], str(i+1), fontsize=14, ha='center', va='bottom')
    ax2.text(Epsilon[i], ETAPROP[i], str(i+1), fontsize=14, ha='center', va='bottom')
    ax2.text(Epsilon[i], FanEfficiency[i], str(i+1), fontsize=14, ha='center', va='bottom')
ax2.set_xlabel(r'$ \varepsilon$ ($V_{\mathrm{fs}}/V_j$)', fontsize=14)
ax2.set_ylabel(r"[-]", fontsize=14)
ax2.set_title(brand+f' Propeller (Diameter: {diameter_in} in, RPM: {rpm})', fontsize=14, fontweight='bold')

ax2.set_ylim(0, 1.05)
ax2.tick_params(labelsize=14)

from matplotlib.lines import Line2D
# Create two distinct legend columns: Theory (left) and Experiment (right)
theory_handles = [
    Line2D([0], [0], color=eta_color, linestyle="--", linewidth=2),
    Line2D([0], [0], color=ct_color, linestyle="--", linewidth=2),
    Line2D([0], [0], color=cp_color, linestyle="--", linewidth=2),
]
theory_labels = [
    r'$\eta_{\mathrm{propulsive}}$',
    r'$C_T$',
    r'$C_P$',
]

experiment_handles = [
    Line2D([0], [0], marker="s", color=eta_color, linestyle="None", markersize=10),
    Line2D([0], [0], marker="s", color=ct_color, linestyle="None", markersize=10),
    Line2D([0], [0], marker="s", color=cp_color, linestyle="None", markersize=10),
    Line2D([0], [0], marker="s", color=faneta_color, linestyle="None", markersize=10),
]
experiment_labels = [
    r'$\eta_{\mathrm{propulsive}}$',
    r'$C_T$',
    r'$C_P$',
    r'$\eta_{\mathrm{fan}}$',
]

legend_theory = ax2.legend(
    theory_handles,
    theory_labels,
    title='Theory',
    fontsize=12,
    title_fontsize=13,
    loc='lower left',
    bbox_to_anchor=(0.02, 0.02),
    framealpha=1.0,
    facecolor='white',
)
legend_experiment = ax2.legend(
    experiment_handles,
    experiment_labels,
    title='Experiment',
    fontsize=12,
    title_fontsize=13,
    loc='lower left',
    bbox_to_anchor=(0.28, 0.02),
    framealpha=1.0,
    facecolor='white',
)
ax2.add_artist(legend_theory)
try:
    fig2.savefig("single_propeller_performance.pdf", bbox_inches="tight")
except PermissionError:
    fig2.savefig("single_propeller_performance_backup.svg", bbox_inches="tight")






#Phitip = V1 / Utip
#Psitip = 0.5 * (Vj**2 - Vfs**2) / (Utip**2)
plt.figure(figsize=(6.0531*2, 3.74110012361*2))
#plt.plot(Phitip, Psitip, linewidth=2, alpha=0.2, zorder=5, color='black')
#plt.scatter(Phitip, Psitip, c=Epsilon, s=20, cmap='viridis', vmin=0, vmax=1, edgecolors='black', linewidths=0.5, zorder=5)

Thrust = CT *  1.2 * (rpm / 60)**2 * (diameter)**4
print('Thrust: (N)', Thrust)
Vfs = J * (rpm / 60) * diameter 
print('Vfs:', Vfs)
Power = CP * (rpm / 60) ** 3 * (diameter) ** 5 * 1.2
print('Power:', Power)
Area = np.pi * (diameter / 2) ** 2
print('Area: (m^2)', Area)
#solving quadratic
a = 1
b = -Vfs
c = -Thrust / (1.2 * Area * 0.5)
Vj = (Vfs + np.sqrt(b**2 - 4*a*c)) /(2 * a)
Vfan = 0.5 * (Vfs + Vj)
Utip = (rpm / 60) * (diameter / 2)
print('Utip:', Utip)
Phi = Vfan / Utip
print('Phi:', Phi)
Psi = 0.5 * (Vj**2 - Vfs**2) / (Utip**2)
print('Psi:', Psi)
eps = Vfs / Vj
print('Epsilon:', eps)
plt.plot(Phi, Psi,  linewidth=2, zorder=5, color = 'black')
#plt.scatter(Phi_static[idx], Psi_static[idx], )
plt.scatter(Phi, Psi, marker='s', c = FanEfficiency, s=100, zorder=10, edgecolors='black', linewidths=1.5, vmin=np.min(FanEfficiency), vmax=np.max(FanEfficiency), cmap='viridis')
#plt.text(0.5, 0.95, f'RPM: {rpm}', transform=plt.gca().transAxes, fontsize=14, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

cbar = plt.colorbar(label=r'$\eta_{\mathrm{fan}}$')
cbar.set_label(r'$\eta_{\mathrm{fan}}$', fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.xlabel(r'$\Phi_{\mathrm{tip}} $', fontsize=14)
plt.ylabel(r'$\Psi_{\mathrm{tip}}$', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0.5, 1.6)
plt.ylim(-0.02, 0.7)
plt.grid(True, alpha=0.3)
plt.title(brand+f' Propeller (Diameter: {diameter_in} in, RPM: {rpm})', fontsize=14, fontweight='bold')

psi_range = np.linspace(-0.02, 0.7, 120)
phi_range = np.linspace(0.001, 1.6, 120)
phi_mesh, psi_mesh = np.meshgrid(phi_range, psi_range)

epsilon_mesh = 2 / (psi_mesh / (2 * phi_mesh**2) + 1) - 1
epsilon_mesh = (2 * phi_mesh**2 - psi_mesh) / (2 * phi_mesh**2 + psi_mesh)
#(2*phi_mesh**2 / psi_mesh) + 1 )**(-1) - 1
print(epsilon_mesh)
contour_levels = [0.0, 0.3, 0.5, 0.7, 0.9,  1.0]
contour_lines = plt.contour(
    phi_mesh,
    psi_mesh,
    epsilon_mesh,
    levels=contour_levels,
    alpha=1.0,
    colors='black',
    linewidths=1,
    zorder=1,
)

def fmt_epsilon(level):
    return rf'$\varepsilon = {level:.1f}$'





#5004
r'''path = r"C:\Users\amr200\OneDrive - University of Cambridge\Desktop\UIUC PROPS\UIUC-propDB\UIUC-propDB\volume-1\data\ance_8.5x6_2850cm_5004.txt"
file_name = os.path.basename(path)
stem, _ = os.path.splitext(file_name)
parts = stem.split("_")
brand = parts[0]
diameter_in = float(parts[1].split("x")[0])
diameter = diameter_in * 0.0254 # convert inches to meters
rpm = int(parts[3])
print(f"brand={brand}, diameter_m={diameter}, rpm={rpm}")
J, CT, CP, eta = np.loadtxt(path, skiprows=1, unpack=True)
Thrust = CT *  1.2 * (rpm / 60)**2 * (diameter)**4
print('Thrust: (N)', Thrust)
Vfs = J * (rpm / 60) * diameter 
print('Vfs:', Vfs)
Power = CP * (rpm / 60) ** 3 * (diameter) ** 5 * 1.2
print('Power:', Power)
Area = np.pi * (diameter / 2) ** 2
print('Area: (m^2)', Area)
#solving quadratic
a = 1
b = -Vfs
c = -Thrust / (1.2 * Area * 0.5)
Vj = (Vfs + np.sqrt(b**2 - 4*a*c)) /(2 * a)
Vfan = 0.5 * (Vfs + Vj)
Utip = (rpm / 60) * (diameter / 2)
print('Utip:', Utip)
Phi = Vfan / Utip
print('Phi:', Phi)
Psi = 0.5 * (Vj**2 - Vfs**2) / (Utip**2)
print('Psi:', Psi)
eps = Vfs / Vj
print('Epsilon:', eps)
plt.plot(Phi, Psi, linewidth=2, alpha=0.8, zorder=5, label=f'RPM: {rpm}')
plt.scatter(Phi, Psi, c=eps, s=45, cmap='viridis', vmin=0, vmax=1, edgecolors='black', linewidths=0.5, zorder=5)

#5986
path = os.path.join(_V1DATA, "ance_8.5x6_2851cm_5986.txt")
file_name = os.path.basename(path)
stem, _ = os.path.splitext(file_name)
parts = stem.split("_")
brand = parts[0]
diameter_in = float(parts[1].split("x")[0])
diameter = diameter_in * 0.0254 # convert inches to meters
rpm = int(parts[3])
print(f"brand={brand}, diameter_m={diameter}, rpm={rpm}")
J, CT, CP, eta = np.loadtxt(path, skiprows=1, unpack=True)
Thrust = CT *  1.2 * (rpm / 60)**2 * (diameter)**4
print('Thrust: (N)', Thrust)
Vfs = J * (rpm / 60) * diameter 
print('Vfs:', Vfs)
Power = CP * (rpm / 60) ** 3 * (diameter) ** 5 * 1.2
print('Power:', Power)
Area = np.pi * (diameter / 2) ** 2
print('Area: (m^2)', Area)
#solving quadratic
a = 1
b = -Vfs
c = -Thrust / (1.2 * Area * 0.5)
Vj = (Vfs + np.sqrt(b**2 - 4*a*c)) /(2 * a)
Vfan = 0.5 * (Vfs + Vj)
Utip = (rpm / 60) * (diameter / 2)
print('Utip:', Utip)
Phi = Vfan / Utip
print('Phi:', Phi)
Psi = 0.5 * (Vj**2 - Vfs**2) / (Utip**2)
print('Psi:', Psi)
eps = Vfs / Vj
print('Epsilon:', eps)
plt.plot(Phi, Psi, linewidth=2, alpha=0.8, zorder=5, label=f'RPM: {rpm}')
plt.scatter(Phi, Psi, c=eps, s=45, cmap='viridis', vmin=0, vmax=1, edgecolors='black', linewidths=0.5, zorder=5)

#6009
path = os.path.join(_V1DATA, "ance_8.5x6_2852cm_6009.txt")
file_name = os.path.basename(path)
stem, _ = os.path.splitext(file_name)
parts = stem.split("_")
brand = parts[0]
diameter_in = float(parts[1].split("x")[0])
diameter = diameter_in * 0.0254 # convert inches to meters
rpm = int(parts[3])
print(f"brand={brand}, diameter_m={diameter}, rpm={rpm}")
J, CT, CP, eta = np.loadtxt(path, skiprows=1, unpack=True)
Thrust = CT *  1.2 * (rpm / 60)**2 * (diameter)**4
print('Thrust: (N)', Thrust)
Vfs = J * (rpm / 60) * diameter 
print('Vfs:', Vfs)
Power = CP * (rpm / 60) ** 3 * (diameter) ** 5 * 1.2
print('Power:', Power)
Area = np.pi * (diameter / 2) ** 2
print('Area: (m^2)', Area)
#solving quadratic
a = 1
b = -Vfs
c = -Thrust / (1.2 * Area * 0.5)
Vj = (Vfs + np.sqrt(b**2 - 4*a*c)) /(2 * a) 
Vfan = 0.5 * (Vfs + Vj)
Utip = (rpm / 60) * (diameter / 2)
print('Utip:', Utip)
Phi = Vfan / Utip
print('Phi:', Phi)
Psi = 0.5 * (Vj**2 - Vfs**2) / (Utip**2)
print('Psi:', Psi)
eps = Vfs / Vj
print('Epsilon:', eps)
plt.plot(Phi, Psi, linewidth=2, alpha=0.8, zorder=5, label=f'RPM: {rpm}')
plt.scatter(Phi, Psi, c=eps, s=45, cmap='viridis', vmin=0, vmax=1, edgecolors='black', linewidths=0.5, zorder=5)

#6928
path = os.path.join(_V1DATA, "ance_8.5x6_2853cm_6928.txt")
file_name = os.path.basename(path)
stem, _ = os.path.splitext(file_name)
parts = stem.split("_")
brand = parts[0]
diameter_in = float(parts[1].split("x")[0])
diameter = diameter_in * 0.0254 # convert inches to meters
rpm = int(parts[3])
print(f"brand={brand}, diameter_m={diameter}, rpm={rpm}")
J, CT, CP, eta = np.loadtxt(path, skiprows=1, unpack=True)
Thrust = CT *  1.2 * (rpm / 60)**2 * (diameter)**4
print('Thrust: (N)', Thrust)
Vfs = J * (rpm / 60) * diameter 
print('Vfs:', Vfs)
Power = CP * (rpm / 60) ** 3 * (diameter) ** 5 * 1.2
print('Power:', Power)
Area = np.pi * (diameter / 2) ** 2
print('Area: (m^2)', Area)
#solving quadratic
a = 1
b = -Vfs
c = -Thrust / (1.2 * Area * 0.5)
Vj = (Vfs + np.sqrt(b**2 - 4*a*c)) /(2 * a)
Vfan = 0.5 * (Vfs + Vj)
Utip = (rpm / 60) * (diameter / 2)
print('Utip:', Utip)
Phi = Vfan / Utip
print('Phi:', Phi)
Psi = 0.5 * (Vj**2 - Vfs**2) / (Utip**2)
print('Psi:', Psi)
eps = Vfs / Vj
print('Epsilon:', eps)
plt.plot(Phi, Psi, linewidth=2, alpha=0.8, zorder=5, label=f'RPM: {rpm}')
plt.scatter(Phi, Psi, c=eps, s=45, cmap='viridis', vmin=0, vmax=1, edgecolors='black', linewidths=0.5, zorder=5)

#6914
path = os.path.join(_V1DATA, "ance_8.5x6_2854cm_6914.txt")
file_name = os.path.basename(path)
stem, _ = os.path.splitext(file_name)
parts = stem.split("_")
brand = parts[0]
diameter_in = float(parts[1].split("x")[0])
diameter = diameter_in * 0.0254 # convert inches to meters
rpm = int(parts[3])
print(f"brand={brand}, diameter_m={diameter}, rpm={rpm}")
J, CT, CP, eta = np.loadtxt(path, skiprows=1, unpack=True)
Thrust = CT *  1.2 * (rpm / 60)**2 * (diameter)**4
print('Thrust: (N)', Thrust)
Vfs = J * (rpm / 60) * diameter 
print('Vfs:', Vfs)
Power = CP * (rpm / 60) ** 3 * (diameter) ** 5 * 1.2
print('Power:', Power)
Area = np.pi * (diameter / 2) ** 2
print('Area: (m^2)', Area)
#solving quadratic
a = 1
b = -Vfs
c = -Thrust / (1.2 * Area * 0.5)
Vj = (Vfs + np.sqrt(b**2 - 4*a*c)) /(2 * a)
Vfan = 0.5 * (Vfs + Vj)
Utip = (rpm / 60) * (diameter / 2)
print('Utip:', Utip)
Phi = Vfan / Utip
print('Phi:', Phi)
Psi = 0.5 * (Vj**2 - Vfs**2) / (Utip**2)
print('Psi:', Psi)
eps = Vfs / Vj
print('Epsilon:', eps)
plt.plot(Phi, Psi, linewidth=2, alpha=0.8, zorder=5, label=f'RPM: {rpm}')
plt.scatter(Phi, Psi, c=eps, s=45, cmap='viridis', vmin=0, vmax=1, edgecolors='black', linewidths=0.5, zorder=5)'''



#Load in static case for the same propeller
path = os.path.join(_V1DATA, "ance_8.5x6_static_2848cm.txt")
file_name = os.path.basename(path)
stem, _ = os.path.splitext(file_name)
parts = stem.split("_")

brand = parts[0].upper()
diameter_in = float(parts[1].split("x")[0])
diameter = diameter_in * 0.0254 # convert inches to meters
#rpm = int(parts[3])

print(f"brand={brand}, diameter_m={diameter}")


RPM, CT, CP = np.loadtxt(path, skiprows=1, unpack=True)

print(RPM)
idx = np.argmin(np.abs(RPM - 4000))
print("idx:", idx, "RPM[idx]:", RPM[idx])

Thrust = CT *  1.2 * (RPM / 60)**2 * (diameter)**4
Power = CP * (RPM / 60) ** 3 * (diameter) ** 5 * 1.2
Area = np.pi * (diameter / 2) ** 2


Vj_static = np.sqrt(Thrust / (1.2 * Area * 0.5) ) 
print('Vj_static:', Vj_static)

Vfan_static = 0.5 * Vj_static
print('Vfan_static:', Vfan_static)

mdot = 1.2*Area*Vfan_static

Utip_static = (RPM / 60) * (diameter / 2)
print('Utip_static:', Utip_static)

Phi_static = Vfan_static / Utip_static
print('Phi_static:', Phi_static)

Psi_static = 0.5 * (Vj_static**2 - 0**2) / (Utip_static**2)
print('Psi_static:', Psi_static)
eps_static = 0 / Vj_static

FanEfficiency_static = mdot/2 * (Vj_static**2 - 0**2) / Power
print('Fan Efficiency:', FanEfficiency_static[idx])

plt.scatter(Phi_static[idx], Psi_static[idx], marker='s', c=FanEfficiency_static[idx], s=100, zorder=10, edgecolors='black', linewidths=1.5, vmin=np.min(FanEfficiency), vmax=np.max(FanEfficiency), cmap='viridis')


# Deterministic label positions (points lie exactly on each epsilon contour,
# chosen clear of the operating-point annotations)
_label_pos = [(0.55, 0.605), (0.62, 0.414), (0.85, 0.482), (1.05, 0.389), (1.05, 0.116), (1.00, 0.0)]
plt.clabel(contour_lines, inline=True, manual=_label_pos, fontsize=14, fmt=fmt_epsilon)

# Operating-point annotations (thesis Prop_Characteristic2)
i_to = int(np.argmin(np.abs(Phi - 1.12)))
i_cr = int(np.argmin(np.abs(Phi - 1.29)))
_arrow = dict(arrowstyle='-|>', color='black', lw=2.5)
_labelbox = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none')
plt.annotate('Take-off', xy=(Phi[i_to], Psi[i_to] + 0.015), xytext=(Phi[i_to], Psi[i_to] + 0.10),
             arrowprops=_arrow, fontsize=16, ha='center', bbox=_labelbox, zorder=12)
plt.annotate('Cruise', xy=(Phi[i_cr], Psi[i_cr] + 0.015), xytext=(Phi[i_cr], Psi[i_cr] + 0.10),
             arrowprops=_arrow, fontsize=16, ha='center', bbox=_labelbox, zorder=12)
plt.annotate('Static', xy=(Phi_static[idx] + 0.025, Psi_static[idx]), xytext=(Phi_static[idx] + 0.15, Psi_static[idx]),
             arrowprops=_arrow, fontsize=16, va='center', ha='left', bbox=_labelbox, zorder=12)

plt.savefig("single_fan_efficiency_contour_scatter.png", bbox_inches="tight", dpi=300)
plt.savefig("Prop_Characteristic2.pdf", bbox_inches="tight")
plt.savefig("Prop_Characteristic2.svg", bbox_inches="tight")
_thesis_figs = os.path.join(_HERE, "..", "Reaves-Thesis", "4. CURTIS and Throughflow", "Figs")
if os.path.isdir(_thesis_figs):
    plt.savefig(os.path.join(_thesis_figs, "Prop_Characteristic2.pdf"), bbox_inches="tight")
    print("exported Prop_Characteristic2.pdf to thesis Figs")



# Add the static operating point to the efficiency figure
ax2.scatter(eps_static[idx], FanEfficiency_static[idx], marker="s", s=60, zorder=10, color=faneta_color)
#ax2.text(eps_static[idx], FanEfficiency_static[idx], 'static', fontsize=12, ha='left', va='bottom')




try:
    fig2.savefig("single_propeller_performance.pdf", bbox_inches="tight")
except PermissionError:
    fig2.savefig("single_propeller_performance_backup.svg", bbox_inches="tight")




plt.show()


