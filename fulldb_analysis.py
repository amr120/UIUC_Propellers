"""Whole-database reduction of the UIUC propeller data into the framework.

Reads all four volumes of the UIUC propeller database and reduces every
measured operating point into the non-dimensional variables of Chapter 3
(eps, chi, sigma, phi_tip, psi_tip, and the fan efficiency), then draws the
population-level figures.

Thesis figures produced (all Chapter 3):
  chi_sigma_eps_contour_fullDB.png   Fig. 3.8  (fig:chi_sigma_eps_contour)
  fan_efficiency_contour_scatter.png Fig. 3.9  (fig:fullDBSmithChart)
  propeller_plots.pdf                Fig. 3.10 (fig:fullDBfanefficiencyeps)

Also written but not used by the thesis: fan_efficiency_contour.png, and the
_backup.svg copies alongside each figure.

The single-propeller counterpart, which draws one machine's working line on
these same axes, is single_prop_analysis.py.
"""

import os
import re
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import fsolve
#import seaborn as sns

#sns.set_style("whitegrid")
#sns.set_palette("colorblind")


_HERE = os.path.dirname(os.path.abspath(__file__))
_THESIS_FIGS = os.path.join(_HERE, "..", "Reaves-Thesis", "3. 1D Models", "Figs")


def _save(fig, name):
    """Write a figure beside the script and into the thesis Figs directory.

    Vector PDF throughout: heavy layers are rasterized individually at the
    call site, so the field art stays compact while axes, labels and legends
    remain real text.
    """
    fig.savefig(os.path.join(_HERE, name), bbox_inches="tight", dpi=300)
    if os.path.isdir(_THESIS_FIGS):
        fig.savefig(os.path.join(_THESIS_FIGS, name), bbox_inches="tight",
                    dpi=300)
        print(f"exported {name} to thesis Figs")
    else:
        print(f"saved {name}")


def _db_root():
    """Locate the extracted database, tolerating either directory layout.

    The UIUC archive unpacks to a doubled UIUC-propDB/UIUC-propDB/ path; that
    redundant level may be flattened. Accept whichever is present.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    nested = os.path.join(here, "UIUC-propDB", "UIUC-propDB")
    flat = os.path.join(here, "UIUC-propDB")
    if os.path.isdir(os.path.join(nested, "volume-1", "data")):
        return nested
    if os.path.isdir(os.path.join(flat, "volume-1", "data")):
        return flat
    raise FileNotFoundError(
        "UIUC database not found; expected volume-1/data under "
        f"{nested} or {flat}. Extract UIUC-propDB.zip alongside this script."
    )


_DB_ROOT = _db_root()
DATA_DIRS = [os.path.join(_DB_ROOT, f"volume-{i}", "data") for i in range(1, 5)]
RHO = 1.2
PLOT = True


def parse_filename(file_name):
    stem, _ = os.path.splitext(file_name)
    parts = stem.split("_")
    if len(parts) < 4:
        return None

    brand = parts[0]
    diameter_match = re.search(r"\d+(?:\.\d+)?", parts[1])
    if not diameter_match:
        return None
    diameter_in = float(diameter_match.group(0))
    rpm_match = re.search(r"\d+", parts[3])
    if not rpm_match:
        return None
    rpm = int(rpm_match.group(0))
    return brand, diameter_in, rpm


def equation(Vj, Vfs_val, eta_val):
    return (2 * Vfs_val * (Vj - Vfs_val)) / (Vj**2 - Vfs_val**2) - eta_val


def solve_vj(Vfs, eta):
    Vj = np.full_like(eta, np.nan, dtype=float)
    for i in range(len(eta)):
        if eta[i] <= 0 or Vfs[i] == 0:
            continue
        solution = fsolve(equation, x0=2 * Vfs[i], args=(Vfs[i], eta[i]))
        Vj[i] = solution[0]
    return Vj


def load_columns(path):
    data_start = None
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                _ = [float(value) for value in parts[:4]]
            except ValueError:
                continue
            data_start = idx
            break

    if data_start is None:
        raise ValueError(f"No numeric data rows found in {path}")

    J, CT, CP, eta = np.loadtxt(path, skiprows=data_start, unpack=True)
    return J, CT, CP, eta


def process_file(path):
    file_name = os.path.basename(path)
    parsed = parse_filename(file_name)
    if parsed is None:
        print(f"Skipping (unrecognized name): {file_name}")
        return None

    brand, diameter_in, rpm = parsed
    diameter_m = diameter_in * 0.0254
    n = rpm / 60

    J, CT, CP, eta = load_columns(path)
    Vfs = J * n * diameter_m
    Thrust = CT * RHO * n**2 * diameter_m**4
    Power = CP * RHO * n**3 * diameter_m**5
    Area = np.pi * (diameter_m / 2) ** 2
    
    # Solve for Vj for each eta value using iterative convergence
    Vj = np.zeros_like(eta)
    for i in range(len(eta)):
        # Initial guess from fsolve
        solution = fsolve(equation, x0=1.5*Vfs[i], args=(Vfs[i], eta[i]))
        Vj[i] = solution[0]
        
        # Iterative convergence loop
        tolerance = 1e-6
        max_iterations = 100
        for iteration in range(max_iterations):
            Vj_old = Vj[i]
            
            # Calculate V1 from average of Vfs and Vj
            V1_temp = 0.5 * (Vfs[i] + Vj[i])
            
            # Recalculate mdot
            mdot_temp = RHO * Area * V1_temp
            
            # Recalculate Vj from thrust balance
            Vj[i] = Thrust[i] / mdot_temp + Vfs[i]
            
            # Check convergence
            if abs(Vj[i] - Vj_old) < tolerance:
                break
    
    # Final calculations after convergence
    V1 = 0.5 * (Vfs + Vj)
    mdot = RHO * Area * V1

    Epsilon = Vfs / Vj
    Chi = Vfs / V1
    Sigma = V1 / Vj
    Utip = 2 * np.pi * n * (diameter_m / 2)  # Omega * r_c (2pi: rev/s -> rad/s)
    Phitip = V1 / Utip
    Psitip = 0.5 * (Vj**2 - Vfs**2) / (Utip**2)
    Jaero = Vfs / Utip
    CTHRUST = Thrust / (mdot * Vj)
    CPOWER = Power / (mdot * Vj**2)
    CPOWER = mdot/2 * (Vj**2 - Vfs**2) / (mdot * Vj**2)
    FanEfficiency = mdot/2 * (Vj**2 - Vfs**2) / Power
    ETAPROP = eta / FanEfficiency
    etatheory = 2 / (1 + 1 / np.linspace(0, 1, len(eta)))
    CTtheory = 1 - np.linspace(0, 1, len(eta))
    CPtheory = 0.5 * (1 - np.linspace(0, 1, len(eta)) ** 2)

    return {
        "file": file_name,
        "brand": brand,
        "diameter_in": diameter_in,
        "diameter_m": diameter_m,
        "rpm": rpm,
        "J": J,
        "CT": CT,
        "CP": CP,
        "eta": eta,
        "Vfs": Vfs,
        "Vj": Vj,
        "V1": V1,
        "Thrust": Thrust,
        "Power": Power,
        "Area": Area,
        "mdot": mdot,
        "Epsilon": Epsilon,
        "Chi": Chi,
        "Sigma": Sigma,
        "Utip": Utip,
        "Phitip": Phitip,
        "Psitip": Psitip,
        "Jaero": Jaero,
        "CTHRUST": CTHRUST,
        "CPOWER": CPOWER,
        "FanEfficiency": FanEfficiency,
        "ETAPROP": ETAPROP,
        "etatheory": etatheory,
        "CTtheory": CTtheory,
        "CPtheory": CPtheory,
    }


def main():
    txt_files = []
    for DATA_DIR in DATA_DIRS:
        txt_files.extend([
            os.path.join(DATA_DIR, name)
            for name in os.listdir(DATA_DIR)
            if name.lower().endswith(".txt")
            and "_static_" not in name.lower()
            and "_geom" not in name.lower()
        ])

    results = []
    for path in txt_files:
        result = process_file(path)
        if result is None:
            continue
        results.append(result)
        print(
            f"{result['file']}: brand={result['brand']}, "
            f"diameter_in={result['diameter_in']}, rpm={result['rpm']}"
        )

    print(f"Processed {len(results)} files.")

    if PLOT and results:
        import matplotlib.pyplot as plt
        # serif body text and Computer Modern math, matching the thesis; the
        # per-call fontsize arguments below are left as they are
        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

        fig, ax = plt.subplots(figsize=(6.0531*2, 3.74110012361*2))

        theory_x = np.linspace(0, 1, len(results[0]["eta"]))
        line_eta = ax.plot(theory_x, results[0]["etatheory"],  linestyle="--", linewidth=2)
        line_ct = ax.plot(theory_x, results[0]["CTtheory"],  linestyle="--", linewidth=2)
        line_cp = ax.plot(theory_x, results[0]["CPtheory"],  linestyle="--", linewidth=2)
        line_faneta = ax.plot(np.linspace(0, 1, len(results[0]["FanEfficiency"])),np.linspace(5, 10, len(results[0]["FanEfficiency"])), linewidth=2, zorder=10)

        eta_color = line_eta[0].get_color()
        ct_color = line_ct[0].get_color()
        cp_color = line_cp[0].get_color()
        faneta_color = line_faneta[0].get_color() 

        for i, result in enumerate(results):
        #    label_prefix = f"{result['brand']} {result['diameter_in']}in {result['rpm']}rpm"
        #    #ax.scatter(result["Epsilon"], result["CTHRUST"],  marker="s", s=20, zorder=10, facecolors='none', edgecolors=ct_color, label=label_exp)
        #    ax.scatter(result["Epsilon"], result["CTHRUST"],  marker="s", s=20, zorder=10,  color=ct_color)
        #    ax.scatter(result["Epsilon"], result["CPOWER"],  marker="s", s=20, zorder=10,  color=cp_color)
        #    ax.scatter(result["Epsilon"], result["ETAPROP"], marker="s", s=20, zorder=10, color=eta_color)
             ax.plot(result["Epsilon"], result["FanEfficiency"], marker="s",  linewidth=2, alpha=0.2, zorder=10, color=faneta_color)
             #ax.scatter(result["Epsilon"], result["FanEfficiency"], marker="s", s=20, zorder=10, alpha=0.3, color=faneta_color)

        ax.set_xlabel(r'$ \varepsilon$ ($V_{fs}/V_j$)', fontsize=20)
        ax.set_ylabel(r"[-]", fontsize=20)
        #ax.set_title("UIUC Propeller Database Volumes 1-4", fontsize=12, fontweight='bold')
        from matplotlib.lines import Line2D

        handles, labels = ax.get_legend_handles_labels()
        
        
        #handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Theory"))
        #handles.append(Line2D([0], [0], marker="s", color="black", linestyle="None", markersize=8, label="Experiment"))
        handles.append(Line2D([0], [0],  color=eta_color, linestyle="--", linewidth=2, label=r'$\eta_{propulsive}$'))
        handles.append(Line2D([0], [0],  color=ct_color, linestyle="--", linewidth=2, label=r'$C_T$'))
        handles.append(Line2D([0], [0],  color=cp_color, linestyle="--", linewidth=2, label=r'$C_P$'))
        handles.append(Line2D([0], [0],  marker="s", color=faneta_color,linestyle="-", linewidth=2, label=r'$\eta_{fan}$'))

        ax.legend(handles=handles, loc='lower center', fontsize=20)
        #ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=20)
        plt.xlim(0, 1)  # Set x-axis limits to focus on the range of interest
        plt.ylim(0, 1)  # Set y-axis limits to focus on the range of interest
        fig.savefig("propeller_plots.pdf", bbox_inches="tight")
        

        # Chi-Sigma-Epsilon contour plot
        if results:
                sigma = np.linspace(0.00, 2.0, 100)
                chi = np.linspace(0.00, 2.0, 100)
                CHI, SIGMA = np.meshgrid(chi, sigma)
                EPS = CHI * SIGMA

                fig2 = plt.figure(figsize=(6.0531*2, 3.74110012361*2))
                contour = plt.contourf(SIGMA, CHI, EPS, levels=np.linspace(0, 1, 256), cmap='viridis', antialiased=False)
                # 256 filled levels are ruinous as vector art; rasterize the
                # field so the PDF stays small while text and lines stay vector
                contour.set_rasterized(True)

                plt.ylabel(r"$\chi = \frac{V_{fs}}{V_1}$", fontsize=14)
                plt.xlabel(r"$\sigma = \frac{V_1}{V_j}$", fontsize=14)
                cbar = plt.colorbar(contour, label=r"$\varepsilon = \frac{V_{fs}}{V_j}$")
                cbar.set_label(r"$\varepsilon = \frac{V_{fs}}{V_j}$", fontsize=14)
                cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
                cbar.ax.tick_params(labelsize=14)
                mask = EPS > 1
                plt.contourf(SIGMA, CHI, mask, levels=[0.5, 1.5], colors='gray', alpha=0.9, hatches=['x'])
                plt.contour(SIGMA, CHI, EPS, levels=[1.0], colors='black', linewidths=4)
                plt.xlim(0, 2)
                plt.ylim(0, 2)
                contour_lines = plt.contour(SIGMA, CHI, EPS, levels=np.linspace(0.1, 0.9, 5), alpha = 0.8, colors='white', linewidths=1)
                plt.clabel(contour_lines, inline=True, fontsize=14, fmt='%.1f')

                # Plot experimental data
                for i, result in enumerate(results):
                    plt.scatter(result["Sigma"], result["Chi"], marker='x', color='red', s=100, zorder=10)

                # Plot the line sigma = 0.5*(chi + 1)
                sigma_line = np.linspace(0.5, 2, 100)
                chi_line = 2 - 1/sigma_line
                plt.plot(sigma_line, chi_line, '--', linewidth=3, color='white',  label=r'$\chi = 2 - 1/\sigma$')

                # Plot experimental data with labels
                #for i, result in enumerate(results):
                #    for j in range(len(result["Chi"])):
                #        plt.text(result["Sigma"][j], result["Chi"][j], str(j+1), fontsize=14, ha='center', va='bottom')

                #plt.title("UIUC Propeller Database", fontsize=14, fontweight='bold')
                handles2 = [
                    Line2D([0], [0], marker='x', color='red',  linestyle='', markersize=10, label='UIUC database'),
                    Line2D([0], [0], color='white', linestyle='--', linewidth=3, label=r'$\chi = 2 - 1/\sigma$')
                ]
                plt.legend(handles=handles2, fontsize=14, loc='lower left')
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()
              
                try:
                    _save(fig2, "chi_sigma_eps_contour_fullDB.pdf")
                except PermissionError:
                    print("Warning: Could not save PDF (file may be open). Saving as SVG instead.")
                    fig2.savefig("chi_sigma_eps_contour_backup.svg", bbox_inches="tight")
                #plt.show()

                # Jaero-Phitip-Chi contour plot
                Jrange = np.linspace(0.01, 3.3, 100) / (2 * np.pi)
                Phirange = np.linspace(0.01, 3.3, 100) / (2 * np.pi)
                J_mesh, Phi_mesh = np.meshgrid(Jrange, Phirange)
                Chirange = np.divide(J_mesh, Phi_mesh)

                fig3, ax3 = plt.subplots(figsize=(6.0531*2, 3.74110012361*2))
                contour3 = ax3.contourf(J_mesh, Phi_mesh, np.minimum(Chirange, 1.5), levels=50, cmap='viridis', vmin=0.25, vmax=1.6, extend='both')
                cbar = plt.colorbar(contour3, ax=ax3, label=r'$\chi$', extend='both')
                cbar.set_label(r"$\chi$", fontsize=12)
                plt.contour(J_mesh, Phi_mesh, Chirange, levels=[1.0], colors='white', linewidths=2)
                cbar.set_ticks([ 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
                cbar.ax.tick_params(labelsize=12)
                
                for i, result in enumerate(results):
                    #ax3.scatter(result["Jaero"], result["Phitip"], marker='x', s=100, linewidth=2, zorder=10, color='red')
                    ax3.plot(result["Jaero"], result["Phitip"], marker='x',  linewidth=2, zorder=10, color='red')
                    #for j in range(len(result["Jaero"])):
                    #    ax3.text(result["Jaero"][j], result["Phitip"][j], str(j+1), fontsize=10, ha='center', va='bottom')
                
                ax3.set_xlabel(r'$J_{\mathrm{aero}} = V_{\mathrm{fs}}/U_{\mathrm{tip}}$', fontsize=12)
                ax3.set_ylabel(r'$\Phi_{\mathrm{tip}} = V_1/U_{\mathrm{tip}}$', fontsize=12)
                ax3.set_title("UIUC Propeller Database Volumes 1-4 - Jaero-Phitip-Chi", fontsize=12, fontweight='bold')
                #ax3.grid(True, alpha=0.3)
                handles3 = [Line2D([0], [0], marker='x', color='red', markerfacecolor='red', markersize=10, label='UIUC database')]
                ax3.legend(handles=handles3, fontsize=12, loc='upper left')
                #ax3.grid(True, alpha=0.3)
                #fig3.savefig("jaero_phitip_chi_contour.pdf", bbox_inches="tight")
                plt.show()


                


                Phitip = np.concatenate([r["Phitip"] for r in results])
                Psitip = np.concatenate([r["Psitip"] for r in results])
                faneta = np.concatenate([r["FanEfficiency"] for r in results])
                
                # Filter valid data
                valid_idx = np.where(faneta > 0)[0]
                Phitip_valid = Phitip[valid_idx]
                Psitip_valid = Psitip[valid_idx]
                faneta_valid = faneta[valid_idx]



                plt.figure(figsize=(6.0531*2, 3.74110012361*2))
                for result in results:
                #    plt.scatter(result["Phitip"], result["Psitip"], c=result["FanEfficiency"], marker='s', s=10, cmap='viridis', vmin=0, vmax=1,alpha=0.5, zorder=10)
                    plt.plot(result["Phitip"], result["Psitip"], linewidth=2, alpha=0.2, zorder=5, color='black')
                _sc = plt.scatter(Phitip_valid, Psitip_valid, c=faneta_valid, s=20, cmap='viridis', vmin=0, vmax=1, edgecolors='black', linewidths=0.5, zorder=5)
                _sc.set_rasterized(True)   # thousands of markers
                #plt.plot(Phitip_valid, Psitip_valid, linewidth=2, alpha=0.2, zorder=5, color='black')
                cbar = plt.colorbar(label=r'$\eta_{\mathrm{fan}}$')
                cbar.set_label(r'$\eta_{\mathrm{fan}}$', fontsize=20)
                cbar.ax.tick_params(labelsize=14)
                plt.xlabel(r'$\Phi_{\mathrm{tip}} $', fontsize=20)
                plt.ylabel(r'$\Psi_{\mathrm{tip}}$', fontsize=20)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlim(0.25 / (2 * np.pi), 2.75 / (2 * np.pi))
                plt.ylim(0, 1 / (2 * np.pi) ** 2)
                plt.grid(True, alpha=0.3)
                _save(plt.gcf(), "fan_efficiency_contour_scatter.pdf")
                plt.show()

                
                # Create grid for contour plot
                from scipy.interpolate import griddata
                phi_grid = np.linspace(Phitip_valid.min(), Phitip_valid.max(), 100)
                psi_grid = np.linspace(Psitip_valid.min(), Psitip_valid.max(), 100)
                PHI, PSI = np.meshgrid(phi_grid, psi_grid)
                
                # Interpolate faneta onto the grid
                faneta_grid = griddata((Phitip_valid, Psitip_valid), faneta_valid, (PHI, PSI), method='linear')
                
                # Create contour plot
                #contour = plt.contourf(PHI, PSI, faneta_grid, levels=20, cmap='viridis', vmin=0, vmax=1)
                plt.figure(figsize=(6.0531*2, 3.74110012361*2))
                contour = plt.contourf(PHI, PSI, faneta_grid, levels=5, cmap='viridis', vmin=0, vmax=1)
                cbar = plt.colorbar(contour, label=r'$\eta_{\mathrm{fan}}$')
                #cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
                lines = plt.contour(PHI, PSI, faneta_grid, levels=np.linspace(0.3, 0.95, 5), colors='black', linewidths=1)
                plt.clabel(lines, inline=True, fontsize=10, fmt='%.1f')
                #plt.scatter(Phitip_valid, Psitip_valid, c=faneta_valid, s=10, cmap='viridis', vmin=0, vmax=1, edgecolors='black', linewidths=0.5, zorder=5)
                plt.xlabel(r'$\Phi_{\mathrm{tip}} = V_1/U_{\mathrm{tip}}$', fontsize=12)
                plt.ylabel(r'$\Psi_{\mathrm{tip}} = \frac{V_j^2 - V_{fs}^2}{ 2 U_{\mathrm{tip}}^2}$', fontsize=12)
                plt.title("Fan Efficiency Contour - UIUC Propeller Database", fontsize=12, fontweight='bold')
                plt.savefig("fan_efficiency_contour.png", bbox_inches="tight", dpi=300)


                plt.show()


if __name__ == "__main__":
    main()

