import os
import re
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import fsolve
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("colorblind")

DATA_DIR = r"c:\Users\amr200\OneDrive - University of Cambridge\Desktop\UIUC PROPS\UIUC-propDB\UIUC-propDB\volume-3\data"
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
    Vj = solve_vj(Vfs, eta)
    mdot = Thrust / (Vj - Vfs)

    Epsilon = Vfs / Vj
    CTHRUST = Thrust / (mdot * Vj)
    CPOWER = Power / (mdot * Vj**2)
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
        "Thrust": Thrust,
        "Power": Power,
        "Area": Area,
        "mdot": mdot,
        "Epsilon": Epsilon,
        "CTHRUST": CTHRUST,
        "CPOWER": CPOWER,
        "etatheory": etatheory,
        "CTtheory": CTtheory,
        "CPtheory": CPtheory,
    }


def main():
    txt_files = [
        os.path.join(DATA_DIR, name)
        for name in os.listdir(DATA_DIR)
        if name.lower().endswith(".txt")
        and "_static_" not in name.lower()
        and "_geom" not in name.lower()
    ]

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

        fig, ax = plt.subplots(figsize=(6.0531*2, 3.74110012361*2))

        theory_x = np.linspace(0, 1, len(results[0]["eta"]))
        line_eta = ax.plot(theory_x, results[0]["etatheory"],  linestyle="--", linewidth=2)
        line_ct = ax.plot(theory_x, results[0]["CTtheory"],  linestyle="--", linewidth=2)
        line_cp = ax.plot(theory_x, results[0]["CPtheory"],  linestyle="--", linewidth=2)
        
        eta_color = line_eta[0].get_color()
        ct_color = line_ct[0].get_color()
        cp_color = line_cp[0].get_color()

        for i, result in enumerate(results):
            label_prefix = f"{result['brand']} {result['diameter_in']}in {result['rpm']}rpm"
            #ax.scatter(result["Epsilon"], result["CTHRUST"],  marker="s", s=20, zorder=10, facecolors='none', edgecolors=ct_color, label=label_exp)
            ax.scatter(result["Epsilon"], result["CTHRUST"],  marker="s", s=20, zorder=10,  color=ct_color)
            ax.scatter(result["Epsilon"], result["CPOWER"],  marker="s", s=20, zorder=10,  color=cp_color)
            ax.scatter(result["Epsilon"], result["eta"], marker="s", s=20, zorder=10, color=eta_color)

        ax.set_xlabel(r'$ \varepsilon$ ($V_{fs}/V_j$)', fontsize=12)
        ax.set_ylabel(r"$\eta_{propulsive}$, $C_T$, $C_P$ [-]", fontsize=12)
        ax.set_title("UIUC Propeller Database Volume 3", fontsize=12, fontweight='bold')
        from matplotlib.lines import Line2D

        handles, labels = ax.get_legend_handles_labels()
        
        
        handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Theory"))
        handles.append(Line2D([0], [0], marker="s", color="black", linestyle="None", markersize=8, label="Experiment"))
        handles.append(Line2D([0], [0], marker="o", color=eta_color, linestyle="None", markersize=8, label=r'$\eta_{propulsive}$'))
        handles.append(Line2D([0], [0], marker="o", color=ct_color, linestyle="None", markersize=8, label=r'$C_T$'))
        handles.append(Line2D([0], [0], marker="o", color=cp_color, linestyle="None", markersize=8, label=r'$C_P$'))

        ax.legend(handles=handles)
        ax.grid(True, alpha=0.3)
        fig.savefig("propeller_plots.pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()

