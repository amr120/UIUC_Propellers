import numpy as np
import matplotlib.pyplot as plt


eps = np.linspace(0.01, 1, 100)
#sigma = np.linspace(0.01, 2, 100)
sigma = (eps + 1) / 2

Th = 30
rho = 1.225
V = 30

#sigma = 1
#epsilon = 0.5

A_prop = sigma**(-1)*(1 / eps**2 - 1 / eps)**(-1) * (Th / (rho * V**2))

deltasigma = np.linspace(-0.5, 0.5, 100)
#deltasigma = np.linspace(0.0, 1, 100)




#deltaeps = np.linspace(-0.5, 0.5, 100)



eps_grid, deltasigma_grid = np.meshgrid(eps, deltasigma)

A_fan = (sigma + deltasigma_grid)**(-1)*(1 / (eps_grid)**2 - 1 / (eps_grid))**(-1) * (Th / (rho * V**2))


plt.figure(figsize=(6.0531*2, 3.74110012361*2))
contour = plt.contour(eps_grid, deltasigma_grid, ((A_fan - A_prop) / A_prop)*100, levels = [-40, -30, -20, -10, 10, 25, 50, 100, 150], colors='black', linestyles='solid')

contour2 = plt.contour(eps_grid, deltasigma_grid, ((A_fan - A_prop) / A_prop)*100, levels = [0], colors='red')
contour2.collections[0].set_label(r'Propeller Equivalent Disk Area')
#plt.plot(eps, (eps+1)/2 - sigma, color='red', linestyle='-', label=r'Propeller Equivalent Disk Area')
plt.clabel(contour, inline=True, fontsize=14, fmt=lambda x: f'+{x:.0f}%' if x > 0 else f'{x:.0f}%', manual=(plt.get_backend().lower() != "agg"))
plt.clabel(contour2, inline=True, fontsize=14, fmt=lambda x: f'{x:.0f}%', manual=(plt.get_backend().lower() != "agg"))
plt.xlabel(r'$ \epsilon$', fontsize=14)
plt.ylabel(r'$\Delta \sigma$', fontsize=14)
legend_handle = plt.Line2D([0], [0], color='red', linewidth=2, label=r'Propeller Equivalent Disk Area')
plt.legend(handles=[legend_handle], loc='lower left', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('Disk_Area_Changes.pdf', dpi=300, bbox_inches='tight')
plt.show()