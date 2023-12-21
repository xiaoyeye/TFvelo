import numpy as np
import matplotlib.pyplot as plt

start = -0.2
end = 1.2
x = np.linspace(start, end, 500) 
t = (x-start)/(end-start)
y1 = np.sin(x*np.pi)
y1_n = y1 + np.random.normal(0, 0.2, len(x))
y2 = np.sin(x*np.pi-1)  
y2_n = y2 + np.random.normal(0, 0.2, len(x))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# First subplot
ax1.plot(t, y1, label='TF', color='blue', linewidth=5)
ax1.plot(t, y2, label='Target', color='red', linewidth=5)
ax1.scatter(t, y1_n, c=t, alpha=0.7)
ax1.scatter(t, y2_n, c=t, alpha=0.7)
ax1.set_xlabel('t', fontsize=25)
ax1.set_ylabel('Abundance', fontsize=25)
ax1.legend(fontsize=25)
ax1.set_xticks([])  
ax1.set_yticks([]) 
#ax1.tick_params(axis='both', labelsize=20) 
# Second subplot
ax2.plot(y2, y1, color='purple', linewidth=5)
scatter = ax2.scatter(y2_n, y1_n, c=t, alpha=0.7)
ax2.set_xlabel('Target', fontsize=25)
ax2.set_ylabel('TF', fontsize=25)
ax2.set_xticks([])  
ax2.set_yticks([]) 
#ax2.tick_params(axis='both', labelsize=20) 
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('t', fontsize=25)
cbar.ax.tick_params(labelsize=20)  
# Save the figure
plt.savefig('figures/TF_target.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()



y2 = np.sin(x*np.pi-0.1)  
y2_n = y2 + np.random.normal(0, 0.2, len(x))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# First subplot
ax1.plot(t, y1, label='Unspliced', color='blue', linewidth=5)
ax1.plot(t, y2, label='Spliced', color='red', linewidth=5)
ax1.scatter(t, y1_n, c=t, alpha=0.7)
ax1.scatter(t, y2_n, c=t, alpha=0.7)
ax1.set_xlabel('t', fontsize=25)
ax1.set_ylabel('Abundance', fontsize=25)
ax1.legend(fontsize=25)
ax1.set_xticks([])  
ax1.set_yticks([]) 
#ax1.tick_params(axis='both', labelsize=20) 
# Second subplot
ax2.plot(y2, y1, color='purple', linewidth=5)
scatter = ax2.scatter(y2_n, y1_n, c=t, alpha=0.7)
ax2.set_xlabel('Spliced', fontsize=25)
ax2.set_ylabel('Unspliced', fontsize=25)
ax2.set_xticks([])  
ax2.set_yticks([]) 
#ax2.tick_params(axis='both', labelsize=20) 
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('t', fontsize=25)
cbar.ax.tick_params(labelsize=20)  
plt.tight_layout()
# Save the figure
plt.savefig('figures/U_S.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()
