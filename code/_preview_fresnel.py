import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
RES = r"D:\workbudd成果\展开式逆散射网络的不动点与收敛性理论研究\results"
fig, axes = plt.subplots(2, 3, figsize=(12,7))
for j,(t,f) in enumerate([('FoamDielInt',2),('FoamTwinDiel',3),('FoamDielExt',4)]):
    d = np.load(os.path.join(RES, 'fresnel_recon_%s_%d.npz'%(t,f)))
    axes[0,j].imshow(d['Y'], vmin=0, vmax=1, cmap='gray')
    axes[0,j].set_title('truth %s %dGHz'%(t,f)); axes[0,j].axis('off')
    axes[1,j].imshow(d['chi_on'], vmin=0, vmax=0.5, cmap='gray')
    axes[1,j].set_title('recon (proj on), max=%.2f'%d['chi_on'].max())
    axes[1,j].axis('off')
plt.tight_layout()
out = os.path.join(RES, 'fresnel_recon_preview.png')
plt.savefig(out, dpi=80)
print('saved', out)
