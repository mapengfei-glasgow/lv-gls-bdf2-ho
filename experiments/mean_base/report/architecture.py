"""Draw the implemented encode-process-decode structure as a static figure."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,3.4),layout='constrained');ax.set(xlim=(0,12),ylim=(0,3));ax.axis('off')
    blocks=[('Graph inputs','18 node features\n8 edge features\nReal + virtual nodes'),
            ('Encode','Node and edge MLPs\n128 latent features\nLayer normalisation'),
            ('Process ×10','Independent residual blocks\nEdge updates + aggregation\nNode updates'),
            ('Decode and advance','3-component update Δu\nu(next) = 2u − u(prev) + Δu\nAutoregressive rollout')]
    for i,(title,body) in enumerate(blocks):
        x=.1+3*i
        ax.add_patch(FancyBboxPatch((x,.72),2.7,1.8,boxstyle='round,pad=.04',facecolor='#edf5f7',edgecolor='#497a8c',linewidth=1.2))
        ax.text(x+1.35,2.12,title,ha='center',va='center',fontsize=12,fontweight='bold',color='#214455')
        ax.text(x+1.35,1.42,body,ha='center',va='center',fontsize=9.5,color='#314d5b',linespacing=1.7)
        if i<3:ax.annotate('',xy=(x+2.93,1.62),xytext=(x+2.76,1.62),arrowprops=dict(arrowstyle='->',color='#497a8c',lw=1.5))
    ax.text(6,.25,'Normalisers are stored with the checkpoint; virtual nodes provide hierarchical communication.',ha='center',fontsize=10,color='#526c78')
    fig.savefig(a.out/'model_architecture.svg');fig.savefig(a.out/'model_architecture.png',dpi=180)
