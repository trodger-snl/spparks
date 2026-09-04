"""Digitize Lamb et al. 2023 (Acta Mater., CoNi GammaPrint-700) Fig. 5a/5c SROD
curves and compare them with a random-axis rotation walk (RMS^2 = D*d, Maxwell).

Inputs fig5a.png / fig5c.png are 300-dpi crops of page 6 of the paper's PDF:
  pdftoppm -f 6 -l 6 -r 300 -png 1-s2.0-S1359646223004918-am.pdf hi
  crop (0.07w,0.50h)-(0.40w,0.68h) -> fig5a.png ; (0.62w,0.50h)-(0.95w,0.68h) -> fig5c.png
The pixel calibration in CAL was read off ruler overlays of those crops.
Output: srod_random_walk.png (docs/) and the fitted D printed to stdout.
"""
import numpy as np, json
from PIL import Image
from scipy.stats import maxwell
from scipy.optimize import least_squares
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- digitize: pixel calibration read off ruler crops of the 300-dpi render
CAL = {'a': dict(fn='fig5a.png', x0px=122, pxperum=(639-122)/250, y0px=445, pxperdeg=(445-85)/25, xr=668, yt=14),
       'c': dict(fn='fig5c.png', x0px=100, pxperum=(157-100)/50,  y0px=445, pxperdeg=(445-85)/25, xr=646, yt=14)}
def digitize(c, seed_um):
    im = np.asarray(Image.open(c['fn']).convert('RGB')).astype(float); wh = im.min(2)  # min channel: white lines pop over any hue
    def peaks(cx):
        col = wh[c['yt']+2:c['y0px']-3, cx]; pk=[]
        for i in range(4, len(col)-4):
            if col[i]>=col[i-1] and col[i]>col[i+1] and col[i]-min(col[i-4],col[i+4])>30: pk.append(i+c['yt']+2)
        return pk
    cols = list(range(c['x0px']+2, c['xr']-2))
    seed = c['x0px'] + int(seed_um*c['pxperum'])
    trk = {}
    pk = sorted(peaks(seed)); assert len(pk)==3, ('seed column needs 3 ridges', pk)
    trk[seed] = pk
    for direction in (1,-1):
        prev = list(pk); cx = seed
        while True:
            cx += direction
            if cx not in cols: break
            p = peaks(cx); cur=[]
            for j in range(3):
                cand = [q for q in p if abs(q-prev[j])<=10]
                cur.append(min(cand, key=lambda q: abs(q-prev[j])) if cand else np.nan)
            trk[cx]=cur
            prev = [cur[j] if not np.isnan(cur[j]) else prev[j] for j in range(3)]
    xs = np.array(sorted(trk)); Y = np.array([trk[x] for x in xs])
    y = lambda p: (c['y0px']-p)/c['pxperdeg']
    d = dict(x=(xs-c['x0px'])/c['pxperum'], p95=y(Y[:,0]), mean=y(Y[:,1]), p5=y(Y[:,2]))
    # overlay for a visual check
    fig,ax=plt.subplots(figsize=(8,5)); ax.imshow(im.astype(np.uint8))
    for j,col in enumerate(('r','lime','r')): ax.plot(xs,Y[:,j],'.',color=col,ms=1.5)
    fig.savefig(c['fn'].replace('.png','_track.png'),dpi=110); plt.close(fig)
    return d
A=digitize(CAL['a'],150); C=digitize(CAL['c'],250)
print('digitized columns a:',len(A['x']),' c:',len(C['x']))

# ---- random-walk model: rotation vector ~ 3D Gaussian, sigma^2 = (RMS0^2 + D d)/3 ; |v| ~ Maxwell(sigma)
MEANF = np.sqrt(8/np.pi)          # Maxwell mean / sigma
def sig(d, D, rms0=0.0): return np.sqrt((rms0**2 + D*d)/3)
def mean_model(d, D, rms0=0.0): return MEANF*sig(d,D,rms0)
def band(d, D, rms0=0.0): return maxwell.ppf(0.05, scale=sig(d,D,rms0)), maxwell.ppf(0.95, scale=sig(d,D,rms0))

# fit D on panel (a) mean, 10..260 um
m = (A['x']>5)&np.isfinite(A['mean'])
res = least_squares(lambda p: mean_model(A['x'][m],p[0])-A['mean'][m], [0.5])
D = res.x[0]
resid = mean_model(A['x'][m],D)-A['mean'][m]
print('fitted D = %.3f deg^2/um  rms residual %.2f deg'%(D, np.sqrt((resid**2).mean())))
for h in (1,2,3,5): print('  per-hop angle at dx=%g um: %.2f deg'%(h, np.sqrt(D*h)))
# panel (c): rms0 from the d=0 mean, D carried over -> prediction
c0=C['mean'][(C['x']<10)&np.isfinite(C['mean'])]; rms0 = c0.mean()/MEANF*np.sqrt(3)
pc = mean_model(C['x'],D,rms0)-C['mean']
print('panel c: rms0 %.2f deg (mean at bottom %.2f); prediction rms residual %.2f deg'%(rms0, c0.mean(), np.sqrt((pc[np.isfinite(pc)]**2).mean())))
for q in (10,30,60,100,200,260): i=np.argmin(abs(A['x']-q)); print(' a d=%3d: data mean %.2f [%.1f,%.1f]  model %.2f [%.1f,%.1f]'%(q,A['mean'][i],A['p5'][i],A['p95'][i],mean_model(q,D),*band(q,D)))
for q in (0,100,250,400,475): i=np.argmin(abs(C['x']-q)); print(' c d=%3d: data mean %.2f [%.1f,%.1f]  model %.2f [%.1f,%.1f]'%(q,C['mean'][i],C['p5'][i],C['p95'][i],mean_model(q,D,rms0),*band(q,D,rms0)))
json.dump(dict(D=D, rms0=rms0), open('fit.json','w'))

# ---- plot
BLUE='#2a78d6'; ORANGE='#eb6834'; INK='#0b0b0b'; INK2='#52514e'; GRID='#e6e5e1'; SURF='#fcfcfb'
plt.rcParams.update({'font.size':10,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':SURF,'axes.facecolor':SURF})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
def panel(ax, data, dmax, D, rms0, title):
    d=np.linspace(0.5,dmax,400); lo,hi=band(d,D,rms0)
    ax.fill_between(d,lo,hi,color=BLUE,alpha=0.15,lw=0,label='random-walk model, 90% band')
    ax.plot(d,mean_model(d,D,rms0),color=BLUE,lw=2,label='random-walk model, mean')
    ax.plot(data['x'],data['mean'],color=INK,lw=1.4,label='Lamb et al. 2023, mean (digitized)')
    ax.plot(data['x'],data['p5'],color=INK,lw=0.9,ls=(0,(3,2)),label='Lamb et al. 2023, 5% / 95%')
    ax.plot(data['x'],data['p95'],color=INK,lw=0.9,ls=(0,(3,2)))
    ax.set_title(title,loc='left',fontsize=11,color=INK)
    ax.grid(axis='y',color=GRID,lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(0,dmax); ax.set_ylim(0,30)
panel(ax1,A,265,D,0.0,'(a) Grains originating in the volume')
# the authors' reading and the current coherent cap, panel (a) only
d=np.linspace(0,265,300)
ax1.plot(d,np.where(d<30,3.5*d/30,3.5+0.85*(d-30)/30),color=ORANGE,lw=1.4,ls=(0,(5,2)),label="authors' reading: fast 30 µm, then 0.85°/layer")
ax1.plot(d,2.0*d/30,color=INK2,lw=1.1,ls=(0,(1,2)),label='current coherent cap 2°/30 µm (no spread)')
ax1.set_xlabel('Distance from grain origin (µm)'); ax1.set_ylabel('SROD (°)')
panel(ax2,C,480,D,rms0,'(b) Established grains (reference at dataset bottom)')
ax2.set_xlabel('Distance from dataset bottom (µm)')
ax1.text(150,27.2,'fit: D = %.2f deg²/µm'%D,color=BLUE,fontsize=9)
ax2.text(20,27.2,'prediction with the same D, plus %.1f° initial spread in quadrature'%rms0,color=BLUE,fontsize=9)
h,l=ax1.get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=3,frameon=False,fontsize=8.5,bbox_to_anchor=(0.5,-0.02))
fig.suptitle('SROD vs distance grown: Lamb et al. (Acta Mater. 2023, Fig. 5, GammaPrint-700 PBF-L) against a random-axis rotation walk',fontsize=10.5,color=INK,x=0.01,ha='left')
fig.tight_layout(rect=(0,0.08,1,0.96))
fig.savefig('srod_random_walk.png',dpi=170); print('saved')
