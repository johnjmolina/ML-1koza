import matplotlib as mpl
import matplotlib.pyplot      as plt
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable

def addtxt(ax,x,y,txt,fs=8,lw=3, clr='k', bclr='w',rot=0):
    return ax.text(x, y, txt, color=clr, ha = 'left', transform=ax.transAxes, rotation=rot, weight='bold', \
                   path_effects=[PathEffects.withStroke(linewidth=lw, foreground=bclr)], fontsize = fs)

def latexit(fmt, data):
    if type(fmt) is list or type(fmt) is tuple or type(fmt):
        return [fmt.format(d) for d in data]
    else:
        return ftm.format(d)

def colorbar(axes, mappable, *, loc="right", size="5%", pad=.1):
    divider = make_axes_locatable(axes)
    cax     = divider.append_axes(loc, size=size, pad=0.1)
    cb      = plt.colorbar(mappable = mappable, cax = cax)
    return cb
def mappable(*, cmap = mpl.cm.viridis, vmin=0, vmax=1):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin,vmax=vmax))
    sm.set_array([])
    return sm

def get_countours(X, Y, fields, level, vmin=0, vmax=1):
    def dmy(field):
        cnt    = plt.contour(X, Y, field, level, vmin=vmin, vmax=vmax)
        points = cnt.collections[0].get_paths()[0].vertices
        if(len(points) > 1):
            return points
        else:
            return None
    return list(filter(None.__ne__, [dmy(field) for field in fields]))
#def get_cbar(axes, *, cmap, vmin=0, vmax=1):
#    cax, kw = mpl.colorbar.make_axes(axes)
#    sm.set_array([])
#    return cax, kw, sm
