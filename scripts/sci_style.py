"""SCI-journal house figure style (aligned with sci-figure-style skill)."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Original reference styles preserved for compatibility
MODEL_STYLES = [
    ('tcn',          'TCN',          '#4A90E2', '-',  'o', 1.8, 6.5),
    ('lstm',         'LSTM',         '#FF7F2A', '--', 's', 1.8, 6.5),
    ('informer',     'Informer',     '#3CC21C', '-.', '^', 1.8, 6.5),
    ('slgformer',    'SLGformer',    '#D62728', '--', 'D', 2.0, 6.5),
    ('graphwavenet', 'GraphWaveNet', '#8B6CF6', '-',  'v', 1.8, 6.5),
    ('var',          'VAR',          '#17A589', ':',  'P', 1.7, 6.5),
    ('persistence',  'Persistence',  '#7F8C8D', ':',  'X', 1.7, 6.2),
]

# Fixed SSAH model palette & line/marker styles across all manuscript figures
SSAH_MODEL_STYLES = {
    'Hybrid_HWB_XGB': {'label': 'Hybrid (HWB+XGB)', 'color': '#D62728', 'ls': '--', 'marker': 'D', 'lw': 2.0, 'ms': 6.2, 'zorder': 5},
    'XGBoost':        {'label': 'XGBoost',          'color': '#4A90E2', 'ls': '-',  'marker': 'o', 'lw': 1.9, 'ms': 6.0, 'zorder': 4},
    'KNN':            {'label': 'KNN',              'color': '#3CC21C', 'ls': '-.', 'marker': '^', 'lw': 1.8, 'ms': 6.0, 'zorder': 3},
    'SVR':            {'label': 'SVR (RBF)',        'color': '#FF7F2A', 'ls': '--', 'marker': 's', 'lw': 1.8, 'ms': 5.8, 'zorder': 3},
    'ANN':            {'label': 'ANN (MLP)',        'color': '#8B6CF6', 'ls': '-',  'marker': 'v', 'lw': 1.8, 'ms': 6.0, 'zorder': 3},
    '1D-CNN':         {'label': '1D-CNN',           'color': '#17A589', 'ls': ':',  'marker': 'P', 'lw': 1.8, 'ms': 6.0, 'zorder': 3},
    'Ridge':          {'label': 'Ridge',            'color': '#7F8C8D', 'ls': ':',  'marker': 'X', 'lw': 1.7, 'ms': 5.8, 'zorder': 2},
    'EB_HWB':         {'label': 'EB-HWB (physics)', 'color': '#1B4F72', 'ls': '-',  'marker': 'p', 'lw': 1.9, 'ms': 6.0, 'zorder': 4},
    'EB_Simple':      {'label': 'EB-Simple',        'color': '#5D6D7E', 'ls': '--', 'marker': 'h', 'lw': 1.6, 'ms': 5.8, 'zorder': 2},
    'Linear_dT':      {'label': 'Linear ΔT',        'color': '#95A5A6', 'ls': ':',  'marker': '*', 'lw': 1.6, 'ms': 6.2, 'zorder': 2},
    'Persistence':    {'label': 'Persistence',      'color': '#BDC3C7', 'ls': ':',  'marker': '+', 'lw': 1.5, 'ms': 6.0, 'zorder': 1},
}

SENSOR_COLS = {'S2-1': '#F26B6B', 'S2-2': '#6CD94A', 'S2-3': '#5B9CF0', 'S2-4': '#F7944F'}
SENSOR_MARKERS = {'S2-1': 'o', 'S2-2': 's', 'S2-3': '^', 'S2-4': 'D'}
SEED_COLS = ['#D62728', '#4A90E2', '#3CC21C', '#FF7F2A', '#8B6CF6']
SEED_MARKERS = ['o', 's', '^', 'D', 'v']
GRID_COL = '#B0B7BB'
GRID_LS = (0, (5, 4))

SHADE_TRAIN = '#EBF5FB'
SHADE_VAL = '#FEF9E7'
SHADE_TEST = '#FDEDEC'
SHADE_MINT = '#E8F8F5'


def apply_rc():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 10.5,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'legend.fontsize': 9.5,
        'axes.linewidth': 1.6,
        'axes.edgecolor': '#000000',
        'axes.axisbelow': True,
    })


def sci_style(ax, minor_x=True, minor_y=True, no_offset=True):
    """Full black frame, inward ticks on all 4 sides, minor ticks, dashed grey grid."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.6)
        sp.set_color('#000000')
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True,
                   length=5.0, width=1.3, colors='#000000', pad=5)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True,
                   length=2.8, width=0.9, colors='#000000')
    if minor_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    if minor_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.tick_params(axis='y', which='minor', left=False, right=False)
    ax.grid(True, which='major', color=GRID_COL, linestyle=GRID_LS, linewidth=1.0)
    if no_offset:
        try:
            ax.ticklabel_format(useOffset=False, axis='both')
        except AttributeError:
            pass


def style_colorbar(cbar, label, fontsize=10.5):
    cbar.set_label(label, fontsize=fontsize)
    cbar.outline.set_linewidth(1.3)
    cbar.outline.set_edgecolor('#000000')
    cbar.ax.tick_params(direction='in', length=3.8, width=1.0, colors='#000000', labelsize=9.5)


def boxed_legend(handles_from, loc='lower center', ncol=4, anchor=(0.5, 0.0), fontsize=9.2, fig=None, handles=None, labels=None, **extra):
    """Rounded black-framed legend. Pass an Axes or explicit handles/labels."""
    kw = dict(loc=loc, ncol=ncol, frameon=True, fancybox=True, edgecolor='#000000',
              framealpha=1.0, fontsize=fontsize)
    kw.update(extra)
    if fig is not None:
        if handles is None:
            h, l = handles_from.get_legend_handles_labels()
        else:
            h, l = handles, labels
        leg = fig.legend(h, l, bbox_to_anchor=anchor, **kw)
    else:
        if handles is not None:
            leg = handles_from.legend(handles, labels, **kw)
        else:
            leg = handles_from.legend(**kw)
    leg.get_frame().set_linewidth(1.3)
    return leg


def caption(ax, text, y=-0.22, fontsize=11):
    """Sub-caption '(a) ...' placed under the axes (below the x-label)."""
    ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va='top', fontsize=fontsize)


def save_both(fig, stem, folder='.'):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, f'{stem}.png'), dpi=600, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(os.path.join(folder, f'{stem}.pdf'), bbox_inches='tight', pad_inches=0.05)
    fig.savefig(os.path.join(folder, f'{stem}.svg'), bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
