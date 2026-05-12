import matplotlib.pyplot as plt


PLOT_STYLE = {
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 13,
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlepad": 12,
    "axes.labelpad": 10,
    "axes.linewidth": 1.1,
}


def configure_plot_style():
    """Apply readable defaults for all generated evaluation figures."""
    plt.rcParams.update(PLOT_STYLE)


def style_axis(ax, legend=True):
    """Make existing axes readable after labels/legends are attached."""
    ax.tick_params(axis="both", which="major", labelsize=14, width=1.1, length=5)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.title.set_size(18)
    ax.title.set_weight("bold")

    if legend:
        legend_obj = ax.get_legend()
        if legend_obj is not None:
            for text in legend_obj.get_texts():
                text.set_fontsize(13)
            if legend_obj.get_title() is not None:
                legend_obj.get_title().set_fontsize(13)
            legend_obj.get_frame().set_alpha(0.9)

