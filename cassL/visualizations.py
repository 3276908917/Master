from cassL import camb_interface as ci
from cassL import utils
import matplotlib.pyplot as plt

def pairwise_errors(lhc, errors, x_index, y_index, x_label, y_label):
    if x_index == y_index:
        raise ValueError("The two parameter indices are the same!")
    
    x = lhc[:, x_index]
    y = lhc[:, y_index]

    # Normalize errors for use in coloring
    z = utils.normalize(errors)
    colors = plt.cm.plasma(z)
    
    plt.colorbar(errors)
    plt.scatter(x, y, color=colors)
    plt.title("Emulator errors in {} and {}".format(y_label, x_label))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    

def model_ratios(snap_index, sims, canvas, massive=True, skips=[],
                 subplot_indices=None, active_labels=['x', 'y'],
                 title="Ground truth", omnuh2_str="0.002", cosm=cosm,
                 suppress_legend=False, linewidth=1):
    """
    There are a couple of annoying formatting differences with the power nu
    dictionary which add up to an unpleasant time trying to squeeze it into the
    existing function...

    Here, the baseline is always model 0,
    but theoretically it should be quite easy
    to generalize this function further.
    """
    P_accessor = None
    if massive is True:
        P_accessor = "P_nu"
    elif massive is False:
        P_accessor = "P_no"

    baseline_h = ci.cosm.loc[0]["h"]
    baseline_k = sims[0][snap_index]["k"]

    baseline_p = sims[0][snap_index]["P_nu"] / \
        sims[0][snap_index]["P_no"]
    if P_accessor is not None:
        baseline_p = sims[0][snap_index][P_accessor]

    plot_area = canvas  # if subplot_indices is None
    if subplot_indices is not None:
        if type(subplot_indices) == int:
            plot_area = canvas[subplot_indices]
        else:  # we assume it's a 2d grid of plots
            plot_area = canvas[subplot_indices[0], subplot_indices[1]]
        # No need to add more if cases because an n-d canvas of n > 2 makes no
        # sense.

    k_list = []
    rat_list = []
    for i in range(1, len(sims)):
        if i in skips:
            continue  # Don't know what's going on with model 8
        this_h = cosm.loc[i]["h"]
        this_k = sims[i][snap_index]["k"]

        this_p = sims[i][snap_index]["P_nu"] / \
            sims[i][snap_index]["P_no"]
        if P_accessor is not None:
            this_p = sims[i][snap_index][P_accessor]

        truncated_k, truncated_p, aligned_p = \
            utils.truncator(baseline_k, baseline_p, this_k, this_p)

        label_in = "model " + str(i)
        plot_area.plot(truncated_k, aligned_p / truncated_p,
                       label=label_in, c=colors[i], linestyle=styles[i],
                       linewidth=linewidth)

        k_list.append(truncated_k)
        rat_list.append(aligned_p / truncated_p)

    plot_area.set_xscale('log')
    if 'x' in active_labels:
        plot_area.set_xlabel(r"k [1 / Mpc]", fontsize=24)

    ylabel = r"и($k$)"
    if P_accessor is not None:
        if massive is True:
            ylabel = r"$P_\mathrm{massive} / P_\mathrm{massive, model \, 0}$"
        if massive is False:
            ylabel = r"$P_\mathrm{massless} / P_\mathrm{massless, model \, 0}$"

    if 'y' in active_labels:
        plot_area.set_ylabel(ylabel, fontsize=24)

    if title != "" and title[-1] != ":":
        if len(title) < 2 or title[-2:] != "\n":
            title += ": "
        
    plot_area.set_title(title + r"$\omega_\nu$ = " + omnuh2_str +
                        "; Snapshot " + str(snap_index), fontsize=24)
    
    if not suppress_legend:
        plot_area.legend(fontsize=24, loc='center left',
                         bbox_to_anchor=(1, 0.5))
        
    plot_area.tick_params(labelsize=24)

    return k_list, rat_list


def compare_wrappers(k_list, p_list, sims, snap_index, canvas, massive,
                     subscript, title, skips=[], subplot_indices=None,
                     active_labels=['x', 'y']):
    """
    Python-wrapper (i.e. Lukas') simulation variables feature the _py ending
    Fortran (i.e. Ariel's) simulation variables feature the _for ending
    """

    P_accessor = None
    if massive is True:
        P_accessor = "P_nu"
    elif massive is False:
        P_accessor = "P_no"
    x_mode = P_accessor is None

    # Remember, the returned redshifts are in increasing order
    # Whereas snapshot indices run from older to newer
    z_index = 4 - snap_index

    baseline_h = cosm.loc[0]["h"]

    baseline_k_py = k_list[0] * baseline_h

    baseline_p_py = None
    if x_mode:
        baseline_p_py = p_list[0][z_index]
    else:
        baseline_p_py = p_list[0][z_index] / baseline_h ** 3

    baseline_k_for = sims[0][snap_index]["k"]

    baseline_p_for = sims[0][snap_index]["P_nu"] / \
        sims[0][snap_index]["P_no"]
    if P_accessor is not None:
        baseline_p_for = sims[0][snap_index][P_accessor]

    plot_area = None
    if subplot_indices is None:
        plot_area = canvas
    elif type(subplot_indices) == int:
        plot_area = canvas[subplot_indices]
    else:
        plot_area = canvas[subplot_indices[0], subplot_indices[1]]

    # k_list is the LCD because Ariel has more working cosm than I do
    for i in range(1, len(k_list)):
        if i in skips:
            continue
        this_h = cosm.loc[i]["h"]

        this_k_py = k_list[i] * this_h
        this_p_py = None
        if x_mode is False:
            this_p_py = p_list[i][z_index] / this_h ** 3
        else:
            this_p_py = p_list[i][z_index]

        this_k_for = sims[i][snap_index]["k"]

        this_p_for = sims[i][snap_index]["P_nu"] / \
            sims[i][snap_index]["P_no"]
        if P_accessor is not None:
            this_p_for = sims[i][snap_index][P_accessor]

        truncated_k_py, truncated_p_py, aligned_p_py = \
            utils.truncator(baseline_k_py, baseline_p_py, this_k_py,
                      this_p_py, interpolation=this_h != baseline_h)
        y_py = aligned_p_py / truncated_p_py

        truncated_k_for, truncated_p_for, aligned_p_for = \
            utils.truncator(baseline_k_for, baseline_p_for, this_k_for,
                      this_p_for, interpolation=this_h != baseline_h)
        y_for = aligned_p_for / truncated_p_for

        truncated_k, truncated_y_py, aligned_p_for = \
            utils.truncator_neutral(truncated_k_py, y_py, truncated_k_for, y_for)

        label_in = "model " + str(i)
        plot_area.plot(truncated_k, truncated_y_py / aligned_p_for,
                       label=label_in, c=colors[i], linestyle=styles[i],
                       linewidth=5)

    plot_area.set_xscale('log')
    if 'x' in active_labels:
        plot_area.set_xlabel(r"k [1 / Mpc]")

    ylabel = None
    if x_mode:
        ylabel = r"$ж_i/ж_0$"
    else:
        ylabel = r"$y_\mathrm{py} / y_\mathrm{fortran}$"

    if 'y' in active_labels:
        plot_area.set_ylabel(ylabel)

    plot_area.set_title(title)
    plot_area.legend()

    plot_area.set_title(title)
    plot_area.legend()

