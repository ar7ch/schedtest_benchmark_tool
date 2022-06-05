#!/usr/bin/python3

from __future__ import annotations
import os.path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import argparse
import config

varying_param = None

def detect_varying(prev_tsys: Tuple, tsys: Tuple):
    for i, _ in enumerate(tsys):
        if tsys[i] != prev_tsys[i]:
            return i


def parse_evaluation(executables_list: List[str], eval_filename: str):
    from config import n, task_len, result_len
    xvalues = []
    global varying_param
    with open(eval_filename, 'r') as file:
        sysparams_tup = None  # a 5-tuple of system parameters
        prev_tup = None   # previous one (for varying parameter detection)
        tasksys_eval = dict() # aggregate results with the same task parameters
        while True:
            in_str = file.readline()
            if len(in_str) <= 0: # got empty string - reached end of file
                break
            elif in_str[0] == '#':  # ignore comment lines
                continue
            exec_results = []
            vals = []
            # parse numerical tab-delimited values into a list of ints (and floats)
            for v in in_str.strip().split('\t'):  # get values in a single line
                if len(v) < 1: continue
                try:
                    vals.append(int(v.strip()))
                except ValueError:
                    vals.append(float(v.strip()))
            prev_tup = sysparams_tup
            sysparams_tup = tuple(vals[0:5])  # first five values are system parameters
            vals = vals[5::]
            if prev_tup != sysparams_tup and prev_tup is not None:
                if varying_param is None:
                    varying_param = detect_varying(prev_tup, sysparams_tup)
                    xvalues.append(prev_tup[varying_param])
                    xvalues.append(sysparams_tup[varying_param])
                else:
                    xvalues.append(sysparams_tup[varying_param])

            # skip tasks themselves; we are not interested in them anymore
            skip_amount = sysparams_tup[n] * task_len
            vals = vals[skip_amount::]
            # fetch execution results
            while len(vals) > 0:
                res = vals[0:result_len]  # fetch one result
                exec_results.append(res) # for every executable
                vals = vals[result_len::]
            if sysparams_tup not in tasksys_eval:
                tasksys_eval[sysparams_tup] = []
            tasksys_eval[sysparams_tup].append(exec_results)  # add results for this line (if it is a part of series of tests for the same system, results will be aggregated)
    return tasksys_eval, xvalues


def prepare_plotting_data(results: Dict, exec_num):
    from config import sched, runtime, states, unsched, total, queue
    execs_avg_rt = []
    execs_avg_rt_sched = []
    execs_avg_rt_unsched = []

    execs_avg_states = []
    execs_avg_states_sched = []
    execs_avg_states_unsched = []

    execs_peak_queue_map = []
    execs_peak_queue_map_sched = []
    execs_peak_queue_map_unsched = []

    sched_ratio = []

    all_rts_sched = []
    all_rts_unsched = []
    all_rts = []

    for i in range(exec_num):
        execs_avg_rt_sched.append([])
        execs_avg_rt_unsched.append([])
        execs_avg_rt.append([])

        execs_avg_states_sched.append([])
        execs_avg_states_unsched.append([])
        execs_avg_states.append([])

        execs_peak_queue_map_sched.append([])
        execs_peak_queue_map_unsched.append([])
        execs_peak_queue_map.append([])

        all_rts_sched.append([])
        all_rts_unsched.append([])
        all_rts.append([])

    #execs_avg_states = []
    for tasksys_tuple, runs in results.items(): # for every tick of varying N
        # split into sched and unsched categories
        sched_runs = []
        unsched_runs = []
        for run in runs:
            if run[0][sched] == 1:
                sched_runs.append(run)
            elif run[0][sched] == 0:
                unsched_runs.append(run)

        sched_ratio.append(100*len(sched_runs) / len(runs))

        def avg(list):
            try:
                return sum(list) / len(list)
            except ZeroDivisionError:
                return None

        for exec_i in range(exec_num):
            i_sched_rts = [run[exec_i][runtime] for run in sched_runs]
            i_unsched_rts = [run[exec_i][runtime] for run in unsched_runs]
            i_all_rts = [run[exec_i][runtime] for run in runs]

            all_rts_sched[exec_i].append(i_sched_rts)
            all_rts_unsched[exec_i].append(i_unsched_rts)
            all_rts[exec_i].append(i_all_rts)

            execs_avg_rt_sched[exec_i].append(avg(i_sched_rts))
            execs_avg_rt_unsched[exec_i].append(avg(i_unsched_rts))
            execs_avg_rt[exec_i].append((sum(i_unsched_rts) + sum(i_sched_rts)) / (len(i_unsched_rts + i_sched_rts)))

            i_sched_states = [run[exec_i][states] for run in sched_runs]
            i_unsched_states = [run[exec_i][states] for run in unsched_runs]

            execs_avg_states_sched[exec_i].append(avg(i_sched_states))
            execs_avg_states_unsched[exec_i].append(avg(i_unsched_states))
            execs_avg_states[exec_i].append((sum(i_unsched_states) + sum(i_sched_states)) / (len(i_unsched_states + i_sched_states)))

            i_sched_queue_map = [run[exec_i][queue] for run in sched_runs]
            i_unsched_queue_map = [run[exec_i][queue] for run in unsched_runs]

            execs_peak_queue_map_sched[exec_i].append(avg(i_sched_queue_map))
            execs_peak_queue_map_unsched[exec_i].append(avg(i_unsched_queue_map))
            execs_peak_queue_map[exec_i].append((sum(i_unsched_queue_map) + sum(i_sched_queue_map)) / (len(i_unsched_queue_map + i_sched_queue_map)))


    return execs_avg_rt_sched, execs_avg_rt_unsched, execs_avg_rt, execs_avg_states_sched, execs_avg_states_unsched, execs_avg_states, sched_ratio, all_rts_sched, all_rts_unsched, all_rts, execs_peak_queue_map_sched, execs_peak_queue_map_unsched, execs_peak_queue_map




def plot_wrapper(x_values: list, y_values_list: list, fig_num: int, xlabel: str, ylabel: str, title: str, legend_labels: List[str], ylog=False, grid=True):
    COLORS = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink']
    #COLORS = ['black']
    plt.figure(fig_num)
    plt.grid(grid, alpha=0.5, linestyle=':')
    if title is None:
        title = ""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i, y_values in enumerate(y_values_list):
        cont_color = COLORS[i % len(COLORS)]
        marker = ['o', 'x', '*', 'D', 'd', '>', '+',  's']
        if ylog:
            plt.semilogy(x_values, y_values, marker=marker[i % len(marker)], label=legend_labels[i], color=cont_color, alpha=0.8, markersize=8)
        else:
            plt.plot(x_values, y_values, marker=marker[i % len(marker)], label=legend_labels[i], color=cont_color, alpha=0.7, markersize=10)
        plt.xticks(x_values)
        #plt.xlim(min(x_values), max(x_values))
    plt.legend()


def boxplot_wrapper(x_values: List, y_values_list: List[List], fig_num: int, xlabel: str, ylabel: str, title: str, legend_labels: List[str], grid=True, output_dir='', out_format='.pdf'):
    # if the values sequence is descending instead of ascending, reverse it for correct boxplot
    if len(x_values) > 1 and x_values[0] > x_values[1]:
        x_values = x_values[::-1]
        y_values_list = [y_value[::-1] for y_value in y_values_list]

    if title is None:
        title = ''
    y_cmp = y_values_list[-1]
    for i, y_i in enumerate(y_values_list[:-1]):
        plt.figure(fig_num+i)
        plt.xlabel(xlabel)
        if ylabel is None:
            ylabel = 'Runtime reduction, times'
        plt.ylabel(ylabel)
        plt.grid(grid, alpha=0.5, linestyle=':')
        for num, _ in enumerate(y_i):
            for k, _ in enumerate(y_i[num]):
                y_i[num][k] /= y_cmp[num][k]
        plot_title = f'Runtime reduction of {legend_labels[-1]} compared to {legend_labels[i]}' + title
        plt.title(plot_title)
        plt.boxplot(y_i, sym='+')
        plt.xticks([i for i in range(1, len(x_values) + 1)], x_values)
        fig_path = os.path.join(output_dir, plot_title.replace(' ', '_').replace('\n', '') + out_format)
        plt.savefig(fig_path)
        print(f'Saving plot as {fig_path}')

def make_plots(meas_results, x_values, output_dir, labels, show_titles=True, extension='.pdf'):
    def savefig(filename, output_dir=output_dir, out_format=extension):
        fig_path = os.path.join(output_dir, filename + out_format)
        plt.savefig(fig_path, bbox_inches='tight')
        print(f'Saving plot as {fig_path}')

    grid = True
    ylabels = ['Runtime, seconds', 'Avg. runtime of schedulable tasksets', 'Avg. runtime among unschedulable tasksets', 'Avg. runtime among all tasksets', 'Number of states', 'Avg. number of states among schedulable taksets',
               'Avg. number of states among unschedulable tasksets', 'Avg. number of states among all tasksets', 'Schedulability ratio', 'Peak total size of queue and visited map, elements']
    execs_avg_rt_sched, execs_avg_rt_unsched, execs_avg_rt, execs_avg_states_sched, execs_avg_states_unsched, \
        execs_avg_states, sched_ratio, all_rt_sched, all_rt_unsched, all_rt, queue_sched, queue_unsched, queue = meas_results

    if show_titles:
        titles = ['Runtime comparison', 'Avg. runtime of schedulable tasksets', 'Avg. runtime among unschedulable tasksets',
         'Avg. runtime among all tasksets', 'Number of states', 'Avg. number of states among schedulable taksets',
         'Avg. number of states among unschedulable tasksets', 'Avg. number of states among all tasksets',
         'Schedulability ratio', 'Peak queue+map size comparison']
    else:
        titles = [None]*len(ylabels)

    all_labels = []
    postfixes = ['(schedulable tasksets)', '(unschedulable tasksets)', '(all tasksets)']
    for postfix in postfixes:
        for label in labels:
            all_labels.append(label + ' ' + postfix)

    fig_num = 0
    plot_wrapper(x_values, execs_avg_rt_sched + execs_avg_rt_unsched + execs_avg_rt, fig_num, config.legend[varying_param], ylabels[fig_num], titles[fig_num], all_labels, ylog=True)
    savefig('overall_runtime')
    fig_num += 1

    plot_wrapper(x_values, execs_avg_rt_sched, fig_num, config.legend[varying_param], ylabels[fig_num], titles[fig_num], labels, ylog=True)
    savefig(ylabels[fig_num].replace(' ', '_'))
    fig_num += 1

    plot_wrapper(x_values, execs_avg_rt_unsched, fig_num, config.legend[varying_param], ylabels[fig_num], titles[fig_num], labels, ylog=True)
    savefig(ylabels[fig_num].replace(' ', '_'))
    fig_num += 1

    plot_wrapper(x_values, execs_avg_rt, fig_num, config.legend[varying_param], ylabels[fig_num], titles[fig_num], labels, ylog=True)
    savefig(ylabels[fig_num].replace(' ', '_'))
    fig_num += 1

    plot_wrapper(x_values, execs_avg_states_sched + execs_avg_states_unsched + execs_avg_states, fig_num, config.legend[varying_param], ylabels[fig_num], titles[fig_num], all_labels, ylog=True)
    savefig('overall_states')
    fig_num += 1

    plot_wrapper(x_values, execs_avg_states_sched, fig_num, config.legend[varying_param], ylabels[fig_num], titles[fig_num], labels)
    savefig(ylabels[fig_num].replace(' ', '_'))
    fig_num += 1

    plot_wrapper(x_values, execs_avg_states_unsched, fig_num, config.legend[varying_param], ylabels[fig_num], titles[fig_num], labels)
    savefig(ylabels[fig_num].replace(' ', '_'))
    fig_num += 1

    plot_wrapper(x_values, execs_avg_states, fig_num, config.legend[varying_param], ylabels[fig_num], titles[fig_num], labels)
    savefig(ylabels[fig_num].replace(' ', '_'))
    fig_num += 1

    # Schedulability ratio
    plot_wrapper(x_values, [sched_ratio, sched_ratio], fig_num, config.legend[varying_param], ylabels[fig_num], titles[fig_num], labels)
    savefig(ylabels[fig_num].replace(' ', '_'))
    plt.yticks(list(range(0, 110, 10)))
    fig_num += 1

    plot_wrapper(x_values, queue_sched + queue_unsched + queue, fig_num, config.legend[varying_param], ylabels[fig_num], titles[fig_num], all_labels, ylog=True)
    savefig('overall_queue_map')
    fig_num += 1

    # Boxplots
    boxplot_wrapper(x_values, all_rt_sched, fig_num, config.legend[varying_param], None, '\namong schedulable tasksets', labels, grid, output_dir=output_dir)
    fig_num += len(all_rt)-1 # creates >= 1 figures

    boxplot_wrapper(x_values, all_rt_unsched, fig_num, config.legend[varying_param], None, '\namong unschedulable tasksets', labels, grid, output_dir=output_dir)
    fig_num += len(all_rt)-1

    boxplot_wrapper(x_values, all_rt, fig_num, config.legend[varying_param], None, '\namong all tasksets', labels, grid, output_dir=output_dir)
    fig_num += len(all_rt)-1


    plt.show(block=False)
    input('Press enter to exit')


def parse():
    parser = argparse.ArgumentParser(description='Visualizes evaluation data produced by evaluate.py')
    parser.add_argument("input_file", help="path to file with evaluation results (evaluation_plotter.txt)")
    parser.add_argument("executables_labels",
                        help="comma-separated list with labels of corresponding executables (that will appear on plots)")
    parser.add_argument("-o", "--output-dir", help="specify custom name for the output directory", type=str)
    parser.add_argument("-n", "--no-titles", help="Don't show titles on plots", action="store_true")
    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = ''
    titles = True
    if args.no_titles:
        titles = False
    return os.path.abspath(args.input_file), args.executables_labels.split(','), os.path.abspath(args.output_dir), titles


def main():
    filename, executables_labels, output_dir, show_titles = parse()
    print(f"Plotting {filename}")
    results_dict, x_values = parse_evaluation(executables_labels, filename)
    meas_results = prepare_plotting_data(results_dict, len(executables_labels))
    make_plots(meas_results, x_values, output_dir, executables_labels, show_titles=show_titles)

if __name__=='__main__':
    main()