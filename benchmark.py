#!/usr/bin/python3

from __future__ import annotations
from typing import List, TextIO, Tuple, Dict
import argparse
import time
import os
import numpy as np
from dataclasses import dataclass, field
import pickle

import matplotlib.pyplot as plt

from util import tsparser, profiler
import util
import config


varying_param = None

@dataclass
class EvalResults:
    evals_list: List[EvaluatedTaskSystem] = field(default_factory=list)
    label: str = field(default='')

    def get_rt_values(self):
        return [e.avg_rt for e in self.evals_list]

    def get_states_values(self):
        return [e.avg_states for e in self.evals_list]

    def get_sched_ratio_values(self):
        return [e.sched_ratio for e in self.evals_list]

    def get_all_runtimes(self):
        return np.array([e.get_tasksys_runtimes() for e in self.evals_list], dtype=object)


@dataclass
class EvaluatedTaskSystem:
    tasksets_runs: List[profiler.CompletedRun] = field(default_factory=list)
    avg_rt: float = field(default=0)
    avg_states: float = field(default=0)
    sched_ratio: float = field(default=0)

    def get_tasksys_runtimes(self):
        return np.array([tsr.runtime for tsr in self.tasksets_runs])

    def get_tasksys_states(self) -> List[int]:
        return [tsr.states for tsr in self.tasksets_runs]


class OutputRecordSingleton(type):
    def __call__(cls, *args, **kwargs):
        try:
            return cls.__instance
        except AttributeError:
            cls.__instance = super(OutputRecordSingleton, cls).__call__(*args, **kwargs)
            return cls.__instance


@dataclass
class OutputRecord(metaclass=OutputRecordSingleton):
    output_dir: str = field(default=None)
    output_file_fd: TextIO = field(default=None)
    output_file_name: str = field(default="evaluation.txt")
    fig_save_ext: str = field(default=".pdf")

    def __init__(self, output_dir=None):
        if output_dir is not None:
            self.output_dir = os.path.abspath(output_dir)
            os.mkdir(self.output_dir)
            self.output_file_name = self.join_filename(self.output_file_name)
            self.output_file_fd = open(self.output_file_name, 'a+')

    def join_filename(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    def write_collection(self, out_col, _end='\n', comment=False) -> None:
        out_str = ''
        for el in out_col:
            out_str += str(el) + '\t'
        self.write_to_file(out_str, _end, comment)

    def write_to_file(self, msg: str, _end='\n', comment=False):
        pref = ''
        if comment:
            pref += '# '
        print(pref + msg, end=_end, file=self.output_file_fd)

    def info(self, msg: str) -> None:
        util.print_if_interactive(msg, __name__)

    def close_output_file(self):
        if self.output_file_fd is not None:
            self.output_file_fd.close()
            self.output_file_fd = None

    def __del__(self):
        if self.output_file_fd is not None:
            self.output_file_fd.close()

    def savefig(self, fname: str):
        if self.output_dir is not None:
            file_ext = fname + self.fig_save_ext
            file_path = self.join_filename(file_ext)
            plt.savefig(file_path)
            self.info(f'Saved as {file_path}')


def evaluate(tasksys: tsparser.TaskSystem, executables_list: str):
    input_str = f'{tasksys.m} {tasksys.n}'
    sched_value = None  # check if all executables yield the same answer
    # fill in the input
    out = OutputRecord()
    out.write_collection(tasksys.sysparam_tuple(), _end='')
    for task in tasksys.taskset:
        out.write_collection(task.as_tuple(), _end='')
        for el in task.as_tuple():
            input_str += f' {el}'
    for i, executable in enumerate(executables_list):
        try:
            completed_run = profiler.profile(os.path.abspath(executable), input_str, trials=1, input_from_file=False)
            if sched_value is None:
                sched_value = completed_run.sched
            else:
                if sched_value != completed_run.sched:
                    raise RuntimeError(f"Results mismatch: {executables_list[0]} reports {sched_value}, {executable} reports {completed_run.sched}")
            out.write_collection(completed_run.as_tuple(), _end='')  # extra spacing between exec results
        except ValueError as err:
            print(f'Error running {os.path.basename(executable)}: {str(err)}, probably out of memory or the executable has wrong output format.')
            return
    out.write_to_file('', _end='\n')


'''
def evaluate(test_set: tsparser.TestSet, executable) -> EvalResults:
    """
    Evaluates tasksets from TestSet provided using external executable.
    :param test_set:
    :param executable:
    :return: a list of runtimes
    """
    results = EvalResults()
    i = 0
    total_ts = test_set.get_total_tasksets_num()
    for tasksys in test_set.tasksys_list:
        ets = EvaluatedTaskSystem()
        for taskset in tasksys.taskset:
            inp = f'{tasksys.m} {tasksys.n}'
            for task in taskset:
                for el in task.as_tuple():
                    inp += f' {el}'
            try:
                tic = time.perf_counter()
                completed_run = profiler.profile(os.path.abspath(executable), inp, trials=1, input_from_file=False)
            except ValueError as err:
                tac = time.perf_counter()
                OutputRecord().write(f'Error running {os.path.basename(executable)}: {str(err)}, probably out of memory or the executable has wrong output format. Skipping')
                ets.tasksets_runs.append(profiler.CompletedRun(runtime=tac-tic, states=1))
                continue
            ets.tasksets_runs.append(completed_run)
            OutputRecord().write(f'test {i + 1}/{total_ts}: n={tasksys.n}, n_heavy={tasksys.n_heavy}, U={tasksys.util}, '
                                 f'D_ratio={tasksys.d_ratio}, PD_ratio={tasksys.pd_ratio} rt={completed_run.runtime} s, '
                                 f'{"SCHEDULABLE" if completed_run.sched else "NOT_SCHEDULABLE"}, '
                                 f'{completed_run.states} states')
            if completed_run.sched:
                ets.sched_ratio += 1
            ets.avg_rt += completed_run.runtime
            ets.avg_states += completed_run.states
            i += 1
        ets.sched_ratio /= len(tasksys.taskset)
        ets.sched_ratio *= 100
        ets.avg_states /= len(tasksys.taskset)
        ets.avg_rt /= len(tasksys.taskset)
        results.evals_list.append(ets)
    return results
'''

def parse():
    parser = argparse.ArgumentParser(description='Accepts generated task sets as input and benchmarks different exact test executables')
    parser.add_argument("taskset_file", help="file with task sets")
    parser.add_argument("executables_list", help="comma-separated list with executables (relative and absolute paths supported), e.g.: /bin/ex1,../ex2,./ex3")
    parser.add_argument("--noplot", help="don't open windows with plots", action="store_true")
    parser.add_argument("-d", "--dump",  help="path to presaved dump", type=str)
    parser.add_argument("-o", "--output-dir", help="specify custom name for the output directory", type=str)
    args = parser.parse_args()
    open_plots = True
    if args.noplot:
        open_plots = False
    return args.taskset_file, args.executables_list.split(','), open_plots, args.dump, args.output_dir


def benchmark_executables(test_set: tsparser.TestSet, executables_list: List[str]) -> List[EvalResults]:
    results_executables = []
    time_meas = []
    main_tic = time.perf_counter()
    for i, _exec in enumerate(executables_list):
        OutputRecord().info(f'({i + 1}/{len(executables_list)}) Evaluating {os.path.basename(_exec)}')
        tic = time.perf_counter()
        res: EvalResults = evaluate(test_set, _exec)
        res.label = _exec
        results_executables.append(res)
        tac = time.perf_counter()
        diff = tac - tic
        time_meas.append(diff)
        OutputRecord().info(f'({i + 1}/{len(executables_list)}) Done (completed in {diff:0.4f} s)')
    main_tac = time.perf_counter()
    main_diff = main_tac - main_tic
    OutputRecord().info(f'Experiment completed in {main_diff:0.4f} s')
    OutputRecord().info(tuple(time_meas))
    return results_executables


def plot_results(test_set: tsparser.TestSet, results_list: List[EvalResults], plot_states=False, plot_runtime=False, plot_sched=False, print_filename=True, open_plots=True):
    import matplotlib.pyplot as plt
    COLORS = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink']
    x_values = test_set.get_varying_parameters()
    fname_str = '' if not print_filename else f'File {os.path.basename(test_set.input_file_name)}'
    legend_labels = [os.path.basename(res.label) for res in results_list]

    def plot_res(y_values_list: list, fig_num: int, y_str: str, labels: List[str]):
        plt.figure(fig_num)
        plt.grid(True)
        plt.title(f'Comparison of exact test implementations\n{fname_str}')
        plt.xlabel(f'{str(test_set.varying_param)}')
        plt.ylabel(y_str)
        for i, y_values in enumerate(y_values_list):
            cont_color = COLORS[i % len(COLORS)]
            point_color = cont_color
            plt.plot(x_values, y_values, '--', label=labels[i], color=cont_color, alpha=0.8)
            plt.scatter(x_values, y_values, color=point_color, alpha=0.7)
        plt.legend()
        figname = y_str.replace(',', '')
        figname = figname.replace(' ', '_')
        figname += '.png'
        plt.savefig(OutputRecord().join_filename(figname))
    fig_num = 0
    if plot_states:
        plot_res([res.get_states_values() for res in results_list], fig_num, 'number of states', legend_labels)
        fig_num += 1
    if plot_runtime:
        plot_res([res.get_rt_values() for res in results_list], fig_num, 'runtime, seconds', legend_labels)
        fig_num += 1
    if plot_sched:
        plot_res([res.get_sched_ratio_values() for res in results_list], fig_num, 'share of schedulable tasks, %', legend_labels)
        fig_num += 1

    if len(x_values) > 1 and x_values[0] > x_values[1]: # values sequence is descending
        x_values = x_values[::-1]
        data = [res.get_all_runtimes()[::-1] for res in results_list]
    else:
        data = [res.get_all_runtimes() for res in results_list]

    for i in range(len(data)-1):
        plt.figure(fig_num)
        plt.grid(True)
        plt.title(f'Comparison of exact test implementations\n{fname_str}\nPerformance gain of {legend_labels[-1]} compared to {legend_labels[i]}')
        plt.xlabel(f'{str(test_set.varying_param)}')
        plt.ylabel('performance gain, times')
        plt.boxplot((data[i] / data[-1]).T, flierprops=dict(markerfacecolor='g', marker='D'))
        plt.xticks([i for i in range(1, len(x_values)+1)], x_values)
        plt.savefig(OutputRecord().join_filename(f"boxplot_{legend_labels[-1]}_to_{legend_labels[i]}.png"))
        fig_num += 1
    if open_plots:
        plt.show(block=False)
        input('Press enter to exit')

def detect_varying(prev_tsys: Tuple, tsys: Tuple):
    for i, _ in enumerate(tsys):
        if tsys[i] != prev_tsys[i]:
            return i


def parse_evaluation(executables_list: list[str], eval_filename: str):
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
    from config import sched, runtime, states, unsched, total
    execs_avg_rt = []
    execs_avg_rt_sched = []
    execs_avg_rt_unsched = []

    execs_avg_states = []
    execs_avg_states_sched = []
    execs_avg_states_unsched = []

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


        for exec_i in range(exec_num):
            i_sched_rts = [run[exec_i][runtime] for run in sched_runs]
            i_unsched_rts = [run[exec_i][runtime] for run in unsched_runs]
            i_all_rts = [run[exec_i][runtime] for run in runs]

            all_rts_sched[exec_i].append(i_sched_rts)
            all_rts_unsched[exec_i].append(i_unsched_rts)
            all_rts[exec_i].append(i_all_rts)


            execs_avg_rt_sched[exec_i].append(sum(i_sched_rts) / len(i_sched_rts))
            execs_avg_rt_unsched[exec_i].append(sum(i_unsched_rts) / len(i_unsched_rts))
            execs_avg_rt[exec_i].append((sum(i_unsched_rts) + sum(i_sched_rts)) / (len(i_unsched_rts + i_sched_rts)))

            i_sched_states = [run[exec_i][states] for run in sched_runs]
            i_unsched_states = [run[exec_i][states] for run in unsched_runs]

            execs_avg_states_sched[exec_i].append(sum(i_sched_states) / len(i_sched_states))
            execs_avg_states_unsched[exec_i].append(sum(i_unsched_states) / len(i_unsched_states))
            execs_avg_states[exec_i].append((sum(i_unsched_states) + sum(i_sched_states)) / (len(i_unsched_states + i_sched_states)))

    return execs_avg_rt_sched, execs_avg_rt_unsched, execs_avg_rt, execs_avg_states_sched, execs_avg_states_unsched,  execs_avg_states, sched_ratio, all_rts_sched, all_rts_unsched, all_rts




def plot_wrapper(x_values: list, y_values_list: list, fig_num: int, xlabel: str, ylabel: str, title: str, legend_labels: List[str], ylog=False, grid=True):
    COLORS = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink']

    plt.figure(fig_num)
    plt.grid(grid)
    if title is None:
        title = ylabel
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i, y_values in enumerate(y_values_list):
        cont_color = COLORS[i % len(COLORS)]
        marker = ['o', 'x', '*', 'D']
        if ylog:
            plt.semilogy(x_values, y_values, marker=marker[i], label=legend_labels[i], color=cont_color, alpha=0.8)
        else:
            plt.plot(x_values, y_values, marker=marker[i], label=legend_labels[i], color=cont_color, alpha=0.7, markersize=10)
        plt.xticks(x_values)
    plt.legend()


def boxplot_wrapper(x_values: List, y_values_list: List[List], fig_num: int, xlabel: str, ylabel: str, title: str, legend_labels: List[str], grid=True):
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
        plt.grid(grid)
        for num, _ in enumerate(y_i):
            for k, _ in enumerate(y_i[num]):
                y_i[num][k] /= y_cmp[num][k]
        plot_title = f'Runtime reduction of {legend_labels[-1]} compared to {legend_labels[i]}' + title
        plt.title(plot_title)
        plt.boxplot(y_i, flierprops=dict(markerfacecolor='g', marker='D'))
        plt.xticks([i for i in range(1, len(x_values) + 1)], x_values)
        OutputRecord().savefig(plot_title.replace(' ', '_').replace('\n', ''))
        #plt.xlim(min(x_values), max(x_values))


def make_plots(meas_results, x_values, open_plots=True):
    out = OutputRecord()
    ylabels = ['Avg. runtime of schedulable tasksets', 'Avg. runtime among unschedulable tasksets', 'Avg. runtime among all tasksets', 'Avg. number of states among schedulable taksets',
               'Avg. number of states among unschedulable tasksets', 'Avg. number of states among all tasksets', 'Schedulability ratio']
    execs_avg_rt_sched, execs_avg_rt_unsched, execs_avg_rt, execs_avg_states_sched, execs_avg_states_unsched, execs_avg_states, sched_ratio, all_rt_sched, all_rt_unsched, all_rt = meas_results

    lines_legend = ['Bonifaci 2012', 'our modification']

    log_y_axis = [True, True, True, False, False, False, False]

    grid = False
    fig_num = 0
    for fig_num, y_values in enumerate(meas_results[:-4]):
        plot_wrapper(x_values, y_values, fig_num, config.legend[varying_param], ylabels[fig_num], None, lines_legend, ylog=log_y_axis[fig_num], grid=grid)
        out.savefig(ylabels[fig_num].replace(' ', '_'))
    # Sched ratio
    fig_num += 1
    plot_wrapper(x_values, [sched_ratio]*len(meas_results[0]), fig_num, config.legend[varying_param], ylabels[fig_num], None, lines_legend, grid=grid)
    out.savefig(ylabels[fig_num].replace(' ', '_'))
    plt.yticks(list(range(0, 110, 10)))
    fig_num += 1
    """
    fig_num = 0
    plot_wrapper(x_values, execs_avg_rt_sched, fig_num, config.legend[varying_param], ylabels[fig_num], None, lines_legend, ylog=True)
    out.savefig(ylabels[fig_num].replace(' ', '_'))
    fig_num += 1

    plot_wrapper(x_values, execs_avg_rt_unsched, fig_num, config.legend[varying_param], ylabels[fig_num], None, lines_legend, ylog=True)
    out.savefig(ylabels[fig_num].replace(' ', '_'))
    fig_num += 1

    plot_wrapper(x_values, execs_avg_rt, fig_num, config.legend[varying_param], ylabels[fig_num], None, lines_legend, ylog=True)
    out.savefig(ylabels[fig_num].replace(' ', '_'))
    fig_num += 1

    plot_wrapper(x_values, execs_avg_states_sched, fig_num, config.legend[varying_param], ylabels[fig_num], None, lines_legend)
    out.savefig(ylabels[fig_num].replace(' ', '_'))
    fig_num += 1

    plot_wrapper(x_values, execs_avg_states_unsched, fig_num, config.legend[varying_param], ylabels[fig_num], None, lines_legend)
    out.savefig(ylabels[fig_num].replace(' ', '_'))
    fig_num += 1

    plot_wrapper(x_values, execs_avg_states, fig_num, config.legend[varying_param], ylabels[fig_num], None, lines_legend)
    out.savefig(ylabels[fig_num].replace(' ', '_'))
    fig_num += 1

    # Schedulability ratio
    plot_wrapper(x_values, [sched_ratio, sched_ratio], fig_num, config.legend[varying_param], ylabels[fig_num], None, lines_legend)
    out.savefig(ylabels[fig_num].replace(' ', '_'))
    plt.yticks(list(range(0, 110, 10)))
    fig_num += 1
    """
    # Boxplots
    boxplot_wrapper(x_values, all_rt_sched, fig_num, config.legend[varying_param], None, '\namong schedulable tasksets', lines_legend, grid)
    fig_num += len(all_rt)-1 # creates >= 1 figures

    boxplot_wrapper(x_values, all_rt_unsched, fig_num, config.legend[varying_param], None, '\namong unschedulable tasksets', lines_legend, grid)
    fig_num += len(all_rt)-1

    boxplot_wrapper(x_values, all_rt, fig_num, config.legend[varying_param], None, '\namong all tasksets', lines_legend, grid)
    fig_num += len(all_rt)-1

    if open_plots:
        plt.show(block=False)
        input('Press enter to exit')


def main():
    input_filename, executables_list, open_plots, dump_path, output_dir = parse()
    write_dir = output_dir is not None
    is_dump = dump_path is not None
    try:
        if not is_dump:
            out = OutputRecord(output_dir)
            out.write_to_file(f"input file {os.path.abspath(input_filename)}", comment=True)
            out.write_to_file(f"processors = {config.PROC_NUM}", comment=True)
            out.write_collection(config.header, comment=True)
            tsparser.read_and_evaluate(input_filename, evaluate, [executables_list])
            out.info(f'Output files saved to {OutputRecord().output_dir}')
            out.close_output_file()
            parse_eval_file = out.output_file_name
        else:
            if write_dir:
                out = OutputRecord(output_dir)
            else:
                out = OutputRecord(None)
            parse_eval_file = os.path.abspath(dump_path)
            out.info(f"Restoring {os.path.basename(dump_path)}")

        results_dict, x_values = parse_evaluation(executables_list, parse_eval_file)
        meas_results = prepare_plotting_data(results_dict, len(executables_list))
        make_plots(meas_results, x_values)


    except BaseException as err:
        out.info(str(err))
    except KeyboardInterrupt as err:
        out.info('Aborting...')

if __name__=='__main__':
    main()
