#!/usr/bin/python3

from __future__ import annotations
from typing import List, TextIO, Tuple
import argparse
import time
import os
import numpy as np
from dataclasses import dataclass, field
import pickle

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

    def __init__(self, output_dir=None, write_to_file=True):
        self.output_dir = os.path.abspath(output_dir)
        if write_to_file:
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
        if self.output_file_fd is not None:
            print(msg, file=self.output_file_fd)

    def close_output_file(self):
        if self.output_file_fd is not None:
            self.output_file_fd.close()
            self.output_file_fd = None

    def __del__(self):
        if self.output_file_fd is not None:
            self.output_file_fd.close()


def evaluate(tasksys: tsparser.TaskSystem, executables_list: str):
    input_str = f'{tasksys.m} {tasksys.n}'
    sched = None
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
            if sched is None:
                sched = completed_run.sched
            else:
                if sched != completed_run.sched:
                    raise RuntimeError(f"Results mismatch: {executables_list[0]} reports {sched}, {executable} reports {completed_run.sched}")
            out.write_collection(completed_run.as_tuple(), _end='\t')
        except ValueError as err:
            print(f'Error running {os.path.basename(executable)}: {str(err)}, probably out of memory or the executable has wrong output format.')
            return


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

@dataclass
class EvaluatedTasksystem:
    sched_part:

def parse_evaluation(executables_list: list[str], eval_filename: str):
    from config import n, task_len, result_len
    with open(eval_filename, 'r') as file:
        sysparams_tup = None  # a 5-tuple of system parameters
        prev_tup = None   # previous one (for varying parameter detection)
        exec_results = []
        tasksys_eval = dict() # aggregate results with the same task parameters
        while True:
            in_str = file.readline()
            if len(in_str) <= 0: # got empty string - reached end of file
                break
            elif in_str[0] == '#':  # ignore comment lines
                continue
            vals = []
            # parse numerical tab-delimited values into a list of ints (and floats)
            for v in in_str.split('\t'):  # get values in a single line
                try:
                    vals.append(int(v.strip()))
                except ValueError:
                    vals.append(float(v.strip()))
            if prev_tup is not None:
                prev_tup = sysparams_tup
            sysparams_tup = tuple(vals[0:5])  # first five values are system parameters
            if prev_tup != sysparams_tup:
                global varying_param
                varying_param = detect_varying(prev_tup, sysparams_tup)
            # skip tasks themselves; we are not interested in them anymore
            skip_amount = sysparams_tup[n] * task_len
            vals = vals[skip_amount::]
            # fetch execution results
            while len(vals) > 0:
                res = vals[0:result_len]  # fetch one result
                res = map(int, res)
                exec_results.append(res) # for every executable
            if sysparams_tup not in tasksys_eval:
                tasksys_eval[sysparams_tup] = []
            tasksys_eval[sysparams_tup].append(exec_results)  # add results for this line (if it is a part of series of tests for the same system, results will be aggregated)
    return tasksys_eval



def main():
    input_filename, executables_list, open_plots, dump_path, output_dir = parse()
    write_dir = output_dir is not None
    is_dump = dump_path is not None
    out = OutputRecord(output_dir)
    out.write_to_file(f"input file {os.path.abspath(input_filename)}", comment=True)
    out.write_to_file(f"processors = {config.PROC_NUM}", comment=True)
    out.write_collection(config.header, comment=True)
    tsparser.read_and_evaluate(input_filename, evaluate, [executables_list])
    results_dict = parse_evaluation(executables_list, out.output_file_name)

    '''
    try:
        OutputRecord(input_filename, output_dir)
        test_set: tsparser.TestSet = tsparser.parse_taskset_file(input_filename, evaluate)
        if is_dump:
            OutputRecord().write(f"Restoring dump {os.path.basename(dump_path)}")
            evaluations_by_exec = OutputRecord().restore_results(dump_path)
        else:
            evaluations_by_exec: List[EvalResults] = benchmark_executables(test_set, executables_list)
            OutputRecord().dump_results(evaluations_by_exec)
        plot_results(test_set, evaluations_by_exec, plot_states=True, plot_runtime=True, plot_sched=True, print_filename=True, open_plots=open_plots)
        if is_dump and not write_dir:
            OutputRecord().cleanup_output_dir()
        else:
            OutputRecord().write(f'Output files saved to {OutputRecord().output_dir}')
    except KeyboardInterrupt:
        OutputRecord().write(f"Aborting, cleaning up the directory {OutputRecord().output_dir}")
        OutputRecord().cleanup_output_dir()
    except FileNotFoundError as err:
        OutputRecord().write('Failed to open file:' + str(err))
    '''

if __name__=='__main__':
    main()
