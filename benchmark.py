#!/usr/bin/python3

from __future__ import annotations
from typing import List, TextIO
import argparse
import time
import os
import numpy as np
from dataclasses import dataclass, field
import pickle

from util import tsparser, profiler
import util


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

    def __init__(self, input_file_name=None, _output_dir: str=None, write_to_file=True):
        if write_to_file:
            import datetime
            date = datetime.datetime.now()
            date_str = str(date.isoformat(timespec='seconds'))
            input_file_basename = os.path.basename(input_file_name)
            dirname = f'run_{date_str}_{os.path.splitext(input_file_basename)[0]}'
            if _output_dir is not None:
                dirname = os.path.basename(_output_dir)
            self.output_dir = os.path.join(os.getcwd(), dirname)
            try:
                os.mkdir(self.output_dir)
            except FileExistsError:
                pass
            self.output_file_fd = open(self.join_filename('log.txt'), 'a+')
            print(f"# dumping file {input_file_basename} on {date_str}", file=self.output_file_fd)

    def join_filename(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    def write(self, msg: str) -> None:
        util.print_if_interactive(msg, __name__)
        if self.output_file_fd is not None:
            print(msg, file=self.output_file_fd)

    def cleanup_output_dir(self):
        from shutil import rmtree
        rmtree(self.output_dir)

    def __del__(self):
        if self.output_file_fd is not None:
            self.output_file_fd.close()

    def dump_results(self, obj: list[EvalResults]):
        with open(self.join_filename('dump.bin'), 'w+b') as dump_fd:
            pickle.dump(obj, dump_fd)

    def restore_results(self, path: str):
        with open(os.path.abspath(path), 'rb') as load_fd:
            loaded_obj = pickle.load(load_fd)
        return loaded_obj


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
        for taskset in tasksys.tasksets:
            inp = f'{tasksys.m} {tasksys.n}'
            for task in taskset:
                for el in task.as_tuple():
                    inp += f' {el}'
            try:
                completed_run = profiler.profile(os.path.abspath(executable), inp, trials=1, input_from_file=False)
            except ValueError as err:
                OutputRecord().write(f'Error running {os.path.basename(executable)}: {str(err)}, probably out of memory or the executable has wrong output format. Skipping')
                ets.tasksets_runs.append(profiler.CompletedRun())
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
        ets.sched_ratio /= len(tasksys.tasksets)
        ets.sched_ratio *= 100
        ets.avg_states /= len(tasksys.tasksets)
        ets.avg_rt /= len(tasksys.tasksets)
        results.evals_list.append(ets)
    return results


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
        OutputRecord().write(f'({i + 1}/{len(executables_list)}) Evaluating {os.path.basename(_exec)}')
        tic = time.perf_counter()
        res: EvalResults = evaluate(test_set, _exec)
        res.label = _exec
        results_executables.append(res)
        tac = time.perf_counter()
        diff = tac - tic
        time_meas.append(diff)
        OutputRecord().write(f'({i + 1}/{len(executables_list)}) Done (completed in {diff:0.4f} s)')
    main_tac = time.perf_counter()
    main_diff = main_tac - main_tic
    OutputRecord().write(f'Experiment completed in {main_diff:0.4f} s')
    OutputRecord().write(tuple(time_meas))
    return results_executables


def plot_results(test_set: tsparser.TestSet, results_list: List[EvalResults], plot_states=False, plot_runtime=False, plot_sched=False, print_filename=True, open_plots=True):
    import matplotlib.pyplot as plt
    COLORS = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink']
    x_values = test_set.get_varying_parameters()

    def plot_res(y_values_list: list, fig_num: int, y_str: str, labels: List[str]):
        plt.figure(fig_num)
        plt.grid(True)
        fname_str = '' if not print_filename else f'File {os.path.basename(test_set.input_file_name)}'
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
        plot_res([res.get_states_values() for res in results_list], fig_num, 'number of states', [os.path.basename(res.label) for res in results_list])
        fig_num += 1
    if plot_runtime:
        plot_res([res.get_rt_values() for res in results_list], fig_num, 'runtime, seconds', [os.path.basename(res.label) for res in results_list])
        fig_num += 1
    if plot_sched:
        plot_res([res.get_sched_ratio_values() for res in results_list], fig_num, 'share of schedulable tasks, %', [os.path.basename(res.label) for res in results_list])
        fig_num += 1

    if len(x_values) > 1 and x_values[0] > x_values[1]: # values sequence is descending
        x_values = x_values[::-1]
        data = [res.get_all_runtimes()[::-1] for res in results_list]
    else:
        data = [res.get_all_runtimes() for res in results_list]

    for i in range(len(data)-1):
        plt.figure(fig_num)
        plt.grid(True)
        plt.title(f'Comparison of exact test implementations\n')
        plt.xlabel(f'{str(test_set.varying_param)}')
        plt.ylabel('performance gain, times')
        plt.boxplot((data[i] / data[-1]).T, flierprops=dict(markerfacecolor='g', marker='D'))
        plt.xticks([i for i in range(1, len(x_values)+1)], x_values)
        fig_num += 1
    plt.savefig(OutputRecord().join_filename('boxplot_rt.png'))
    if open_plots:
        plt.show(block=False)
        input('Press enter to exit')


def main():
    input_filename, executables_list, open_plots, dump_path, output_dir = parse()
    is_dump = dump_path is not None
    try:
        OutputRecord(input_filename, output_dir)
        test_set: tsparser.TestSet = tsparser.parse_taskset_file(input_filename)
        if is_dump:
            OutputRecord().write(f"Restoring dump {os.path.basename(dump_path)}")
            evaluations_by_exec = OutputRecord().restore_results(dump_path)
        else:
            evaluations_by_exec: List[EvalResults] = benchmark_executables(test_set, executables_list)
        plot_results(test_set, evaluations_by_exec, plot_states=True, plot_runtime=True, plot_sched=True, print_filename=True, open_plots=open_plots)
        if is_dump:
            OutputRecord().cleanup_output_dir()
        else:
            OutputRecord().write(f'Output files saved to {OutputRecord().output_dir}')
            OutputRecord().dump_results(evaluations_by_exec)
    except KeyboardInterrupt:
        OutputRecord().write(f"Aborting, cleaning up the directory {OutputRecord().output_dir}")
        OutputRecord().cleanup_output_dir()
    except FileNotFoundError as err:
        OutputRecord().write('Failed to open file:' + str(err))

if __name__=='__main__':
    main()
