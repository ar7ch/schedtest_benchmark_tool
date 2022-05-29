#!/usr/bin/python3

from __future__ import annotations

import io
from typing import TextIO, Dict, List
import sys
import os
import time
from dataclasses import dataclass, field
import profiler
import enum

PROC_NUMBER = 2

executables = ['gfp_mod']
interactive_flag = __name__ == '__main__'




@dataclass
class Task:
    c_i: int = field(default=0)
    d_i: int = field(default=0)
    p_i: int = field(default=0)

    def as_tuple(self) -> tuple[int, int, int]:
        return self.c_i, self.d_i, self.p_i


@dataclass
class TaskSystem:
    m: int = field(default=PROC_NUMBER)
    n: int = field(default=0)
    n_heavy: int = field(default=0)
    util: float = field(default=0)
    d_ratio: float = field(default=0)
    pd_ratio: float = field(default=0)
    tasksets: list[list[Task]] = field(default_factory=list)  # nested lists in case of a multiple tasksets with same parameters (easier to handle)

    def sysparam_tuple(self) -> tuple[int, int, float, float, float]:
        return self.n, self.n_heavy, self.util, self.d_ratio, self.pd_ratio


@enum.unique
class VaryingParameters(enum.IntEnum):
    N = 0
    N_heavy = 1
    Utilization = 2
    D_ratio = 3

    def __str__(self):
        return self.name


@dataclass
class EvaluatedTS:
    runtimes: list[float] = field(default=None)
    states: list[int] = field(default=None)
    sched_ratio: list[float] = field(default=None)


@dataclass
class TestSet:
    """
        Mostly a container for a collection of TaskSystems.
        Automatically detects varying quantity for further use (in comparisions and plotting)
    """
    tasksys_list: list[TaskSystem] = field(default_factory=list)
    varying_param: VaryingParameters = field(default=None)
    input_file_name: str = field(default=None)

    def __init__(self, tasksys_list: list[TaskSystem], input_file_name: str):
        self.tasksys_list = tasksys_list
        self.input_file_name = input_file_name
        if len(tasksys_list) > 0:
            self.detect_varying_parameter()
        if self.varying_param is not None:
            profiler.print_if_interactive(f'Autodetect varying parameter: {str(self.varying_param)}', flag=interactive_flag)
        else:
            raise ValueError('Failed to autodetect varying parameter')

    def get_total_tasksets_num(self) -> int:
        ans = 0
        for tasksys in self.tasksys_list:
            ans += len(tasksys.tasksets)
        return ans

    def get_varying_parameters(self) -> list:
        return [tasksys.sysparam_tuple()[int(self.varying_param)] for tasksys in self.tasksys_list]

    def detect_varying_parameter(self) -> VaryingParameters:
        if self.varying_param is None:
            prev_tsys = self.tasksys_list[0]
            for tsys in self.tasksys_list[1:]:
                tup = prev_tsys.sysparam_tuple()
                prev_tup = tsys.sysparam_tuple()
                if tup != prev_tup:
                    if tsys.n != prev_tsys.n:
                        self.varying_param = VaryingParameters.N
                    elif tsys.n_heavy != prev_tsys.n_heavy:
                        self.varying_param = VaryingParameters.N_heavy
                    elif tsys.util != prev_tsys.util:
                        self.varying_param = VaryingParameters.Utilization
                    elif tsys.d_ratio != prev_tsys.d_ratio:
                        self.varying_param = VaryingParameters.D_ratio
                    else:
                        assert False
        return self.varying_param


def parse_taskset_file(input_file_name: str) -> TestSet:
    """
    Parses input file (CPLEX-generated tasksets) into a collection of TaskSystem representation (a TestSet).
    :param input_file_name:
    :return:
    """
    testset_arr = []
    tasksys_dict = dict()
    with open(os.path.abspath(input_file_name), 'r') as inp_file:
        while True:
            in_str = inp_file.readline()
            if len(in_str) <= 0:
                break
            elif in_str[0] == '#':
                continue
            vals = []
            # parse numerical tab-delimited values into a list of ints (and floats)
            for v in in_str.split('\t'):
                try:
                    vals.append(int(v.strip()))
                except ValueError:
                    vals.append(float(v.strip()))
            sysparams_tup = tuple(vals[0:5])
            if sysparams_tup not in tasksys_dict:
                tasksys_dict[sysparams_tup] = TaskSystem(PROC_NUMBER, *sysparams_tup)
            cur_tasksys = tasksys_dict[sysparams_tup]  # get system parameters
            # cut to tasks description only
            vals = vals[5:]
            taskset = []
            while len(vals) > 0:
                task = Task(*vals[0:3])
                vals = vals[3:]
                taskset.append(task)
            assert len(taskset) == cur_tasksys.n
            cur_tasksys.tasksets.append(taskset)
    return TestSet(list(tasksys_dict.values()), input_file_name)

"""
class OutputRecord:
    instance: OutputRecordInstance = None

    @classmethod
    def get_instance(cls, input_file_name=None, write_to_file=True):
        if input_file_name is not None and cls.instance is None:
            cls.instance = OutputRecordInstance(input_file_name, write_to_file)
            return cls.instance
        elif input_file_name is None and cls.instance is not None:
            return cls.instance
        else:
            raise ValueError('Singleton is already initialized')

    @classmethod
    def replace_instance(cls, input_file_name, write_to_file=True):
        cls.instance = OutputRecordInstance(input_file_name, write_to_file)
"""


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

    def __init__(self, input_file_name=None, write_to_file=True):
        if write_to_file:
            import datetime
            date = datetime.datetime.now()
            date_str = str(date.isoformat(timespec='seconds'))
            input_file_basename = os.path.basename(input_file_name)
            self.output_dir = os.path.join(os.getcwd(), f'run_{date_str}_{os.path.splitext(input_file_basename)[0]}')
            try:
                os.mkdir(self.output_dir)
            except FileExistsError:
                pass
            self.output_file_fd = open(self.join_filename('run_dump.txt'), 'a+')
            print(f"# dumping file {input_file_basename} on {date_str}", file=self.output_file_fd)

    def join_filename(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    def write(self, msg: str) -> None:
        profiler.print_if_interactive(msg, flag=interactive_flag)
        if self.output_file_fd is not None:
            print(msg, file=self.output_file_fd)

    def __del__(self):
        if self.output_file_fd is not None:
            self.output_file_fd.close()



def evaluate(test_set: TestSet, executable) -> EvaluatedTS:
    """
    Evaluates tasksets from TestSet provided using external executable.
    :param test_set:
    :param executable:
    :return: a list of runtimes
    """
    runtimes = []
    states = []
    sched_ratio = []
    i = 0
    total_ts = test_set.get_total_tasksets_num()
    OutputRecord(test_set.input_file_name)
    for tasksys in test_set.tasksys_list:
        avg_rt = 0  # if we're given a task system with different tasksets but identical system parameters, we compute average value of all such runs
        avg_states = 0 # same with states
        sched_num = 0
        for taskset in tasksys.tasksets:
            inp = f'{tasksys.m} {tasksys.n}'
            for task in taskset:
                for el in task.as_tuple():
                    inp += f' {el}'
            completed_run = profiler.profile(os.path.abspath(executable), inp, trials=1, input_from_file=False)
            OutputRecord().write(f'test {i + 1}/{total_ts}: n={tasksys.n}, n_heavy={tasksys.n_heavy}, U={tasksys.util}, '
                        f'D_ratio={tasksys.d_ratio}, PD_ratio={tasksys.pd_ratio} rt={completed_run.runtime} s, '
                        f'{"SCHEDULABLE" if completed_run.sched else "NOT_SCHEDULABLE"}, '
                        f'{completed_run.states} states')
            if completed_run.sched:
                sched_num += 1
            avg_rt += completed_run.runtime
            avg_states += completed_run.states
            i += 1
        sched_num /= len(tasksys.tasksets)
        sched_num *= 100
        avg_states /= len(tasksys.tasksets)
        avg_rt /= len(tasksys.tasksets)
        states.append(avg_states)
        runtimes.append(avg_rt)
        sched_ratio.append(sched_num)
    return EvaluatedTS(runtimes, states, sched_ratio)


def plot_results(test_set: TestSet, evaluations_by_exec: List[EvaluatedTS], plot_states=False, plot_runtime=False, plot_sched=False, print_filename=True):
    import matplotlib.pyplot as plt
    COLORS = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink']
    x_values = test_set.get_varying_parameters()

    def plot_res(y_values_list: list, fig_num: int, y_str: str):
        plt.figure(fig_num)
        plt.grid(True)
        fname_str = '' if not print_filename else f'File {os.path.basename(test_set.input_file_name)}'
        plt.title(f'Comparison of exact test implementations\n{fname_str}')
        plt.xlabel(f'{str(test_set.varying_param)}')
        plt.ylabel(y_str)
        for i, y_values in enumerate(y_values_list):
            cont_color = COLORS[i % len(COLORS)]
            point_color = cont_color
            plt.plot(x_values, y_values, '--', label=os.path.basename(executables[i]), color=cont_color, alpha=0.8)
            plt.scatter(x_values, y_values, color=point_color, alpha=0.7)
        plt.legend()
        figname = y_str.replace(',', '')
        figname = figname.replace(' ', '_')
        figname += '.png'
        plt.savefig(OutputRecord().join_filename(figname))

    fig_num = 0
    if plot_states:
        plot_res([ev.states for ev in evaluations_by_exec], fig_num, 'number of states')
        fig_num += 1
    if plot_runtime:
        plot_res([ev.runtimes for ev in evaluations_by_exec], fig_num, 'runtime, seconds')
        fig_num += 1
    if plot_sched:
        plot_res([ev.sched_ratio for ev in evaluations_by_exec], fig_num, 'share of schedulable tasks, %')
        fig_num += 1
    plt.show()



def main():
    assert (len(sys.argv) == 2)
    input_filename = os.path.abspath(sys.argv[1])
    test_set: TestSet = parse_taskset_file(input_filename)
    main_tic = time.perf_counter()
    evaluations_by_exec = []
    time_meas = []
    for i, _exec in enumerate(executables):
        print(f'({i+1}/{len(executables)}) Evaluating {os.path.basename(_exec)}')
        tic = time.perf_counter()
        res: EvaluatedTS = evaluate(test_set, _exec)
        evaluations_by_exec.append(res)
        tac = time.perf_counter()
        diff = tac - tic
        time_meas.append(diff)
        print(f'({i+1}/{len(executables)}) Done (completed in {diff:0.4f} s)')
    main_tac = time.perf_counter()
    main_diff = main_tac - main_tic
    print(f'Experiment completed in {main_diff:0.4f} s')
    print(tuple(time_meas))
    plot_results(test_set, evaluations_by_exec, plot_states=True, plot_runtime=True, plot_sched=True, print_filename=True)


if __name__ == '__main__':
    main()
