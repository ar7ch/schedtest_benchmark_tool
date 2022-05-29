#!/usr/bin/python3

from __future__ import annotations

import io
from typing import List, Tuple
import os
from dataclasses import dataclass, field
import util
import enum

PROC_NUMBER = 2

@dataclass
class Task:
    c_i: int = field(default=0)
    d_i: int = field(default=0)
    p_i: int = field(default=0)

    def as_tuple(self) -> Tuple[int, int, int]:
        return self.c_i, self.d_i, self.p_i

@dataclass
class TaskSystem:
    m: int = field(default=PROC_NUMBER)
    n: int = field(default=0)
    n_heavy: int = field(default=0)
    util: float = field(default=0)
    d_ratio: float = field(default=0)
    pd_ratio: float = field(default=0)
    tasksets: List[List[Task]] = field(default_factory=list)  # nested lists in case of a multiple tasksets with same parameters (easier to handle)

    def sysparam_tuple(self) -> Tuple[int, int, float, float, float]:
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
class TestSet:
    """
        Mostly a container for a collection of TaskSystems.
        Automatically detects varying quantity for further use (in comparisions and plotting)
    """
    tasksys_list: List[TaskSystem] = field(default_factory=list)
    varying_param: VaryingParameters = field(default=None)
    input_file_name: str = field(default=None)

    def __init__(self, tasksys_list: List[TaskSystem], input_file_name: str):
        self.tasksys_list = tasksys_list
        self.input_file_name = input_file_name
        if len(tasksys_list) > 0:
            self.detect_varying_parameter()
        if self.varying_param is not None:
            util.print_if_interactive(f'Autodetect varying parameter: {str(self.varying_param)}', __name__)
        else:
            raise ValueError('Failed to autodetect varying parameter')

    def get_total_tasksets_num(self) -> int:
        ans = 0
        for tasksys in self.tasksys_list:
            ans += len(tasksys.tasksets)
        return ans

    def get_varying_parameters(self) -> List:
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
