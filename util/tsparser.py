#!/usr/bin/python3

from __future__ import annotations

from typing import List, Tuple, Callable
import os
from dataclasses import dataclass, field
import config

@dataclass
class Task:
    c_i: int = field(default=0)
    d_i: int = field(default=0)
    p_i: int = field(default=0)

    def as_tuple(self) -> Tuple[int, int, int]:
        return self.c_i, self.d_i, self.p_i

@dataclass
class TaskSystem:
    m: int = field(default=config.PROC_NUM)
    n: int = field(default=0)
    n_heavy: int = field(default=0)
    util: float = field(default=0)
    d_ratio: float = field(default=0)
    pd_ratio: float = field(default=0)
    taskset: List[Task] = field(default_factory=list)  # nested lists in case of a multiple tasksets with same parameters (easier to handle)

    def sysparam_tuple(self) -> Tuple:
        return self.n, self.n_heavy, self.util, self.d_ratio, self.pd_ratio




def read_and_evaluate(input_file_name: str, unit_action: Callable, action_args: List):
    """
    Parses input file (CPLEX-generated tasksets) into a collection of TaskSystem representation (a TestSet).
    :param input_file_name:
    :param unit_action action to be performed on a parsed taskset
    :return:
    """
    fname = os.path.abspath(input_file_name)
    total_lines = sum(1 if line[0] != '#' else 0 for line in open(fname))
    lineno = 0
    with open(fname, 'r') as inp_file:
        while True:
            in_str = inp_file.readline()
            lineno += 1
            print(f'Test {lineno}/{total_lines}...', end=' ')
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
            cur_tasksys = TaskSystem(config.PROC_NUM, *sysparams_tup)
            # cut to tasks description only
            vals = vals[5:]
            while len(vals) > 0:
                task = Task(*vals[0:3])
                vals = vals[3:]
                cur_tasksys.taskset.append(task)
            assert len(cur_tasksys.taskset) == cur_tasksys.n
            unit_action(cur_tasksys, *action_args)  # evaluate