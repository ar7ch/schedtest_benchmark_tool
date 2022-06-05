#!/usr/bin/python3

from __future__ import annotations

import time
from typing import List, TextIO, Tuple
from dataclasses import dataclass, field
import argparse
import os
import util
from util import tsparser, profiler
import config

xls=0
plot=1

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
    out_fnames = ["evaluation_xls.txt", "evaluation_plotter_q.txt"]

    def __init__(self, output_dir=None):
        self.out_fds = []
        if output_dir is not None:
            self.output_dir = os.path.abspath(output_dir)
            os.mkdir(self.output_dir)
            for fname in self.out_fnames:
                fname = self.join_filename(fname)
                self.out_fds.append(open(fname, 'a+'))

    def files_info(self):
        self.write_to_file(f"processors = {config.PROC_NUM}", comment=True, fdi=(xls, plot))
        self.write_collection(config.header[xls], comment=True, fdi=(xls,))
        self.write_collection(config.header[plot], comment=True, fdi=(plot,))

    def join_filename(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    def write_collection(self, out_col, _end='\n', comment=False, fdi: Tuple=(0,)) -> None:
        out_str = ''
        for el in out_col:
            out_str += str(el) + '\t'
        self.write_to_file(out_str, _end, comment, fdi)

    def write_to_file(self, msg: str, _end='\n', comment=False, fdi: Tuple=(0,)):
        pref = ''
        if comment:
            pref += '# '
        for i in fdi:
            print(pref + msg, end=_end, file=self.out_fds[i])

    def info(self, msg: str, end='\n') -> None:
        print(msg, end=end)

    def __del__(self):
        for fd in self.out_fds:
            if fd is not None:
                fd.close()


def evaluate(tasksys: tsparser.TaskSystem, executables_list: str):
    input_str = f'{tasksys.m} {tasksys.n}'
    sched_value = None  # check if all executables yield the same answer
    # fill in the input
    out = OutputRecord()
    tup_xls = tasksys.sysparam_tuple()
    tup_plt = tasksys.sysparam_tuple()
    for task in tasksys.taskset:
        tup_xls += task.as_tuple()
        tup_plt += task.as_tuple()
        for el in task.as_tuple():
            input_str += f' {el}'
    for i, executable in enumerate(executables_list):
        try:
            tic = time.perf_counter()
            completed_run = profiler.profile(os.path.abspath(executable), input_str, trials=1, input_from_file=False)
            tac = time.perf_counter()
            if sched_value is None:
                sched_value = completed_run.sched
            else:
                if sched_value != completed_run.sched:
                    raise RuntimeError(f"Results mismatch: {executables_list[0]} reports {sched_value}, {executable} reports {completed_run.sched}")
            exec_name = os.path.basename(executable)
            out.info(f'{exec_name} ok ({tac-tic:.6f} s);', end=' ')
            tup_plt += completed_run.as_tuple()
            tup_xls = completed_run.as_tuple() + tup_xls
        except ValueError as err:
            print(f'Error running {os.path.basename(executable)}: {str(err)}, probably out of memory or the executable has wrong output format.')
            return
    out.info('', end='\n')
    out.write_collection(tup_xls, _end='\n', fdi=(xls,))
    out.write_collection(tup_plt, _end='\n', fdi=(plot,))


def parse():
    parser = argparse.ArgumentParser(description='Accepts generated task sets as input and benchmarks different GFP schedulability test executables')
    parser.add_argument("taskset_file", help="file with task sets")
    parser.add_argument("executables_list", help="comma-separated list with executables (relative and absolute paths supported), e.g.: /bin/ex1,../ex2,./ex3")
    parser.add_argument("output_dir", help="specify name for the output directory", type=str)
    args = parser.parse_args()
    return args.taskset_file, args.executables_list.split(','), args.output_dir

def main():
    input_filename, executables_list, output_dir = parse()
    input_filename = os.path.abspath(input_filename)
    try:
        out = OutputRecord(output_dir)
        out.write_to_file(f"input file {input_filename}, executables {executables_list}", comment=True, fdi=(xls, plot))
        out.files_info()
        tsparser.read_and_evaluate(input_filename, evaluate, [executables_list])
        out.info(f'Output files saved to {OutputRecord().output_dir}')
    except KeyboardInterrupt:
        print("Aborting...")
    except (FileExistsError, FileNotFoundError) as err:
        print(err)


if __name__=='__main__':
    main()