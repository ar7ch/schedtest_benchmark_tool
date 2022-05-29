#!/usr/bin/python3

from __future__ import annotations
import typing
import os
import sys
import subprocess
import re
import time
import logging
from dataclasses import dataclass, field



def print_if_interactive(msg: str, flag=False, dump_fd=None):
    if __name__=='__main__' or flag:
        print(msg)
        if dump_fd is not None:
            print(msg, file=dump_fd)


@dataclass
class CompletedRun:
    states: int = field(default=0, hash=True)
    runtime: float = field(default=0, hash=True)
    sched: bool = field(default=False, hash=True)


rexp_rt = None
rexp_states = None


def profile(exec_name: str, stdin: str, trials=3, input_from_file=True) -> CompletedRun:
    global rexp_rt
    global rexp_states
    if rexp_rt is None:
        rexp_rt = re.compile(r'Elapsed time: (\d*\.\d*(e(\+|\-)\d*)*).', flags=re.MULTILINE)
    if rexp_states is None:
        rexp_states = re.compile(r'(\d*) states generated.')
    avg = 0
    states = 0
    sched = True
    inp = stdin 
    if input_from_file:
        inp = ""
        with open(stdin, 'r') as stdin_fd:
            in_str = stdin_fd.readline().strip()
            while len(in_str) > 0:
                inp += in_str + ' '
                in_str = stdin_fd.readline().strip()
    for i in range(trials):
        match_rt = None
        match_states = None
        ###tic = time.perf_counter()
        p = subprocess.run([exec_name], input=inp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=3600)
        """
        ### DEBUG
        logger = logging.getLogger(__name__)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(ch)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        tac = time.perf_counter()
        ###
        """
        stdout = str(p.stdout)
        for _line in stdout.split('\n'):
            line = _line.strip()
            if 'NOT SCHEDULABLE' in line:
                sched = False
                continue
            if not match_rt:
                match_rt = rexp_rt.match(line)
            if not match_states:
                match_states = rexp_states.match(line)
        if match_rt and match_states:
            states = int(match_states.group(1))
            run_time = float(match_rt.group(1))
            print_if_interactive(f'{i}. rt={run_time}, states={states}')
            #logger.debug(f'running exec takes {tac - tic}')
            avg += run_time
        else:
            raise ValueError(f'Unable to fetch {"runtime" if not match_rt else ""}{"states num" if not match_states else ""} from{exec_name}: {stdout}, input: {inp}, aborting')

    avg = avg / trials
    s_msg = 'SCHEDULABLE'
    if not sched:
        s_msg = 'NOT SCHEDULABLE'
    print_if_interactive('Schedulable')
    print_if_interactive(f'AVG runtime: {avg:.6f} seconds')
    return CompletedRun(states=states, runtime=avg, sched=sched)



def main():
    assert len(sys.argv) == 4, f'usage: {sys.argv[0]} <executable_to_run> <input_to_pass> <number of trials>'
    exec_name = os.path.abspath(sys.argv[1])
    #execs = exec_name.split(',')
    stdin_fname = os.path.abspath(sys.argv[2])
    trials = int(sys.argv[3])
    profile(exec_name, stdin_fname, trials)


if __name__=='__main__':
    main()
