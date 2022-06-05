#!/usr/bin/python3

from __future__ import annotations
import os
import sys
import subprocess
import re
from dataclasses import dataclass, field


def print_if_interactive(msg: str, flag=False):
    if __name__=='__main__' or flag:
        print(msg)


@dataclass
class CompletedRun:
    states: int = field(default=0, hash=True)
    runtime: float = field(default=0, hash=True)
    sched: bool = field(default=False, hash=True)

    def as_tuple(self):
        s = '1' if self.sched else '0'
        return s, self.runtime, self.states


@dataclass
class CompletedRunWithQueue(CompletedRun):
    queue_states: int = field(default=0, hash=True)

    def as_tuple(self):
        return super().as_tuple() + (self.queue_states,)


rexp_rt = re.compile(r'Elapsed time: (\d*(\.\d*)*(e(\+|\-)?\d*)*).', flags=re.MULTILINE)
rexp_states = re.compile(r'(\d*) states generated.')
rexp_queue_states = re.compile(r'Max(imum)? queue size: (\d*).')


def profile(exec_name: str, stdin: str, trials=3, input_from_file=True) -> CompletedRun:
    avg = 0
    states = 0
    sched = True
    queue_states = 0
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
        match_queue = None
        p = subprocess.run([exec_name], input=inp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=3600)
        stdout = str(p.stdout)
        for _line in stdout.split('\n'):
            line = _line.strip()
            if len(line) <= 0 or line[-1] == '?':
                continue
            if 'NOT SCHEDULABLE' in line:
                sched = False
                continue
            if match_rt is None:
                match_rt = rexp_rt.match(line)
            if match_states is None:
                match_states = rexp_states.match(line)
            if match_queue is None:
                match_queue = rexp_queue_states.match(line)

        if match_rt and match_states and match_queue:
            states = int(match_states.group(1))
            run_time = float(match_rt.group(1))
            queue_states = int(match_queue.group(2))
            print_if_interactive(f'{i}. rt={run_time}, states={states}')
            avg += run_time
        else:
            failure_list = []
            if not match_rt: failure_list.append('runtime')
            if not match_states: failure_list.append('number of states')
            if not match_queue: failure_list.append('max queue')
            raise ValueError(f'Unable to fetch {str(failure_list)} from {exec_name}: {stdout}, input: {inp}')

    avg = avg / trials
    s_msg = 'SCHEDULABLE'
    if not sched:
        s_msg = 'NOT SCHEDULABLE'
    print_if_interactive('Schedulable')
    print_if_interactive(f'AVG runtime: {avg:.6f} seconds')
    return CompletedRunWithQueue(states=states, runtime=avg, sched=sched, queue_states=queue_states)



def main():
    assert len(sys.argv) == 4, f'usage: {sys.argv[0]} <executable_to_run> <input_to_pass> <number of trials>'
    exec_name = os.path.abspath(sys.argv[1])
    #execs = exec_name.split(',')
    stdin_fname = os.path.abspath(sys.argv[2])
    trials = int(sys.argv[3])
    profile(exec_name, stdin_fname, trials)


if __name__=='__main__':
    main()
