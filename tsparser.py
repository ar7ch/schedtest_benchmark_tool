#!/usr/bin/python3

import typing
import sys
import os
import matplotlib.pyplot as plt
import time

import profiler

m = 2

executables = ['../distrib/gfp_orig', '../distrib/gfp_copy_2']


def parse_taskset_file(input_file: str) -> list:
    testset = []
    with open(os.path.abspath(input_file), 'r') as inp_file:
        in_str = inp_file.readline()
        while len(in_str) > 0:
            taskset = []
            vals = []
            # parse numerical tab-delimited values into a list of ints (and floats)
            for v in in_str.split('\t'):
                try:
                    vals.append(int(v.strip()))
                except ValueError:
                    vals.append(float(v.strip()))
            n, n_heavy, util, d_ratio, pd_ratio = vals[0:5]  # cut off system parameters
            sys_params = (n, n_heavy, util, d_ratio, pd_ratio)
            # next, input task parameters
            vals = vals[5:]
            while len(vals) > 0:
                task = vals[0:3]
                vals = vals[3:]
                taskset.append(task)
            assert len(taskset) == n
            testset.append((n, n_heavy, util, d_ratio, pd_ratio, taskset))
            in_str = inp_file.readline()
    return testset


def evaluate(test_set: tuple, executable) -> dict:
    rt_by_util = dict()
    ctr = dict()
    for i, test in enumerate(test_set):
        n, n_heavy, util, d_ratio, pd_ratio, taskset = test
        inp = f'{m} {n}'
        for task in taskset:
            for el in task:
                inp += f' {el}'
        completed_run = profiler.profile(executable, inp, trials=1, input_from_file=False)
        print(f'test {i+1}/{len(test_set)}: U={util}, rt={completed_run.runtime} s, '
              f'{"SCHEDULABLE" if completed_run.sched else "NOT SCHEDULABLE"}, '
              f'{completed_run.states} states')
        if util in rt_by_util:
            rt_by_util[util] = rt_by_util[util] + completed_run.runtime
            ctr[util] = ctr[util] + 1         
        else:
            rt_by_util[util] = completed_run.runtime
            ctr[util] = 1
    for util, n_times in ctr.items():
        rt_by_util[util] /= n_times
    return rt_by_util


def plot_results(exec_results: list[dict]):
    COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    for i, exec_res in enumerate(exec_results):
        cont_color = COLORS[i % len(COLORS)]
        point_color = cont_color
        plt.plot(exec_res.keys(), exec_res.values(), '--', label=os.path.basename(executables[i]), color=cont_color)
        plt.scatter(exec_res.keys(), exec_res.values(), color=point_color)
    plt.title('comparison of different BFS implementations')
    plt.xlabel('utilization')
    plt.ylabel('runtime, seconds')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    assert (len(sys.argv) == 2)
    test_set = parse_taskset_file(sys.argv[1])
    exec_results = []
    _t1 = time.perf_counter() 
    for i, _exec in enumerate(executables):
        print(f'({i+1}/{len(executables)}) Evaluating {os.path.basename(_exec)}')
        t1 = time.perf_counter()
        res = evaluate(test_set, _exec)
        t2 = time.perf_counter()
        print(f'({i+1}/{len(executables)}) Done (completed in {t2-t1:0.4f} s)')
        exec_results.append(res)
    _t2 = time.perf_counter()
    print(f'Experiment completed in {_t2-_t1:0.4f} s')
    plot_results(exec_results)


if __name__ == '__main__':
    main()
            

        
