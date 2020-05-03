import matrixre as mr
import re
import time
import regex


def print_trans(nfa, char):
    try:
        print(nfa.transitions[char])
    except KeyError:
        pass


def run():
    n = 20
    pattern = '(This )+(is |was )+some text'*2
    text = 'This This This This is is was is was was is was some text'*10
    pattern = 'a?' * n + 'a' * n
    text = 'a' * n
    total_compile = 0
    total_run = 0
    num_trials = 100
    for i in range(num_trials):
        t_start = time.time()
        nfa = mr.compile(pattern, cupy=True)
        total_compile += time.time()-t_start
        t_start = time.time()
        nfa.match(text)
        total_run += time.time()-t_start
    print('Cupy took %.3f ms to compile' % (total_compile*1000/num_trials))
    print('Cupy took %.3f ms to run' % (total_run*1000/num_trials))
    if True:
        total_compile = 0
        total_run = 0
        for i in range(num_trials):
            t_start = time.time()
            nfa = mr.compile(pattern, cupy=False)
            total_compile += time.time()-t_start
            t_start = time.time()
            nfa.match(text)
            total_run += time.time()-t_start
        print('Numpy took %.3f ms to compile' % (total_compile*1000/num_trials))
        print('Numpy took %.3f ms to run' % (total_run*1000/num_trials))



if __name__=='__main__':
    run()
