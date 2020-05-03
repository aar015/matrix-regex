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
    n = 500
    pattern = '(This )+(is |was )+some text'*2
    text = 'This This This This is is was is was was is was some text'*10
    # pattern = 'a?' * n + 'a' * n
    # text = 'a' * n
    if True:
        total_time = 0
        for i in range(100):
            t_start = time.time()
            nfa = mr.compile(pattern)
            nfa.match(text)
            total_time += time.time()-t_start
        print('Numpy took %.3f ms' % (total_time*10))
    total_time = 0
    for i in range(100):
        t_start = time.time()
        nfa = mr.compile(pattern, cupy=True)
        nfa.match(text)
        total_time += time.time()-t_start
    print('Cupy took %.3f ms' % (total_time*10))
    total_time = 0
    for i in range(100):
        t_start = time.time()
        nfa = regex.compile(pattern)
        nfa.match(text)
        total_time += time.time()-t_start
    print('Regex function took %.3f ms' % (total_time*10))
    total_time = 0
    if True:
        for i in range(100):
            t_start = time.time()
            nfa = re.compile(pattern)
            nfa.match(text)
            total_time += time.time()-t_start
        print('Python function took %.3f ms' % (total_time*10))


if __name__=='__main__':
    run()
