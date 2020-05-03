import matrixre as mr
import re
import time
import regex


def run():
    pattern = '(This )+(is |was )+some text'
    text = 'This This This This is is was is was was is was some text'
    nfa = mr.compile(pattern, cupy=True)
    print(nfa.match(text))
    nfa = regex.compile(pattern)
    print(nfa.match(text).group())
    nfa = re.compile(pattern)
    print(nfa.match(text).group())


if __name__=='__main__':
    run()
