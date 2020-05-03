import math
import numpy


class NumpyNFA(object):

    def __init__(self, char=None):
        self.transitions = dict()
        self.states = 1
        if char is not None:
            self.states += 1
            self.transitions[char] = numpy.zeros((self.states, self.states), dtype=bool)
            self.transitions[char][1,0] = True

    def match(self, text):
        state = numpy.zeros(self.states, dtype=bool)
        consumed = ''
        found = None
        state[0] = True
        try:
            state = numpy.matmul(self.transitions['epsilon'], state)
        except Exception:
            pass
        for char in text:
            try:
                state = numpy.matmul(self.transitions[char], state)
            except KeyError:
                return False
            consumed += char
            if state[self.states-1]:
                found = consumed
        return found

    def _combine(self, operation, right=None):
        if operation == '|':
            return self._or(right)
        if operation == 'AND':
            return self._and(right)
        if operation == '*':
            return self._star()
        if operation == '+':
            return self._plus()
        if operation == '?':
            return self._question()
        raise Exception(''.join(operation, ' is not a valid operator'))

    def _or(self, right):
        result = NumpyNFA()
        result.states = self.states + right.states
        for key in set(self.transitions.keys()).union(set(right.transitions.keys())):
            result.transitions[key] = numpy.zeros((result.states, result.states), dtype=bool)
            result.transitions[key][:self.states, :self.states] += self.transitions.get(key, numpy.zeros((self.states, self.states), dtype=bool))
            result.transitions[key][self.states:, self.states:] += right.transitions.get(key, numpy.zeros((right.states, right.states), dtype=bool))
        try:
            result.transitions['epsilon'][result.states-1, self.states-1] = True
            result.transitions['epsilon'][self.states, 0] = True
        except KeyError:
            result.transitions['epsilon'] = numpy.zeros((result.states, result.states), dtype=bool)
            result.transitions['epsilon'][self.states, 0] = True
            result.transitions['epsilon'][result.states-1, self.states-1] = True
        return result


    def _and(self, right):
        result = NumpyNFA()
        result.states = self.states + right.states - 1
        for key in set(self.transitions.keys()).union(set(right.transitions.keys())):
            result.transitions[key] = numpy.zeros((result.states, result.states), dtype=bool)
            result.transitions[key][:self.states, :self.states] += self.transitions.get(key, numpy.zeros((self.states, self.states), dtype=bool))
            result.transitions[key][self.states-1:, self.states-1:] += right.transitions.get(key, numpy.zeros((right.states, right.states), dtype=bool))
        return result

    def _star(self):
        try:
            self.transitions['epsilon'][self.states-1, 0] = True
            self.transitions['epsilon'][0, self.states-1] = True
        except KeyError:
            self.transitions['epsilon'] = numpy.zeros((self.states, self.states), dtype=bool)
            self.transitions['epsilon'][self.states-1, 0] = True
            self.transitions['epsilon'][0, self.states-1] = True
        return self

    def _plus(self):
        try:
            self.transitions['epsilon'][0, self.states-1] = True
        except KeyError:
            self.transitions['epsilon'] = numpy.zeros((self.states, self.states), dtype=bool)
            self.transitions['epsilon'][0, self.states-1] = True
        return self

    def _question(self):
        try:
            self.transitions['epsilon'][self.states-1, 0] = True
        except KeyError:
            self.transitions['epsilon'] = numpy.zeros((self.states, self.states), dtype=bool)
            self.transitions['epsilon'][self.states-1, 0] = True
        return self


    def _finalize(self):
        epsilon = self.transitions.get('epsilon', None)
        if epsilon is not None:
            epsilon = epsilon + numpy.identity(epsilon.shape[0], dtype=bool)
            for i in range(math.ceil(math.log(epsilon.shape[0], 2))):
                epsilon = numpy.matmul(epsilon, epsilon)
            for key in self.transitions:
                self.transitions[key] = numpy.matmul(epsilon, self.transitions[key])
            self.transitions['epsilon'] = epsilon
        return self
