from collections import deque
from .cupynfa import CupyNFA
from .numpynfa import NumpyNFA


def compile(pattern, cupy=False):
    # Define order of operations
    order_of_ops = {'|':0, 'AND':1, '*':-1, '+':-1, '?':-1, '(':-2, ')':-2}

    # Define stacks and some temporary variables
    nfas = deque()
    operations = deque()
    prev_operand = false
    tmp = None

    # Construct NFA from pattern
    for char in pattern:
        # If previous character and this character are operands insert an 'AND'
        if prev_operand and (char not in order_of_ops.keys() or char == '('):
            tmp = char
            char = 'AND'

        # Add character to appropriate stack
        # If Character is an operator
        if char in order_of_ops.keys():
            order = order_of_ops[char]
            # If char is '(' add it stack
            if char == '(':
                operations.append(char)
                prev_operand = False
            # If char is ')' pop stack until reach '('
            elif char == ')':
                while operations[-1] != '(':
                    right = nfas.pop()
                    nfas.append(nfas.pop()._combine(operations.pop(), right))
                operations.pop()
                prev_operand = True
            # If char is a unary operator perform it on top NFA
            elif order == -1:
                nfas.append(nfas.pop()._combine(char))
                prev_operand = True
            # If char is binary operator perform operations of higher order and then it
            elif order >= 0:
                while len(operations) > 0 and order < order_of_ops[operations[-1]]:
                    right = nfas.pop()
                    nfas.append(nfas.pop()._combine(operations.pop(), right))
                prev_operand = False
                operations.append(char)
            # If char is not recognized raise exception
            else:
                raise Exception(''.join(char, ' operator is not recognized'))
        # If character is an operand
        else:
            if cupy:
                nfas.append(CupyNFA(char))
            else:
                nfas.append(NumpyNFA(char))
            prev_operand = True

        # If we inserted an 'AND' add the original character
        if tmp == '(':
            operations.append(tmp)
            prev_operand = False
        elif tmp is not None:
            if cupy:
                nfas.append(CupyNFA(tmp))
            else:
                nfas.append(NumpyNFA(tmp))
            prev_operand = True
        tmp = None

    # Pop out the rest of the operations
    while len(operations) > 0:
        right = nfas.pop()
        nfas.append(nfas.pop()._combine(operations.pop(), right))

    # Check to make sure only have one NFA left
    if len(nfas) != 1:
        raise Exception('Failed to process pattern')

    # Clean up and return final NFA
    return nfas.pop()._finalize()
