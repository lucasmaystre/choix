#!/usr/bin/env python
import argparse
import random


class Tracker(object):

    def __init__(self, lt):
        self._lt = lt
        self._comps = list()
        self._misses = 0

    @property
    def misses(self):
        return self._misses

    @property
    def comparisons(self):
        return self._comps

    def lt(self, a, b):
        try:
            a_loses = self._lt(a, b)
        except ValueError:
            a_loses = random.choice([True, False])
            self._misses += 1
        if a_loses:
            self._comps.append((b, a))
        else:
            self._comps.append((a, b))
        return a_loses


def std_lt(a, b):
    return a < b


def _split(tup, lt):
    pos = random.randint(0, len(tup) - 1)
    pivot = tup[pos]
    lower = list()
    upper = list()
    for elem in tup[:pos] + tup[pos+1:]:
        if lt(elem, pivot):
            lower.append(elem)
        else:
            upper.append(elem)
    return tuple(lower), (pivot,), tuple(upper)


def quicksort(lst, lt=std_lt):
    """Iterative, breadth-first implementation of Quicksort.

    The implementation has an impact on the order of the sequence of
    comparisons. Arguably, this iterative version is better than the recursive
    one, as comparisons are spread more evenly over items across time.
    
    As a rough sketch: instead of spending the first half of the time in
    getting the exact order of the left part of the list, we order increasingly
    small but evenly sized sublists.

    `lt` defines the comparator, with semantics `lt(a, b) == a < b`.
    """
    nxt = [tuple(lst)]
    while len(nxt) < len(lst):
        cur = nxt
        nxt = list()
        for sub in cur:
            if len(sub) > 1:
                parts = _split(sub, lt)
                nxt.extend(parts)
            else:
                nxt.append(sub)
        # Filter out empty sublists.
        nxt = filter(lambda x: len(x) > 0, nxt)
    return tuple(map(lambda x: x[0], nxt))


def quicksort_recursive(lst, lt=std_lt):
    """Recursive, depth-first implementation of Quicksort."""
    if len(lst) < 2:
        return tuple(lst)
    lower, pivot, upper = _split(lst, lt)
    return quicksort(lower, lt) + pivot + quicksort(upper, lt)


def _merge(tup1, tup2, lt):
    idx = 0
    merged = list()
    for x in tup1:
        while idx < len(tup2) and lt(tup2[idx], x):
            merged.append(tup2[idx])
            idx += 1
        merged.append(x)
    merged.extend(tup2[idx:])
    return tuple(merged)


def mergesort(lst, lt=std_lt):
    """Bottom-up, iterative implementation of mergesort.
    
    The implementation has an impact on the order of the sequence of
    comparisons. Arguably, this iterative version is better than the recursive
    one, as comparisons are spread more evenly over items across time.
    
    As a rough sketch: instead of spending the first half of the time in
    getting the exact order of the left part of the list, we order all pairs
    first, and iteratively merge them into larger sorted sublists.

    `lt` defines the comparator, with semantics `lt(a, b) == a < b`.
    """
    # `nxt` contains sorted tuples to be merged.
    nxt = list((x,) for x in lst)
    while len(nxt) > 1:
        cur = nxt
        nxt = list()
        while len(cur) >= 2:
            nxt.append(_merge(cur.pop(), cur.pop(), lt))
        nxt.extend(cur)
    return nxt[0]


def mergesort_recursive(lst, lt=std_lt):
    """Top-down, recursive implementation of mergesort."""
    if len(lst) < 2:
        return tuple(lst)
    mid = len(lst) / 2
    tup1 = mergesort(lst[:mid], lt)
    tup2 = mergesort(lst[mid:], lt)
    return _merge(tup1, tup2, lt)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mergesort', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = _parse_args()
    lst = range(100)
    random.shuffle(lst)
    if args.mergesort:
        sorted_lst = mergesort(lst)
    else:
        sorted_lst = quicksort(lst)
    assert tuple(sorted_lst) == tuple(xrange(100))
    print "success"


if __name__ == '__main__':
    main()
