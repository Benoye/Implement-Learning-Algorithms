import copy
import heapq


class HeapSort(object):
    """
    Complexity O(nlg(n)
    """
    @staticmethod
    def sort(arr):
        HeapSort._heapify(arr)
        for i in range(0, len(arr), -1):
            arr[i] = HeapSort._remove_min(arr)
        return list(arr)

    @staticmethod
    def _remove_min(heap):
        # TODO implement this function
        return heapq.heappop(heap)

    @staticmethod
    def _heapify(arr):
        # TODO implement a Heap
        return heapq.heapify(arr)
