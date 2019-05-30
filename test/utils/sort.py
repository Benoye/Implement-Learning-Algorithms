from utils.heap_sort import HeapSort
from utils.max_sort import MaxSort

T = [1, 5, 57, 2, 4, 40, 12]

ms = MaxSort()
print(ms.sort(T))

hs = HeapSort()
print(hs.sort(T))
