class MaxSort(object):
    """
    Complexity O(n^2)
    """
    @staticmethod
    def sort(arr):
        for i in range(len(arr)):
            sub_length = len(arr) - i

            tmp_max = 0
            idx_max = 0
            for j in range(sub_length):
                if arr[j] > tmp_max:
                    tmp_max = arr[j]
                    idx_max = j

            tmp = arr[sub_length - 1]
            arr[sub_length - 1] = tmp_max
            arr[idx_max] = tmp
        return arr
