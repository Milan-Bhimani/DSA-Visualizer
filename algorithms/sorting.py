def bubble_sort(arr):
    n = len(arr)
    steps = []
    
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            steps.append({
                "array": arr.copy(),
                "comparing": [j, j + 1],
                "description": f"Comparing elements {arr[j]} and {arr[j + 1]} at positions {j} and {j + 1}"
            })
            
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
                steps.append({
                    "array": arr.copy(),
                    "comparing": [j, j + 1],
                    "description": f"Swapping {arr[j + 1]} and {arr[j]} since {arr[j + 1]} < {arr[j]}"
                })
        
        if not swapped:
            steps.append({
                "array": arr.copy(),
                "comparing": [],
                "description": "No swaps needed in this pass - array is sorted!"
            })
            break
        else:
            steps.append({
                "array": arr.copy(),
                "comparing": [],
                "sorted": list(range(n-i-1, n)),
                "description": f"Pass {i+1} complete. Last {i+1} element(s) are now sorted."
            })
    
    return steps

def insertion_sort(arr):
    n = len(arr)
    steps = []
    
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        steps.append({
            "array": arr.copy(),
            "comparing": [i],
            "key": key,
            "description": f"Selected {key} as the key element to insert into the sorted portion"
        })
        
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            steps.append({
                "array": arr.copy(),
                "comparing": [j, j + 1],
                "key": key,
                "description": f"Moving {arr[j]} one position ahead to make room for {key}"
            })
            j -= 1
            
        arr[j + 1] = key
        steps.append({
            "array": arr.copy(),
            "comparing": [j + 1],
            "key": key,
            "description": f"Inserted {key} at position {j + 1}"
        })
    
    return steps

def selection_sort(arr):
    n = len(arr)
    steps = []
    
    for i in range(n):
        min_idx = i
        steps.append({
            "array": arr.copy(),
            "comparing": [i],
            "min_idx": min_idx,
            "description": f"Starting new pass. Current minimum is {arr[min_idx]} at position {min_idx}"
        })
        
        for j in range(i + 1, n):
            steps.append({
                "array": arr.copy(),
                "comparing": [min_idx, j],
                "min_idx": min_idx,
                "description": f"Comparing current minimum {arr[min_idx]} with {arr[j]}"
            })
            if arr[j] < arr[min_idx]:
                min_idx = j
                steps.append({
                    "array": arr.copy(),
                    "comparing": [min_idx],
                    "min_idx": min_idx,
                    "description": f"Found new minimum: {arr[min_idx]} at position {min_idx}"
                })
                
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            steps.append({
                "array": arr.copy(),
                "comparing": [i, min_idx],
                "min_idx": min_idx,
                "description": f"Swapping {arr[min_idx]} with {arr[i]} to put minimum in correct position"
            })
        
        steps.append({
            "array": arr.copy(),
            "comparing": [],
            "sorted": list(range(i + 1)),
            "description": f"Position {i} now contains the correct element. First {i + 1} elements are sorted."
        })
    
    return steps

def merge_sort(arr):
    steps = []
    
    def merge(left, right, start_idx):
        result = []
        i = j = 0
        
        steps.append({
            "array": arr.copy(),
            "comparing": [],
            "subarray": (start_idx, start_idx + len(left) + len(right)),
            "description": f"Merging two subarrays: {left} and {right}"
        })
        
        while i < len(left) and j < len(right):
            steps.append({
                "array": arr.copy(),
                "comparing": [start_idx + i, start_idx + len(left) + j],
                "subarray": (start_idx, start_idx + len(left) + len(right)),
                "description": f"Comparing elements: {left[i]} and {right[j]}"
            })
            
            if left[i] <= right[j]:
                result.append(left[i])
                steps.append({
                    "array": arr.copy(),
                    "comparing": [start_idx + i],
                    "subarray": (start_idx, start_idx + len(left) + len(right)),
                    "description": f"Taking {left[i]} from left subarray"
                })
                i += 1
            else:
                result.append(right[j])
                steps.append({
                    "array": arr.copy(),
                    "comparing": [start_idx + len(left) + j],
                    "subarray": (start_idx, start_idx + len(left) + len(right)),
                    "description": f"Taking {right[j]} from right subarray"
                })
                j += 1
        
        while i < len(left):
            result.append(left[i])
            steps.append({
                "array": arr.copy(),
                "comparing": [start_idx + i],
                "subarray": (start_idx, start_idx + len(left) + len(right)),
                "description": f"Adding remaining element {left[i]} from left subarray"
            })
            i += 1
        
        while j < len(right):
            result.append(right[j])
            steps.append({
                "array": arr.copy(),
                "comparing": [start_idx + len(left) + j],
                "subarray": (start_idx, start_idx + len(left) + len(right)),
                "description": f"Adding remaining element {right[j]} from right subarray"
            })
            j += 1
        
        for k in range(len(result)):
            arr[start_idx + k] = result[k]
            steps.append({
                "array": arr.copy(),
                "comparing": [start_idx + k],
                "subarray": (start_idx, start_idx + len(left) + len(right)),
                "description": f"Placing {result[k]} in final position {start_idx + k}"
            })
        
        return result
    
    def sort(start_idx, end_idx):
        if end_idx - start_idx <= 1:
            return arr[start_idx:end_idx]
        
        mid = (start_idx + end_idx) // 2
        steps.append({
            "array": arr.copy(),
            "comparing": [start_idx, mid, end_idx],
            "subarray": (start_idx, end_idx),
            "description": f"Dividing array into two parts at position {mid}"
        })
        
        left = sort(start_idx, mid)
        right = sort(mid, end_idx)
        return merge(left, right, start_idx)
    
    sort(0, len(arr))
    steps.append({
        "array": arr.copy(),
        "comparing": [],
        "description": "Merge sort completed!"
    })
    return steps

def quick_sort(arr):
    steps = []
    
    def partition(low, high):
        pivot = arr[high]
        i = low - 1
        
        steps.append({
            "array": arr.copy(),
            "comparing": [high],
            "pivot": high,
            "subarray": (low, high + 1),
            "description": f"Selected {pivot} as pivot element"
        })
        
        for j in range(low, high):
            steps.append({
                "array": arr.copy(),
                "comparing": [j, high],
                "pivot": high,
                "subarray": (low, high + 1),
                "description": f"Comparing element {arr[j]} with pivot {pivot}"
            })
            
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                steps.append({
                    "array": arr.copy(),
                    "comparing": [i, j],
                    "pivot": high,
                    "subarray": (low, high + 1),
                    "description": f"Swapping {arr[j]} and {arr[i]} since {arr[j]} ≤ {pivot}"
                })
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        steps.append({
            "array": arr.copy(),
            "comparing": [i + 1, high],
            "pivot": i + 1,
            "subarray": (low, high + 1),
            "description": f"Placing pivot {pivot} in its final position {i + 1}"
        })
        
        return i + 1
    
    def sort(low, high):
        if low < high:
            steps.append({
                "array": arr.copy(),
                "comparing": [low, high],
                "subarray": (low, high + 1),
                "description": f"Partitioning subarray from index {low} to {high}"
            })
            
            pi = partition(low, high)
            
            steps.append({
                "array": arr.copy(),
                "comparing": [],
                "pivot": pi,
                "subarray": (low, high + 1),
                "description": f"Recursively sorting left partition (elements < {arr[pi]}) and right partition (elements > {arr[pi]})"
            })
            
            sort(low, pi - 1)
            sort(pi + 1, high)
    
    sort(0, len(arr) - 1)
    steps.append({
        "array": arr.copy(),
        "comparing": [],
        "description": "Quick sort completed!"
    })
    return steps

def heap_sort(arr):
    steps = []
    
    def heapify(n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        steps.append({
            "array": arr.copy(),
            "comparing": [i],
            "heap_size": n,
            "description": f"Heapifying subtree rooted at index {i}"
        })
        
        if left < n:
            steps.append({
                "array": arr.copy(),
                "comparing": [largest, left],
                "heap_size": n,
                "description": f"Comparing root {arr[largest]} with left child {arr[left]}"
            })
            if arr[left] > arr[largest]:
                largest = left
                steps.append({
                    "array": arr.copy(),
                    "comparing": [largest],
                    "heap_size": n,
                    "description": f"Left child {arr[largest]} is larger than root"
                })
        
        if right < n:
            steps.append({
                "array": arr.copy(),
                "comparing": [largest, right],
                "heap_size": n,
                "description": f"Comparing largest element {arr[largest]} with right child {arr[right]}"
            })
            if arr[right] > arr[largest]:
                largest = right
                steps.append({
                    "array": arr.copy(),
                    "comparing": [largest],
                    "heap_size": n,
                    "description": f"Right child {arr[largest]} is the largest element"
                })
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            steps.append({
                "array": arr.copy(),
                "comparing": [i, largest],
                "heap_size": n,
                "description": f"Swapping root {arr[i]} with largest child {arr[largest]}"
            })
            heapify(n, largest)
    
    n = len(arr)
    
    # Build max heap
    steps.append({
        "array": arr.copy(),
        "comparing": [],
        "description": "Building max heap from the array"
    })
    
    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)
    
    steps.append({
        "array": arr.copy(),
        "comparing": [],
        "description": "Max heap built. Starting to extract elements."
    })
    
    # Extract elements from heap
    for i in range(n - 1, 0, -1):
        steps.append({
            "array": arr.copy(),
            "comparing": [0, i],
            "heap_size": i,
            "description": f"Moving largest element {arr[0]} to the end at position {i}"
        })
        arr[0], arr[i] = arr[i], arr[0]
        steps.append({
            "array": arr.copy(),
            "comparing": [0, i],
            "heap_size": i,
            "sorted": list(range(i, n)),
            "description": f"Heapifying reduced heap of size {i}"
        })
        heapify(i, 0)
    
    steps.append({
        "array": arr.copy(),
        "comparing": [],
        "sorted": list(range(n)),
        "description": "Heap sort completed!"
    })
    return steps