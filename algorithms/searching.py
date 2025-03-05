def linear_search(arr, target):
    steps = []
    for i in range(len(arr)):
        steps.append({
            "array": arr.copy(),
            "comparing": [i],
            "found": False
        })
        if arr[i] == target:
            steps[-1]["found"] = True
            return steps
    return steps

def binary_search(arr, target):
    arr = sorted(arr)
    steps = []
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        steps.append({
            "array": arr.copy(),
            "comparing": [mid],
            "left": left,
            "right": right,
            "found": False
        })
        
        if arr[mid] == target:
            steps[-1]["found"] = True
            return steps
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return steps

def jump_search(arr, target):
    n = len(arr)
    step = int(n ** 0.5)
    steps = []
    
    prev = 0
    while prev < n and arr[min(step, n) - 1] < target:
        steps.append({
            "array": arr.copy(),
            "comparing": [min(step, n) - 1],
            "prev": prev,
            "step": step,
            "found": False
        })
        prev = step
        step += int(n ** 0.5)
    
    while prev < n and arr[prev] < target:
        steps.append({
            "array": arr.copy(),
            "comparing": [prev],
            "prev": prev,
            "step": step,
            "found": False
        })
        prev += 1
    
    if prev < n and arr[prev] == target:
        steps.append({
            "array": arr.copy(),
            "comparing": [prev],
            "prev": prev,
            "step": step,
            "found": True
        })
        return steps
    
    return steps

def exponential_search(arr, target):
    steps = []
    if arr[0] == target:
        steps.append({
            "array": arr.copy(),
            "comparing": [0],
            "bound": 1,
            "found": True
        })
        return steps
    
    bound = 1
    while bound < len(arr) and arr[bound] < target:
        steps.append({
            "array": arr.copy(),
            "comparing": [bound],
            "bound": bound,
            "found": False
        })
        bound *= 2
    
    def binary_search_bounded(left, right):
        while left <= right:
            mid = (left + right) // 2
            steps.append({
                "array": arr.copy(),
                "comparing": [mid],
                "bound": bound,
                "left": left,
                "right": right,
                "found": False
            })
            
            if arr[mid] == target:
                steps[-1]["found"] = True
                return True
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return False
    
    binary_search_bounded(bound // 2, min(bound, len(arr) - 1))
    return steps

def interpolation_search(arr, target):
    steps = []
    low = 0
    high = len(arr) - 1
    
    while low <= high and target >= arr[low] and target <= arr[high]:
        if low == high:
            steps.append({
                "array": arr.copy(),
                "comparing": [low],
                "low": low,
                "high": high,
                "found": arr[low] == target
            })
            return steps
        
        pos = low + ((high - low) * (target - arr[low]) // 
                    (arr[high] - arr[low]))
        
        steps.append({
            "array": arr.copy(),
            "comparing": [pos],
            "low": low,
            "high": high,
            "found": False
        })
        
        if arr[pos] == target:
            steps[-1]["found"] = True
            return steps
        
        if arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1
    
    return steps 