def binary_search_dc(arr, target, left, right):
    """Binary Search using divide and conquer"""
    steps = []
    
    if left > right:
        steps.append({
            "array": arr,
            "left": left,
            "right": right,
            "mid": None,
            "target_index": -1,
            "explanation": "Target not found"
        })
        return steps
    
    mid = (left + right) // 2
    steps.append({
        "array": arr,
        "left": left,
        "right": right,
        "mid": mid,
        "comparing": arr[mid],
        "explanation": f"Comparing target {target} with middle element {arr[mid]}"
    })
    
    if arr[mid] == target:
        steps.append({
            "array": arr,
            "left": left,
            "right": right,
            "mid": mid,
            "target_index": mid,
            "explanation": f"Target {target} found at index {mid}"
        })
    elif arr[mid] > target:
        steps.extend(binary_search_dc(arr, target, left, mid - 1))
    else:
        steps.extend(binary_search_dc(arr, target, mid + 1, right))
    
    return steps

def merge_sort_dc(arr, left, right):
    """Merge Sort using divide and conquer"""
    steps = []
    
    if left < right:
        mid = (left + right) // 2
        steps.append({
            "array": arr[:],
            "left": left,
            "right": right,
            "mid": mid,
            "dividing": True,
            "explanation": f"Dividing array at index {mid}"
        })
        
        steps.extend(merge_sort_dc(arr, left, mid))
        steps.extend(merge_sort_dc(arr, mid + 1, right))
        
        # Merge
        i = left
        j = mid + 1
        temp = []
        
        while i <= mid and j <= right:
            steps.append({
                "array": arr[:],
                "comparing": (i, j),
                "explanation": f"Comparing elements {arr[i]} and {arr[j]}"
            })
            
            if arr[i] <= arr[j]:
                temp.append(arr[i])
                i += 1
            else:
                temp.append(arr[j])
                j += 1
        
        while i <= mid:
            temp.append(arr[i])
            i += 1
        
        while j <= right:
            temp.append(arr[j])
            j += 1
        
        # Copy back
        for i in range(len(temp)):
            arr[left + i] = temp[i]
            steps.append({
                "array": arr[:],
                "merged": (left + i,),
                "explanation": f"Placing element {temp[i]} at index {left + i}"
            })
    
    return steps

def quick_sort_dc(arr, low, high):
    """Quick Sort using divide and conquer"""
    steps = []
    
    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        
        steps.append({
            "array": arr[:],
            "pivot_index": high,
            "pivot": pivot,
            "explanation": f"Choosing pivot element {pivot} at index {high}"
        })
        
        for j in range(low, high):
            steps.append({
                "array": arr[:],
                "comparing": (j, high),
                "explanation": f"Comparing element {arr[j]} with pivot {pivot}"
            })
            
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                steps.append({
                    "array": arr[:],
                    "swapped": (i, j),
                    "explanation": f"Swapping elements {arr[i]} and {arr[j]}"
                })
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        steps.append({
            "array": arr[:],
            "partition": i + 1,
            "explanation": f"Placing pivot element {pivot} at index {i + 1}"
        })
        
        return i + 1
    
    if low < high:
        pi = partition(arr, low, high)
        steps.extend(quick_sort_dc(arr, low, pi - 1))
        steps.extend(quick_sort_dc(arr, pi + 1, high))
    
    return steps

def closest_pair(points):
    """Closest Pair of Points using divide and conquer"""
    def distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    
    def strip_closest(strip, d, steps):
        min_dist = d
        closest_pair = None
        
        strip.sort(key=lambda x: x[1])
        
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and strip[j][1] - strip[i][1] < min_dist:
                dist = distance(strip[i], strip[j])
                steps.append({
                    "points": points,
                    "comparing": (strip[i], strip[j]),
                    "distance": dist,
                    "explanation": f"Comparing points {strip[i]} and {strip[j]} with distance {dist:.2f}"
                })
                
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (strip[i], strip[j])
                j += 1
        
        return min_dist, closest_pair, steps
    
    def closest_util(points_x, points_y):
        steps = []
        n = len(points_x)
        
        if n <= 3:
            min_dist = float('inf')
            closest_pair = None
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist = distance(points_x[i], points_x[j])
                    steps.append({
                        "points": points,
                        "comparing": (points_x[i], points_x[j]),
                        "distance": dist,
                        "explanation": f"Comparing points {points_x[i]} and {points_x[j]} with distance {dist:.2f}"
                    })
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (points_x[i], points_x[j])
            
            return min_dist, closest_pair, steps
        
        mid = n // 2
        mid_point = points_x[mid]
        
        points_yl = [p for p in points_y if p[0] <= mid_point[0]]
        points_yr = [p for p in points_y if p[0] > mid_point[0]]
        
        steps.append({
            "points": points,
            "mid_point": mid_point,
            "dividing": True,
            "explanation": f"Dividing points at {mid_point}"
        })
        
        d1, pair1, steps1 = closest_util(points_x[:mid], points_yl)
        d2, pair2, steps2 = closest_util(points_x[mid:], points_yr)
        steps.extend(steps1)
        steps.extend(steps2)
        
        d = min(d1, d2)
        closest_pair = pair1 if d1 < d2 else pair2
        
        strip = [p for p in points_y if abs(p[0] - mid_point[0]) < d]
        strip_dist, strip_pair, strip_steps = strip_closest(strip, d, [])
        steps.extend(strip_steps)
        
        if strip_dist < d:
            d = strip_dist
            closest_pair = strip_pair
        
        return d, closest_pair, steps
    
    points_x = sorted(points, key=lambda p: p[0])
    points_y = sorted(points, key=lambda p: p[1])
    
    _, _, steps = closest_util(points_x, points_y)
    return steps

def karatsuba(x, y):
    """Karatsuba multiplication algorithm using divide and conquer"""
    steps = []
    
    def karatsuba_rec(x, y, depth=0):
        if x < 10 or y < 10:
            steps.append({
                "x": x,
                "y": y,
                "result": x * y,
                "depth": depth,
                "base_case": True,
                "explanation": f"Base case multiplication: {x} * {y} = {x * y}"
            })
            return x * y
        
        n = max(len(str(x)), len(str(y)))
        m = n // 2
        
        # Split the numbers
        power = 10 ** m
        a = x // power
        b = x % power
        c = y // power
        d = y % power
        
        steps.append({
            "x": x,
            "y": y,
            "split": {
                "a": a, "b": b,
                "c": c, "d": d
            },
            "depth": depth,
            "explanation": f"Splitting numbers: {x} -> ({a}, {b}), {y} -> ({c}, {d})"
        })
        
        # Recursive steps
        ac = karatsuba_rec(a, c, depth + 1)
        bd = karatsuba_rec(b, d, depth + 1)
        ad_plus_bc = karatsuba_rec(a + b, c + d, depth + 1) - ac - bd
        
        # Combine results
        result = ac * (10 ** (2 * m)) + ad_plus_bc * (10 ** m) + bd
        
        steps.append({
            "x": x,
            "y": y,
            "ac": ac,
            "bd": bd,
            "ad_plus_bc": ad_plus_bc,
            "result": result,
            "depth": depth,
            "explanation": f"Combining results: {ac} * 10^{2 * m} + {ad_plus_bc} * 10^{m} + {bd} = {result}"
        })
        
        return result
    
    karatsuba_rec(x, y)
    return steps

def strassen_matrix_multiply(A, B):
    """Strassen's matrix multiplication algorithm using divide and conquer."""
    steps = []
    n = len(A)
    
    def add_matrices(X, Y):
        return [[X[i][j] + Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]
    
    def subtract_matrices(X, Y):
        return [[X[i][j] - Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]
    
    def split_matrix(M):
        """Split matrix into quarters"""
        n = len(M)
        mid = n // 2
        top_left = [[M[i][j] for j in range(mid)] for i in range(mid)]
        top_right = [[M[i][j] for j in range(mid, n)] for i in range(mid)]
        bot_left = [[M[i][j] for j in range(mid)] for i in range(mid, n)]
        bot_right = [[M[i][j] for j in range(mid, n)] for i in range(mid, n)]
        return top_left, top_right, bot_left, bot_right
    
    def strassen_recursive(A, B, depth=0):
        if len(A) <= 2:  # Base case for 2x2 matrices
            steps.append({
                "step": "Base case multiplication",
                "depth": depth,
                "a": A,
                "b": B,
                "explanation": "Direct multiplication for 2x2 matrices",
                "result": [[sum(A[i][k] * B[k][j] for k in range(len(B)))
                           for j in range(len(B[0]))]
                          for i in range(len(A))]
            })
            return [[sum(A[i][k] * B[k][j] for k in range(len(B)))
                    for j in range(len(B[0]))]
                   for i in range(len(A))]
        
        # Split matrices into quarters
        a11, a12, a21, a22 = split_matrix(A)
        b11, b12, b21, b22 = split_matrix(B)
        
        steps.append({
            "step": "Split matrices",
            "depth": depth,
            "a_quarters": {"a11": a11, "a12": a12, "a21": a21, "a22": a22},
            "b_quarters": {"b11": b11, "b12": b12, "b21": b21, "b22": b22},
            "explanation": "Dividing matrices into quarters for Strassen's algorithm"
        })
        
        # Compute the seven products
        steps.append({
            "step": "Computing products",
            "depth": depth,
            "explanation": "Calculate the seven products required by Strassen's algorithm"
        })
        
        # P1 = A11 * (B12 - B22)
        s1 = subtract_matrices(b12, b22)
        p1 = strassen_recursive(a11, s1, depth + 1)
        steps.append({
            "step": "P1 calculation",
            "depth": depth,
            "formula": "P1 = A11 * (B12 - B22)",
            "intermediate": {"B12-B22": s1},
            "result": p1
        })
        
        # P2 = (A11 + A12) * B22
        s2 = add_matrices(a11, a12)
        p2 = strassen_recursive(s2, b22, depth + 1)
        steps.append({
            "step": "P2 calculation",
            "depth": depth,
            "formula": "P2 = (A11 + A12) * B22",
            "intermediate": {"A11+A12": s2},
            "result": p2
        })
        
        # P3 = (A21 + A22) * B11
        s3 = add_matrices(a21, a22)
        p3 = strassen_recursive(s3, b11, depth + 1)
        steps.append({
            "step": "P3 calculation",
            "depth": depth,
            "formula": "P3 = (A21 + A22) * B11",
            "intermediate": {"A21+A22": s3},
            "result": p3
        })
        
        # P4 = A22 * (B21 - B11)
        s4 = subtract_matrices(b21, b11)
        p4 = strassen_recursive(a22, s4, depth + 1)
        steps.append({
            "step": "P4 calculation",
            "depth": depth,
            "formula": "P4 = A22 * (B21 - B11)",
            "intermediate": {"B21-B11": s4},
            "result": p4
        })
        
        # P5 = (A11 + A22) * (B11 + B22)
        s5 = add_matrices(a11, a22)
        s6 = add_matrices(b11, b22)
        p5 = strassen_recursive(s5, s6, depth + 1)
        steps.append({
            "step": "P5 calculation",
            "depth": depth,
            "formula": "P5 = (A11 + A22) * (B11 + B22)",
            "intermediate": {"A11+A22": s5, "B11+B22": s6},
            "result": p5
        })
        
        # P6 = (A12 - A22) * (B21 + B22)
        s7 = subtract_matrices(a12, a22)
        s8 = add_matrices(b21, b22)
        p6 = strassen_recursive(s7, s8, depth + 1)
        steps.append({
            "step": "P6 calculation",
            "depth": depth,
            "formula": "P6 = (A12 - A22) * (B21 + B22)",
            "intermediate": {"A12-A22": s7, "B21+B22": s8},
            "result": p6
        })
        
        # P7 = (A11 - A21) * (B11 + B12)
        s9 = subtract_matrices(a11, a21)
        s10 = add_matrices(b11, b12)
        p7 = strassen_recursive(s9, s10, depth + 1)
        steps.append({
            "step": "P7 calculation",
            "depth": depth,
            "formula": "P7 = (A11 - A21) * (B11 + B12)",
            "intermediate": {"A11-A21": s9, "B11+B12": s10},
            "result": p7
        })
        
        # Calculate quarters of result matrix
        c11 = add_matrices(subtract_matrices(add_matrices(p5, p4), p2), p6)
        c12 = add_matrices(p1, p2)
        c21 = add_matrices(p3, p4)
        c22 = subtract_matrices(subtract_matrices(add_matrices(p5, p1), p3), p7)
        
        steps.append({
            "step": "Combining results",
            "depth": depth,
            "formulas": {
                "C11": "P5 + P4 - P2 + P6",
                "C12": "P1 + P2",
                "C21": "P3 + P4",
                "C22": "P5 + P1 - P3 - P7"
            },
            "results": {
                "c11": c11, "c12": c12,
                "c21": c21, "c22": c22
            },
            "explanation": "Combining the products to form the final result quarters"
        })
        
        # Combine quarters into result matrix
        result = [[0 for _ in range(n)] for _ in range(n)]
        mid = n // 2
        for i in range(mid):
            for j in range(mid):
                result[i][j] = c11[i][j]
                result[i][j + mid] = c12[i][j]
                result[i + mid][j] = c21[i][j]
                result[i + mid][j + mid] = c22[i][j]
        
        steps.append({
            "step": "Final result",
            "depth": depth,
            "result": result,
            "explanation": "Combined final result matrix"
        })
        
        return result
    
    result = strassen_recursive(A, B)
    return steps