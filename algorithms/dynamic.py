def fibonacci(n):
    """Fibonacci sequence using dynamic programming"""
    if n <= 0:
        return []
    
    # Initialize sequence with first two numbers
    sequence = [0] * (n + 1)
    if n >= 1:
        sequence[1] = 1
    
    steps = []
    # Add initial state
    steps.append({
        "sequence": sequence[:n + 1],
        "current_index": 1,
        "n": n,
        "calculation": "F(1) = 1"
    })
    
    # Calculate remaining numbers
    for i in range(2, n + 1):
        sequence[i] = sequence[i-1] + sequence[i-2]
        steps.append({
            "sequence": sequence[:n + 1],
            "current_index": i,
            "n": n,
            "calculation": f"F({i}) = F({i-1}) + F({i-2}) = {sequence[i-1]} + {sequence[i-2]} = {sequence[i]}"
        })
    
    return steps

def longest_common_subsequence(str1, str2):
    """Longest Common Subsequence using dynamic programming"""
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    steps = []
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            steps.append({
                "dp": [row.copy() for row in dp],
                "i": i-1,
                "j": j-1,
                "match": str1[i-1] == str2[j-1] if i > 0 and j > 0 else False
            })
    
    # Reconstruct the sequence
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            lcs.append(str1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
        
        steps.append({
            "dp": [row.copy() for row in dp],
            "i": i-1 if i > 0 else None,
            "j": j-1 if j > 0 else None,
            "lcs": lcs.copy()
        })
    
    return steps

def knapsack(values, weights, capacity):
    """0/1 Knapsack Problem using dynamic programming"""
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    steps = []
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    values[i-1] + dp[i-1][w-weights[i-1]],
                    dp[i-1][w]
                )
            else:
                dp[i][w] = dp[i-1][w]
            
            steps.append({
                "dp": [row.copy() for row in dp],
                "item": i-1,
                "capacity": w,
                "included": dp[i][w] != dp[i-1][w]
            })
    
    # Reconstruct the solution
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i-1)
            w -= weights[i-1]
            steps.append({
                "dp": [row.copy() for row in dp],
                "selected": selected.copy(),
                "remaining_capacity": w
            })
    
    return steps

def edit_distance(str1, str2):
    """Edit Distance (Levenshtein Distance) using dynamic programming"""
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    steps = []
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    steps.append({
        "dp": [row.copy() for row in dp],
        "operation": "initialize"
    })
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
                steps.append({
                    "dp": [row.copy() for row in dp],
                    "i": i-1,
                    "j": j-1,
                    "operation": "match"
                })
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
                steps.append({
                    "dp": [row.copy() for row in dp],
                    "i": i-1,
                    "j": j-1,
                    "operation": "edit"
                })
    
    return steps

def matrix_chain_multiplication(dimensions):
    """Matrix Chain Multiplication using dynamic programming"""
    n = len(dimensions) - 1
    dp = [[0] * n for _ in range(n)]
    parenthesis = [[0] * n for _ in range(n)]
    steps = []
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            for k in range(i, j):
                cost = (dp[i][k] + dp[k+1][j] + 
                       dimensions[i] * dimensions[k+1] * dimensions[j+1])
                
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    parenthesis[i][j] = k
                
                steps.append({
                    "dp": [row.copy() for row in dp],
                    "parenthesis": [row.copy() for row in parenthesis],
                    "i": i,
                    "j": j,
                    "k": k,
                    "cost": cost
                })
    
    return steps

def longest_increasing_subsequence(arr):
    """Longest Increasing Subsequence using dynamic programming"""
    n = len(arr)
    dp = [1] * n
    prev = [-1] * n
    steps = []
    
    for i in range(1, n):
        for j in range(i):
            if arr[i] > arr[j] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j
                steps.append({
                    "dp": dp.copy(),
                    "prev": prev.copy(),
                    "i": i,
                    "j": j,
                    "current_sequence": []
                })
    
    # Reconstruct the sequence
    max_length = max(dp)
    last_index = dp.index(max_length)
    sequence = []
    while last_index != -1:
        sequence.append(arr[last_index])
        last_index = prev[last_index]
        steps.append({
            "dp": dp.copy(),
            "prev": prev.copy(),
            "sequence": sequence.copy()
        })
    
    return steps

def rod_cutting(prices, length):
    """Rod Cutting Problem using dynamic programming"""
    dp = [0] * (length + 1)
    cuts = [0] * (length + 1)
    steps = []
    
    for i in range(1, length + 1):
        max_val = float('-inf')
        best_cut = 0
        
        for j in range(i):
            current = prices[j] + dp[i - j - 1]
            if current > max_val:
                max_val = current
                best_cut = j + 1
        
        dp[i] = max_val
        cuts[i] = best_cut
        steps.append({
            "length": i,
            "dp": dp.copy(),
            "cuts": cuts.copy(),
            "current_cut": best_cut
        })
    
    return steps

def palindrome_partitioning(s):
    """Palindrome Partitioning using dynamic programming"""
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    cuts = [[0] * n for _ in range(n)]
    steps = []
    
    # All substrings of length 1 are palindromes
    for i in range(n):
        dp[i][i] = True
    
    # Check for substrings of length 2 and more
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if length == 2:
                dp[i][j] = s[i] == s[j]
            else:
                dp[i][j] = s[i] == s[j] and dp[i+1][j-1]
            
            steps.append({
                "dp": [row.copy() for row in dp],
                "i": i,
                "j": j,
                "is_palindrome": dp[i][j],
                "substring": s[i:j+1]
            })
    
    return steps

def word_break(s, wordDict):
    """Word Break Problem using dynamic programming"""
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    steps = []
    
    for i in range(1, n + 1):
        for word in wordDict:
            if i >= len(word) and s[i-len(word):i] == word:
                dp[i] = dp[i] or dp[i-len(word)]
                if dp[i]:
                    steps.append({
                        "dp": dp.copy(),
                        "position": i,
                        "word": word,
                        "substring": s[:i]
                    })
    
    return steps

def egg_dropping(eggs, floors):
    """Egg Dropping Puzzle using dynamic programming"""
    dp = [[0] * (floors + 1) for _ in range(eggs + 1)]
    steps = []
    
    # Base cases
    for i in range(1, eggs + 1):
        dp[i][1] = 1
    for j in range(1, floors + 1):
        dp[1][j] = j
    
    # Fill rest of the table
    for i in range(2, eggs + 1):
        for j in range(2, floors + 1):
            dp[i][j] = float('inf')
            for x in range(1, j + 1):
                res = 1 + max(dp[i-1][x-1], dp[i][j-x])
                dp[i][j] = min(dp[i][j], res)
            
            steps.append({
                "dp": [row.copy() for row in dp],
                "eggs": i,
                "floor": j,
                "attempts": dp[i][j]
            })
    
    return steps

def optimal_bst(keys, freq):
    """Optimal Binary Search Tree using dynamic programming"""
    n = len(keys)
    dp = [[0] * n for _ in range(n)]
    root = [[0] * n for _ in range(n)]
    steps = []
    
    # Initialize for length 1
    for i in range(n):
        dp[i][i] = freq[i]
        root[i][i] = i
    
    # Fill for other lengths
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            dp[i][j] = float('inf')
            sum_freq = sum(freq[i:j+1])
            
            for r in range(i, j + 1):
                left = dp[i][r-1] if r > i else 0
                right = dp[r+1][j] if r < j else 0
                cost = left + right + sum_freq
                
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    root[i][j] = r
            
            steps.append({
                "dp": [row.copy() for row in dp],
                "root": [row.copy() for row in root],
                "i": i,
                "j": j,
                "optimal_cost": dp[i][j]
            })
    
    return steps

def boolean_parenthesization(symbols, operators):
    """Boolean Parenthesization Problem using dynamic programming"""
    n = len(symbols)
    T = [[0] * n for _ in range(n)]  # True count
    F = [[0] * n for _ in range(n)]  # False count
    steps = []
    
    # Initialize diagonal
    for i in range(n):
        T[i][i] = 1 if symbols[i] == 'T' else 0
        F[i][i] = 1 if symbols[i] == 'F' else 0
    
    # Fill table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            for k in range(i, j):
                total_ik = T[i][k] + F[i][k]
                total_kj = T[k+1][j] + F[k+1][j]
                
                if operators[k] == '&':
                    T[i][j] += T[i][k] * T[k+1][j]
                    F[i][j] += (total_ik * total_kj - T[i][k] * T[k+1][j])
                elif operators[k] == '|':
                    F[i][j] += F[i][k] * F[k+1][j]
                    T[i][j] += (total_ik * total_kj - F[i][k] * F[k+1][j])
                elif operators[k] == '^':
                    T[i][j] += (F[i][k] * T[k+1][j] + T[i][k] * F[k+1][j])
                    F[i][j] += (T[i][k] * T[k+1][j] + F[i][k] * F[k+1][j])
            
            steps.append({
                "T": [row.copy() for row in T],
                "F": [row.copy() for row in F],
                "i": i,
                "j": j,
                "true_ways": T[i][j],
                "false_ways": F[i][j]
            })
    
    return steps 