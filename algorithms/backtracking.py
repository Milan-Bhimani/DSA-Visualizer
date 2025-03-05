def n_queens(n):
    """Solve N-Queens problem and return visualization steps."""
    board = [[0] * n for _ in range(n)]
    steps = []
    
    def is_safe(row, col):
        # Check row on left side
        for j in range(col):
            if board[row][j] == 1:
                return False
        
        # Check upper diagonal on left side
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        
        # Check lower diagonal on left side
        for i, j in zip(range(row, n, 1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False
        
        return True
    
    def solve(col):
        if col >= n:
            steps.append({
                'board': [row[:] for row in board],
                'explanation': f'Solution found! All {n} queens are placed.',
                'current': None
            })
            return True
        
        for row in range(n):
            if is_safe(row, col):
                board[row][col] = 1
                steps.append({
                    'board': [row[:] for row in board],
                    'explanation': f'Trying queen at position ({row}, {col})',
                    'current': {'row': row, 'col': col}
                })
                
                if solve(col + 1):
                    return True
                
                board[row][col] = 0
                steps.append({
                    'board': [row[:] for row in board],
                    'explanation': f'Backtracking from position ({row}, {col})',
                    'current': {'row': row, 'col': col}
                })
        
        return False
    
    solve(0)
    return steps

def solve_sudoku(board):
    if not isinstance(board, list) or len(board) != 9 or not all(len(row) == 9 for row in board):
        return {"error": "Invalid board format. Expected 9x9 grid."}
    
    # Create a deep copy to avoid modifying the input
    board = [row[:] for row in board]
    steps = []
    
    def is_valid(num, pos, board):
        # Check row
        for x in range(len(board[0])):
            if board[pos[0]][x] == num and pos[1] != x:
                return False
        
        # Check column
        for x in range(len(board)):
            if board[x][pos[1]] == num and pos[0] != x:
                return False
        
        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if board[i][j] == num and (i, j) != pos:
                    return False
        
        return True
    
    def find_empty(board):
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    return (i, j)
        return None
    
    # Initial validation
    steps.append({
        "board": [row[:] for row in board],
        "explanation": "Initial board state",
        "current": None
    })
    
    # Check if initial board is valid
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                temp = board[i][j]
                board[i][j] = 0
                if not is_valid(temp, (i, j), board):
                    steps.append({
                        "board": [row[:] for row in board],
                        "explanation": f"Invalid initial board: Conflict at position ({i+1}, {j+1}) with value {temp}",
                        "current": {"row": i, "col": j},
                        "success": False
                    })
                    return {"steps": steps, "solution": None}
                board[i][j] = temp
    
    def solve():
        find = find_empty(board)
        if not find:
            steps.append({
                "board": [row[:] for row in board],
                "explanation": "Solution found!",
                "current": None,
                "success": True
            })
            return True
        
        row, col = find
        for num in range(1, 10):
            if is_valid(num, (row, col), board):
                board[row][col] = num
                steps.append({
                    "board": [row[:] for row in board],
                    "explanation": f"Trying {num} at position ({row+1}, {col+1})",
                    "current": {"row": row, "col": col},
                    "value": num
                })
                
                if solve():
                    return True
                
                board[row][col] = 0
                steps.append({
                    "board": [row[:] for row in board],
                    "explanation": f"Backtracking: {num} doesn't work at ({row+1}, {col+1})",
                    "current": {"row": row, "col": col},
                    "value": 0
                })
        
        return False
    
    if solve():
        return {"steps": steps, "solution": board}
    else:
        steps.append({
            "board": [row[:] for row in board],
            "explanation": "No solution exists for this puzzle",
            "current": None,
            "success": False
        })
        return {"steps": steps, "solution": None}

def subset_sum(numbers, target):
    """Find subset with given sum and return visualization steps."""
    steps = []
    current_subset = []
    
    def solve(index, current_sum):
        if current_sum == target:
            steps.append({
                'numbers': numbers,
                'current_subset': current_subset[:],
                'current_sum': current_sum,
                'explanation': f'Found solution! Sum = {target}'
            })
            return True
        
        if index >= len(numbers) or current_sum > target:
            return False
        
        # Include current number
        current_subset.append(index)
        steps.append({
            'numbers': numbers,
            'current_subset': current_subset[:],
            'current_sum': current_sum + numbers[index],
            'explanation': f'Including {numbers[index]}'
        })
        
        if solve(index + 1, current_sum + numbers[index]):
            return True
        
        # Exclude current number
        current_subset.pop()
        steps.append({
            'numbers': numbers,
            'current_subset': current_subset[:],
            'current_sum': current_sum,
            'explanation': f'Excluding {numbers[index]}'
        })
        
        return solve(index + 1, current_sum)
    
    steps.append({
        'numbers': numbers,
        'current_subset': [],
        'current_sum': 0,
        'explanation': 'Starting search'
    })
    solve(0, 0)
    return steps

def permutations(array):
    """Generate all permutations and return visualization steps."""
    steps = []
    result = []
    
    def solve(arr, temp_perm):
        if len(arr) == 0:
            result.append(temp_perm[:])
            steps.append({
                'current_permutation': temp_perm[:],
                'all_permutations': [p[:] for p in result],
                'explanation': f'Found permutation: {temp_perm}'
            })
            return
        
        for i in range(len(arr)):
            current = arr[i]
            remaining = arr[:i] + arr[i+1:]
            temp_perm.append(current)
            
            steps.append({
                'current_permutation': temp_perm[:],
                'all_permutations': [p[:] for p in result],
                'explanation': f'Adding {current} to current permutation'
            })
            
            solve(remaining, temp_perm)
            temp_perm.pop()
            
            steps.append({
                'current_permutation': temp_perm[:],
                'all_permutations': [p[:] for p in result],
                'explanation': f'Removing {current}, backtracking'
            })
    
    steps.append({
        'current_permutation': [],
        'all_permutations': [],
        'explanation': 'Starting permutation generation'
    })
    solve(array, [])
    return steps

def graph_coloring(graph, num_colors):
    """Color graph vertices and return visualization steps."""
    steps = []
    colors = {}
    vertices = list(graph.keys())
    
    def is_safe(vertex, color):
        for neighbor in graph[vertex]:
            if neighbor in colors and colors[neighbor] == color:
                return False
        return True
    
    def solve(vertex_index):
        if vertex_index == len(vertices):
            steps.append({
                'graph': graph,
                'colors': dict(colors),
                'explanation': 'Solution found!'
            })
            return True
        
        vertex = vertices[vertex_index]
        for color in range(num_colors):
            if is_safe(vertex, color):
                colors[vertex] = color
                steps.append({
                    'graph': graph,
                    'colors': dict(colors),
                    'explanation': f'Coloring vertex {vertex} with color {color}'
                })
                
                if solve(vertex_index + 1):
                    return True
                
                colors[vertex] = None
                steps.append({
                    'graph': graph,
                    'colors': dict(colors),
                    'explanation': f'Backtracking from vertex {vertex}'
                })
        
        return False
    
    steps.append({
        'graph': graph,
        'colors': {},
        'explanation': 'Starting graph coloring'
    })
    solve(0)
    return steps 