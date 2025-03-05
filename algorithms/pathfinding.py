from collections import deque

def bfs(grid, start, end):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start[0], start[1])])
    visited = set([(start[0], start[1])])
    parent = {(start[0], start[1]): None}
    
    while queue:
        current = queue.popleft()
        if current == end:
            break
            
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            new_x, new_y = current[0] + dx, current[1] + dy
            if (0 <= new_x < rows and 0 <= new_y < cols and 
                grid[new_x][new_y] != 1 and 
                (new_x, new_y) not in visited):
                queue.append((new_x, new_y))
                visited.add((new_x, new_y))
                parent[(new_x, new_y)] = current
    
    path = []
    current = end
    while current:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    
    return path, list(visited)

def tree_bfs(tree_data, start, end):
    # Parse tree data
    values = tree_data.split(',')
    values = [int(x) if x != 'null' else None for x in values]
    
    # Build adjacency list
    adj = {}
    for i, val in enumerate(values):
        if val is not None:
            adj[val] = []
            # Left child
            left = 2 * i + 1
            if left < len(values) and values[left] is not None:
                adj[val].append(values[left])
                if values[left] not in adj:
                    adj[values[left]] = []
                adj[values[left]].append(val)
            
            # Right child
            right = 2 * i + 2
            if right < len(values) and values[right] is not None:
                adj[val].append(values[right])
                if values[right] not in adj:
                    adj[values[right]] = []
                adj[values[right]].append(val)
    
    # BFS
    queue = deque([start])
    visited = []
    parent = {start: None}
    
    while queue:
        current = queue.popleft()
        visited.append(current)
        
        if current == end:
            break
            
        for neighbor in adj.get(current, []):
            if neighbor not in parent:
                queue.append(neighbor)
                parent[neighbor] = current
    
    # Reconstruct path
    path = []
    current = end
    while current:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    
    return visited, path

def graph_bfs(nodes, edges, start, end):
    # Build adjacency list
    adj = {node: [] for node in nodes}
    for edge in edges:
        u, v = edge.split('-')
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS
    queue = deque([start])
    visited = []
    parent = {start: None}
    
    while queue:
        current = queue.popleft()
        visited.append(current)
        
        if current == end:
            break
            
        for neighbor in adj[current]:
            if neighbor not in parent:
                queue.append(neighbor)
                parent[neighbor] = current
    
    # Reconstruct path
    path = []
    current = end
    while current:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    
    return visited, path