from collections import defaultdict, deque
import heapq

def dfs(graph, start, end=None):
    """Depth-First Search implementation"""
    visited = set()
    path = []
    steps = []
    
    def dfs_recursive(node):
        visited.add(node)
        path.append(node)
        steps.append({
            "visited": list(visited),
            "path": path.copy(),
            "current": node,
            "found": node == end if end else False
        })
        
        if end and node == end:
            return True
            
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs_recursive(neighbor):
                    return True
        
        if end is None:
            path.pop()
        
        return False
    
    dfs_recursive(start)
    return steps

def bfs(graph, start, end=None):
    """Breadth-First Search implementation"""
    visited = set([start])
    queue = deque([(start, [start])])
    steps = []
    
    while queue:
        vertex, path = queue.popleft()
        steps.append({
            "visited": list(visited),
            "path": path,
            "current": vertex,
            "found": vertex == end if end else False
        })
        
        if end and vertex == end:
            return steps
            
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return steps

def dijkstra(graph, start, end=None):
    """Dijkstra's shortest path algorithm"""
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    previous = {vertex: None for vertex in graph}
    visited = set()
    steps = []
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        if current_vertex in visited:
            continue
            
        visited.add(current_vertex)
        steps.append({
            "distances": distances.copy(),
            "visited": list(visited),
            "current": current_vertex,
            "found": current_vertex == end if end else False
        })
        
        if end and current_vertex == end:
            path = []
            current = end
            while current:
                path.append(current)
                current = previous[current]
            steps[-1]["path"] = path[::-1]
            return steps
            
        for neighbor, weight in graph[current_vertex].items():
            if neighbor in visited:
                continue
                
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))
    
    return steps

def prim(graph):
    """Prim's Minimum Spanning Tree algorithm"""
    if not graph:
        return []
        
    start_vertex = list(graph.keys())[0]
    visited = set([start_vertex])
    edges = []
    steps = []
    
    def find_min_edge():
        min_edge = (None, None, float('infinity'))
        for vertex in visited:
            for neighbor, weight in graph[vertex].items():
                if neighbor not in visited and weight < min_edge[2]:
                    min_edge = (vertex, neighbor, weight)
        return min_edge
    
    while len(visited) < len(graph):
        vertex1, vertex2, weight = find_min_edge()
        if vertex1 is None:
            break
            
        visited.add(vertex2)
        edges.append((vertex1, vertex2, weight))
        steps.append({
            "visited": list(visited),
            "edges": edges.copy(),
            "current_edge": (vertex1, vertex2)
        })
    
    return steps

def kruskal(graph):
    """Kruskal's Minimum Spanning Tree algorithm"""
    edges = []
    for vertex in graph:
        for neighbor, weight in graph[vertex].items():
            if (neighbor, vertex, weight) not in edges:
                edges.append((vertex, neighbor, weight))
    
    edges.sort(key=lambda x: x[2])
    parent = {vertex: vertex for vertex in graph}
    rank = {vertex: 0 for vertex in graph}
    mst = []
    steps = []
    
    def find(vertex):
        if parent[vertex] != vertex:
            parent[vertex] = find(parent[vertex])
        return parent[vertex]
    
    def union(vertex1, vertex2):
        root1 = find(vertex1)
        root2 = find(vertex2)
        
        if root1 != root2:
            if rank[root1] < rank[root2]:
                root1, root2 = root2, root1
            parent[root2] = root1
            if rank[root1] == rank[root2]:
                rank[root1] += 1
    
    for vertex1, vertex2, weight in edges:
        if find(vertex1) != find(vertex2):
            union(vertex1, vertex2)
            mst.append((vertex1, vertex2, weight))
            steps.append({
                "mst": mst.copy(),
                "current_edge": (vertex1, vertex2),
                "sets": {v: find(v) for v in graph}
            })
    
    return steps

def bellman_ford(graph, start):
    """Bellman-Ford shortest path algorithm"""
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    steps = []
    
    edges = []
    for vertex in graph:
        for neighbor, weight in graph[vertex].items():
            edges.append((vertex, neighbor, weight))
    
    for _ in range(len(graph) - 1):
        for vertex1, vertex2, weight in edges:
            if distances[vertex1] + weight < distances[vertex2]:
                distances[vertex2] = distances[vertex1] + weight
                steps.append({
                    "distances": distances.copy(),
                    "current_edge": (vertex1, vertex2),
                    "updated": vertex2
                })
    
    # Check for negative cycles
    for vertex1, vertex2, weight in edges:
        if distances[vertex1] + weight < distances[vertex2]:
            steps.append({
                "distances": distances.copy(),
                "negative_cycle": True,
                "cycle_edge": (vertex1, vertex2)
            })
            return steps
    
    return steps

def floyd_warshall(graph):
    """Floyd-Warshall all-pairs shortest path algorithm"""
    vertices = list(graph.keys())
    n = len(vertices)
    distances = {v: {u: float('infinity') for u in vertices} for v in vertices}
    
    # Initialize distances
    for v in vertices:
        distances[v][v] = 0
        for u, w in graph[v].items():
            distances[v][u] = w
    
    steps = []
    
    for k in vertices:
        for i in vertices:
            for j in vertices:
                if distances[i][k] + distances[k][j] < distances[i][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]
                    steps.append({
                        "distances": {v: dict(d) for v, d in distances.items()},
                        "intermediate": k,
                        "from": i,
                        "to": j
                    })
    
    return steps 