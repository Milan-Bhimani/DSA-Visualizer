from flask import Flask, render_template, request, jsonify
from algorithms.sorting import (
    bubble_sort, insertion_sort, selection_sort,
    merge_sort, quick_sort, heap_sort
)
from algorithms.searching import (
    linear_search, binary_search, jump_search,
    exponential_search, interpolation_search
)
from algorithms.dynamic import (
    fibonacci, longest_common_subsequence, knapsack,
    edit_distance, matrix_chain_multiplication,
    longest_increasing_subsequence, rod_cutting,
    palindrome_partitioning, word_break, egg_dropping,
    optimal_bst, boolean_parenthesization
)
from algorithms.greedy import (
    activity_selection, fractional_knapsack, huffman_coding,
    coin_change_greedy, job_scheduling, minimum_platforms
)
from algorithms.backtracking import (
    n_queens,
    solve_sudoku,
    subset_sum,
    permutations,
    graph_coloring
)
from algorithms.divide_conquer import (
    binary_search_dc, merge_sort_dc, quick_sort_dc,
    closest_pair, karatsuba, strassen_matrix_multiply
)
from collections import defaultdict
import json

app = Flask(__name__)

# Algorithm descriptions
SORTING_ALGORITHMS = {
    'bubble': {
        'name': 'Bubble Sort',
        'description': 'A simple sorting algorithm that repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order.',
        'time': 'O(n²)',
        'space': 'O(1)'
    },
    'insertion': {
        'name': 'Insertion Sort',
        'description': 'Builds the final sorted array one item at a time by repeatedly inserting a new element into the sorted portion of the array.',
        'time': 'O(n²)',
        'space': 'O(1)'
    },
    'selection': {
        'name': 'Selection Sort',
        'description': 'Divides the input list into a sorted and an unsorted region, and repeatedly selects the smallest element from the unsorted region to add to the sorted region.',
        'time': 'O(n²)',
        'space': 'O(1)'
    },
    'merge': {
        'name': 'Merge Sort',
        'description': 'A divide-and-conquer algorithm that recursively breaks down a list into smaller sublists until each sublist consists of a single element, then merges those sublists to produce a sorted list.',
        'time': 'O(n log n)',
        'space': 'O(n)'
    },
    'quick': {
        'name': 'Quick Sort',
        'description': 'A divide-and-conquer algorithm that works by selecting a pivot element and partitioning the array around it, such that smaller elements are on the left and larger elements are on the right.',
        'time': 'O(n log n)',
        'space': 'O(log n)'
    },
    'heap': {
        'name': 'Heap Sort',
        'description': 'A comparison-based sorting algorithm that uses a binary heap data structure. It divides the input into a sorted and an unsorted region, and iteratively shrinks the unsorted region by extracting the largest element.',
        'time': 'O(n log n)',
        'space': 'O(1)'
    }
}

SEARCHING_ALGORITHMS = {
    'linear': {
        'name': 'Linear Search',
        'description': 'Sequentially checks each element in a list until a match is found or the whole list has been searched.',
        'time': 'O(n)',
        'space': 'O(1)'
    },
    'binary': {
        'name': 'Binary Search',
        'description': 'A divide-and-conquer search algorithm that finds the position of a target value within a sorted array by repeatedly dividing the search space in half.',
        'time': 'O(log n)',
        'space': 'O(1)'
    },
    'jump': {
        'name': 'Jump Search',
        'description': 'Works on sorted arrays by skipping a fixed number of elements and then performing a linear search.',
        'time': 'O(√n)',
        'space': 'O(1)'
    },
    'exponential': {
        'name': 'Exponential Search',
        'description': 'Works on sorted arrays by finding a range where the element might be present and then performing a binary search.',
        'time': 'O(log n)',
        'space': 'O(1)'
    },
    'interpolation': {
        'name': 'Interpolation Search',
        'description': 'An improvement over binary search that works on uniformly distributed sorted arrays by making a position estimate based on the value being searched.',
        'time': 'O(log log n)',
        'space': 'O(1)'
    }
}

DP_ALGORITHMS = {
    'fibonacci': {
        'name': 'Fibonacci Sequence',
        'description': 'Calculates the nth Fibonacci number using dynamic programming to avoid redundant calculations.',
        'time': 'O(n)',
        'space': 'O(n)'
    },
    'lcs': {
        'name': 'Longest Common Subsequence',
        'description': 'Finds the longest subsequence present in both strings in the same relative order.',
        'time': 'O(mn)',
        'space': 'O(mn)'
    },
    'knapsack': {
        'name': '0/1 Knapsack',
        'description': 'Solves the problem of choosing items with given weights and values to maximize total value while keeping total weight under a limit.',
        'time': 'O(nW)',
        'space': 'O(nW)'
    },
    'edit_distance': {
        'name': 'Edit Distance',
        'description': 'Calculates the minimum number of operations required to transform one string into another.',
        'time': 'O(mn)',
        'space': 'O(mn)'
    },
    'matrix_chain': {
        'name': 'Matrix Chain Multiplication',
        'description': 'Determines the most efficient way to multiply a sequence of matrices.',
        'time': 'O(n³)',
        'space': 'O(n²)'
    },
    'lis': {
        'name': 'Longest Increasing Subsequence',
        'description': 'Finds the length of the longest subsequence of a given sequence such that all elements of the subsequence are sorted in ascending order.',
        'time': 'O(n²)',
        'space': 'O(n)'
    },
    'rod-cutting': {
        'name': 'Rod Cutting',
        'description': 'Finds the best way to cut a rod of given length to maximize profit.',
        'time': 'O(n²)',
        'space': 'O(n)'
    },
    'palindrome-partition': {
        'name': 'Palindrome Partitioning',
        'description': 'Finds the minimum number of cuts needed to partition a string such that every part is a palindrome.',
        'time': 'O(n³)',
        'space': 'O(n²)'
    },
    'word-break': {
        'name': 'Word Break',
        'description': 'Determines if a string can be segmented into space-separated sequence of dictionary words.',
        'time': 'O(n²)',
        'space': 'O(n)'
    },
    'egg-dropping': {
        'name': 'Egg Dropping Puzzle',
        'description': 'Finds minimum number of trials needed to find the critical floor in a building.',
        'time': 'O(n*k²)',
        'space': 'O(n*k)'
    },
    'optimal-bst': {
        'name': 'Optimal Binary Search Tree',
        'description': 'Constructs an optimal binary search tree for given keys with their frequencies.',
        'time': 'O(n³)',
        'space': 'O(n²)'
    },
    'boolean-parenthesis': {
        'name': 'Boolean Parenthesization',
        'description': 'Counts number of ways to parenthesize boolean expression to make it evaluate to true.',
        'time': 'O(n³)',
        'space': 'O(n²)'
    }
}

GRAPH_ALGORITHMS = {
    'dfs': {
        'name': 'Depth-First Search',
        'description': 'A graph traversal algorithm that explores as far as possible along each branch before backtracking.',
        'time': 'O(V + E)',
        'space': 'O(V)'
    },
    'bfs': {
        'name': 'Breadth-First Search',
        'description': 'A graph traversal algorithm that explores all vertices at the present depth before moving on to vertices at the next depth level.',
        'time': 'O(V + E)',
        'space': 'O(V)'
    },
    'dijkstra': {
        'name': "Dijkstra's Algorithm",
        'description': 'Finds the shortest paths between nodes in a weighted graph.',
        'time': 'O((V + E) log V)',
        'space': 'O(V)'
    },
    'prim': {
        'name': "Prim's Algorithm",
        'description': 'Finds a minimum spanning tree for a weighted undirected graph.',
        'time': 'O(E log V)',
        'space': 'O(V)'
    },
    'kruskal': {
        'name': "Kruskal's Algorithm",
        'description': 'Finds a minimum spanning tree for a weighted undirected graph.',
        'time': 'O(E log E)',
        'space': 'O(V)'
    },
    'bellman-ford': {
        'name': 'Bellman-Ford Algorithm',
        'description': 'Finds the shortest paths from a source vertex to all other vertices in a weighted graph, even with negative edge weights.',
        'time': 'O(VE)',
        'space': 'O(V)'
    },
    'floyd-warshall': {
        'name': 'Floyd-Warshall Algorithm',
        'description': 'Finds shortest paths between all pairs of vertices in a weighted graph.',
        'time': 'O(V³)',
        'space': 'O(V²)'
    }
}

GREEDY_ALGORITHMS = {
    'activity': {
        'name': 'Activity Selection',
        'description': 'Selects the maximum number of activities that can be performed by a single person, assuming that a person can only work on a single activity at a time.',
        'time': 'O(n log n)',
        'space': 'O(1)'
    },
    'fractional-knapsack': {
        'name': 'Fractional Knapsack',
        'description': 'Maximizes the value of items in a knapsack where items can be broken into smaller pieces.',
        'time': 'O(n log n)',
        'space': 'O(1)'
    },
    'huffman': {
        'name': 'Huffman Coding',
        'description': 'Generates optimal prefix codes for data compression.',
        'time': 'O(n log n)',
        'space': 'O(n)'
    },
    'coin-change': {
        'name': 'Coin Change (Greedy)',
        'description': 'Finds minimum number of coins that make a given value (may not always give optimal solution).',
        'time': 'O(n)',
        'space': 'O(1)'
    },
    'job-scheduling': {
        'name': 'Job Scheduling',
        'description': 'Schedules jobs with deadlines to maximize profit.',
        'time': 'O(n²)',
        'space': 'O(n)'
    },
    'platforms': {
        'name': 'Minimum Platforms',
        'description': 'Finds minimum number of platforms required for a railway station.',
        'time': 'O(n log n)',
        'space': 'O(1)'
    }
}

BACKTRACKING_ALGORITHMS = {
    'n-queens': {
        'name': 'N-Queens Problem',
        'description': 'Place N queens on an NxN chessboard such that no two queens threaten each other.',
        'time_complexity': 'O(N!)',
        'space_complexity': 'O(N)',
        'function': n_queens
    },
    'sudoku': {
        'name': 'Sudoku Solver',
        'description': 'Solve a 9x9 Sudoku puzzle using backtracking.',
        'time_complexity': 'O(9^(N*N))',
        'space_complexity': 'O(N*N)',
        'function': solve_sudoku
    },
    'subset-sum': {
        'name': 'Subset Sum',
        'description': 'Find a subset of numbers that sum to a target value.',
        'time_complexity': 'O(2^N)',
        'space_complexity': 'O(N)',
        'function': subset_sum
    },
    'permutations': {
        'name': 'Permutations',
        'description': 'Generate all permutations of an array.',
        'time_complexity': 'O(N!)',
        'space_complexity': 'O(N!)',
        'function': permutations
    },
    'graph-coloring': {
        'name': 'Graph Coloring',
        'description': 'Color vertices of a graph such that no two adjacent vertices share the same color.',
        'time_complexity': 'O(M^V)',
        'space_complexity': 'O(V)',
        'function': graph_coloring
    }
}

DIVIDE_CONQUER_ALGORITHMS = {
    'binary-search': {
        'name': 'Binary Search',
        'description': 'Search a sorted array by repeatedly dividing the search interval in half.',
        'time': 'O(log N)',
        'space': 'O(log N)'
    },
    'merge-sort': {
        'name': 'Merge Sort',
        'description': 'Sort an array by dividing it into two halves, sorting them, and then merging the sorted halves.',
        'time': 'O(N log N)',
        'space': 'O(N)'
    },
    'quick-sort': {
        'name': 'Quick Sort',
        'description': 'Sort an array by selecting a pivot element and partitioning the array around it.',
        'time': 'O(N log N)',
        'space': 'O(log N)'
    },
    'closest-pair': {
        'name': 'Closest Pair of Points',
        'description': 'Find the closest pair of points in a set of points in 2D space.',
        'time': 'O(N log N)',
        'space': 'O(N)'
    },
    'karatsuba': {
        'name': 'Karatsuba Multiplication',
        'description': 'Multiply two large numbers using divide and conquer approach.',
        'time': 'O(N^1.585)',
        'space': 'O(N)'
    },
    'strassen': {
        'name': "Strassen's Matrix Multiplication",
        'description': 'Multiply two matrices using divide and conquer approach.',
        'time': 'O(N^2.807)',
        'space': 'O(N²)'
    }
}

@app.route('/')
def index():
    return render_template('index.html')

# Sorting Routes
@app.route('/sorting/<algorithm>')
def sorting_page(algorithm):
    if algorithm not in SORTING_ALGORITHMS:
        return 'Algorithm not found', 404
    
    return render_template('sorting.html',
        algorithm_id=f'sorting/{algorithm}',
        algorithm_name=SORTING_ALGORITHMS[algorithm]['name'],
        algorithm_description=SORTING_ALGORITHMS[algorithm]['description'],
        time_complexity=SORTING_ALGORITHMS[algorithm]['time'],
        space_complexity=SORTING_ALGORITHMS[algorithm]['space']
    )

@app.route('/api/sorting/<algorithm>', methods=['POST'])
def api_sorting(algorithm):
    data = request.json
    if algorithm == 'bubble':
        return jsonify({"steps": bubble_sort(data['array'])})
    elif algorithm == 'insertion':
        return jsonify({"steps": insertion_sort(data['array'])})
    elif algorithm == 'selection':
        return jsonify({"steps": selection_sort(data['array'])})
    elif algorithm == 'merge':
        return jsonify({"steps": merge_sort(data['array'])})
    elif algorithm == 'quick':
        return jsonify({"steps": quick_sort(data['array'])})
    elif algorithm == 'heap':
        return jsonify({"steps": heap_sort(data['array'])})
    return 'Algorithm not found', 404

# Searching Routes
@app.route('/searching/<algorithm>')
def searching_page(algorithm):
    if algorithm not in SEARCHING_ALGORITHMS:
        return 'Algorithm not found', 404
    
    return render_template('searching.html',
        algorithm_id=f'searching/{algorithm}',
        algorithm_name=SEARCHING_ALGORITHMS[algorithm]['name'],
        algorithm_description=SEARCHING_ALGORITHMS[algorithm]['description'],
        time_complexity=SEARCHING_ALGORITHMS[algorithm]['time'],
        space_complexity=SEARCHING_ALGORITHMS[algorithm]['space']
    )

@app.route('/api/searching/<algorithm>', methods=['POST'])
def api_searching(algorithm):
    data = request.json
    if algorithm == 'linear':
        return jsonify({"steps": linear_search(data['array'], data['target'])})
    elif algorithm == 'binary':
        return jsonify({"steps": binary_search(data['array'], data['target'])})
    elif algorithm == 'jump':
        return jsonify({"steps": jump_search(data['array'], data['target'])})
    elif algorithm == 'exponential':
        return jsonify({"steps": exponential_search(data['array'], data['target'])})
    elif algorithm == 'interpolation':
        return jsonify({"steps": interpolation_search(data['array'], data['target'])})
    return 'Algorithm not found', 404

# Dynamic Programming Routes
@app.route('/dp/<algorithm>')
def dp_page(algorithm):
    if algorithm not in DP_ALGORITHMS:
        return 'Algorithm not found', 404
    
    return render_template('dynamic.html',
        algorithm_id=f'dp/{algorithm}',
        algorithm_name=DP_ALGORITHMS[algorithm]['name'],
        algorithm_description=DP_ALGORITHMS[algorithm]['description'],
        time_complexity=DP_ALGORITHMS[algorithm]['time'],
        space_complexity=DP_ALGORITHMS[algorithm]['space']
    )

@app.route('/api/dp/<algorithm>', methods=['POST'])
def api_dp(algorithm):
    data = request.json
    if algorithm == 'fibonacci':
        return jsonify({"steps": fibonacci(data['n'])})
    elif algorithm == 'lcs':
        return jsonify({"steps": longest_common_subsequence(data['str1'], data['str2'])})
    elif algorithm == 'knapsack':
        return jsonify({"steps": knapsack(data['values'], data['weights'], data['capacity'])})
    elif algorithm == 'edit_distance':
        return jsonify({"steps": edit_distance(data['str1'], data['str2'])})
    elif algorithm == 'matrix_chain':
        return jsonify({"steps": matrix_chain_multiplication(data['dimensions'])})
    elif algorithm == 'lis':
        return jsonify({"steps": longest_increasing_subsequence(data['array'])})
    elif algorithm == 'rod-cutting':
        return jsonify({"steps": rod_cutting(data['prices'], data['length'])})
    elif algorithm == 'palindrome-partition':
        return jsonify({"steps": palindrome_partitioning(data['string'])})
    elif algorithm == 'word-break':
        return jsonify({"steps": word_break(data['string'], data['dictionary'])})
    elif algorithm == 'egg-dropping':
        return jsonify({"steps": egg_dropping(data['eggs'], data['floors'])})
    elif algorithm == 'optimal-bst':
        return jsonify({"steps": optimal_bst(data['keys'], data['frequencies'])})
    elif algorithm == 'boolean-parenthesis':
        return jsonify({"steps": boolean_parenthesization(data['symbols'], data['operators'])})
    return 'Algorithm not found', 404

# Graph Routes
@app.route('/graph/<algorithm>')
def graph_page(algorithm):
    if algorithm not in GRAPH_ALGORITHMS:
        return 'Algorithm not found', 404
    
    return render_template('graph.html',
        algorithm_id=f'graph/{algorithm}',
        algorithm_name=GRAPH_ALGORITHMS[algorithm]['name'],
        algorithm_description=GRAPH_ALGORITHMS[algorithm]['description'],
        time_complexity=GRAPH_ALGORITHMS[algorithm]['time'],
        space_complexity=GRAPH_ALGORITHMS[algorithm]['space']
    )

@app.route('/api/graph/<algorithm>', methods=['POST'])
def api_graph(algorithm):
    data = request.json
    nodes = data['nodes']
    edges = data['edges']
    
    # Convert edges list to adjacency list/matrix format
    graph = defaultdict(dict)
    for edge in edges:
        source, target = edge['source'], edge['target']
        weight = edge.get('weight', 1)
        graph[source][target] = weight
        if algorithm not in ['bellman-ford', 'floyd-warshall']:  # For undirected graphs
            graph[target][source] = weight
    
    if algorithm == 'dfs':
        return jsonify({"steps": dfs(graph, data['start'], data.get('end'))})
    elif algorithm == 'bfs':
        return jsonify({"steps": bfs(graph, data['start'], data.get('end'))})
    elif algorithm == 'dijkstra':
        return jsonify({"steps": dijkstra(graph, data['start'], data.get('end'))})
    elif algorithm == 'prim':
        return jsonify({"steps": prim(graph)})
    elif algorithm == 'kruskal':
        return jsonify({"steps": kruskal(graph)})
    elif algorithm == 'bellman-ford':
        return jsonify({"steps": bellman_ford(graph, data['start'])})
    elif algorithm == 'floyd-warshall':
        return jsonify({"steps": floyd_warshall(graph)})
    return 'Algorithm not found', 404

# Greedy Routes
@app.route('/greedy/<algorithm>')
def greedy_page(algorithm):
    if algorithm not in GREEDY_ALGORITHMS:
        return 'Algorithm not found', 404
    
    return render_template('greedy.html',
        algorithm_id=f'greedy/{algorithm}',
        algorithm_name=GREEDY_ALGORITHMS[algorithm]['name'],
        algorithm_description=GREEDY_ALGORITHMS[algorithm]['description'],
        time_complexity=GREEDY_ALGORITHMS[algorithm]['time'],
        space_complexity=GREEDY_ALGORITHMS[algorithm]['space']
    )

@app.route('/api/greedy/<algorithm>', methods=['POST'])
def api_greedy(algorithm):
    data = request.json
    if algorithm == 'activity':
        return jsonify({"steps": activity_selection(data['start_times'], data['finish_times'])})
    elif algorithm == 'fractional-knapsack':
        return jsonify({"steps": fractional_knapsack(data['values'], data['weights'], data['capacity'])})
    elif algorithm == 'huffman':
        return jsonify({"steps": huffman_coding(data['chars'], data['freqs'])})
    elif algorithm == 'coin-change':
        return jsonify({"steps": coin_change_greedy(data['coins'], data['amount'])})
    elif algorithm == 'job-scheduling':
        return jsonify({"steps": job_scheduling(data['jobs'], data['deadlines'], data['profits'])})
    elif algorithm == 'platforms':
        return jsonify({"steps": minimum_platforms(data['arrivals'], data['departures'])})
    return 'Algorithm not found', 404

# Backtracking Routes
@app.route('/backtracking/<algorithm>')
def backtracking_page(algorithm):
    if algorithm not in BACKTRACKING_ALGORITHMS:
        return "Algorithm not found", 404
    
    algo_info = BACKTRACKING_ALGORITHMS[algorithm]
    return render_template('backtracking.html',
                         algorithm_id=f'backtracking/{algorithm}',
                         algorithm_name=algo_info['name'],
                         algorithm_description=algo_info['description'],
                         time_complexity=algo_info['time_complexity'],
                         space_complexity=algo_info['space_complexity'])

@app.route('/api/backtracking/<algorithm>', methods=['POST'])
def backtracking_api(algorithm):
    try:
        if algorithm == 'n-queens':
            board_size = request.json.get('board_size', 8)
            steps = n_queens(board_size)
            return jsonify({'steps': steps})
        elif algorithm == 'sudoku':
            board = request.json.get('board')
            result = solve_sudoku(board)
            if 'error' in result:
                return jsonify({'error': result['error']}), 400
            return jsonify({'steps': result['steps']})
        elif algorithm == 'subset-sum':
            numbers = request.json.get('numbers')
            target = request.json.get('target')
            steps = subset_sum(numbers, target)
            return jsonify({'steps': steps})
        elif algorithm == 'permutations':
            array = request.json.get('array')
            steps = permutations(array)
            return jsonify({'steps': steps})
        elif algorithm == 'graph-coloring':
            graph = request.json.get('graph')
            colors = request.json.get('colors')
            steps = graph_coloring(graph, colors)
            return jsonify({'steps': steps})
        else:
            return jsonify({'error': 'Algorithm not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Divide and Conquer Routes
@app.route('/divide-conquer/<algorithm>')
def divide_conquer_page(algorithm):
    if algorithm not in DIVIDE_CONQUER_ALGORITHMS:
        return 'Algorithm not found', 404
    
    return render_template('divide_conquer.html',
        algorithm_id=f'divide-conquer/{algorithm}',
        algorithm_name=DIVIDE_CONQUER_ALGORITHMS[algorithm]['name'],
        algorithm_description=DIVIDE_CONQUER_ALGORITHMS[algorithm]['description'],
        time_complexity=DIVIDE_CONQUER_ALGORITHMS[algorithm]['time'],
        space_complexity=DIVIDE_CONQUER_ALGORITHMS[algorithm]['space']
    )

@app.route('/api/divide-conquer/<algorithm>', methods=['POST'])
def api_divide_conquer(algorithm):
    data = request.json
    if algorithm == 'binary-search':
        arr = data['array']
        return jsonify({"steps": binary_search_dc(arr, data['target'], 0, len(arr) - 1)})
    elif algorithm == 'merge-sort':
        arr = data['array']
        return jsonify({"steps": merge_sort_dc(arr, 0, len(arr) - 1)})
    elif algorithm == 'quick-sort':
        arr = data['array']
        return jsonify({"steps": quick_sort_dc(arr, 0, len(arr) - 1)})
    elif algorithm == 'closest-pair':
        return jsonify({"steps": closest_pair(data['points'])})
    elif algorithm == 'karatsuba':
        return jsonify({"steps": karatsuba(data['x'], data['y'])})
    elif algorithm == 'strassen':
        return jsonify({"steps": strassen_matrix_multiply(data['matrix_a'], data['matrix_b'])})
    return 'Algorithm not found', 404

if __name__ == '__main__':
    app.run(debug=True)