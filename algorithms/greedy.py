def activity_selection(start_times, finish_times):
    """Activity Selection Problem using greedy approach"""
    if not start_times or not finish_times:
        raise ValueError("Start times and finish times cannot be empty")
    
    if len(start_times) != len(finish_times):
        raise ValueError("Number of start times must match number of finish times")
    
    n = len(start_times)
    activities = sorted(range(n), key=lambda k: finish_times[k])
    steps = []
    
    # Add initial state
    steps.append({
        "activities": activities,
        "selected": [],
        "current": None,
        "start_times": start_times,
        "finish_times": finish_times,
        "explanation": "Initial state: Activities sorted by finish time"
    })
    
    selected = [activities[0]]
    last_finish = finish_times[activities[0]]
    
    # First activity is always selected
    steps.append({
        "activities": activities,
        "selected": selected.copy(),
        "current": activities[0],
        "start_times": start_times,
        "finish_times": finish_times,
        "explanation": f"Select first activity {activities[0]} (Start: {start_times[activities[0]]}, Finish: {finish_times[activities[0]]})"
    })
    
    for i in range(1, n):
        current = activities[i]
        if start_times[current] >= last_finish:
            selected.append(current)
            last_finish = finish_times[current]
            steps.append({
                "activities": activities,
                "selected": selected.copy(),
                "current": current,
                "start_times": start_times,
                "finish_times": finish_times,
                "explanation": f"Select activity {current} (Start: {start_times[current]}, Finish: {finish_times[current]}) as it starts after {last_finish}"
            })
        else:
            steps.append({
                "activities": activities,
                "selected": selected.copy(),
                "current": current,
                "start_times": start_times,
                "finish_times": finish_times,
                "explanation": f"Skip activity {current} as it overlaps (Start: {start_times[current]}, Finish: {finish_times[current]}, Previous finish: {last_finish})"
            })
    
    # Add final state
    steps.append({
        "activities": activities,
        "selected": selected.copy(),
        "current": None,
        "start_times": start_times,
        "finish_times": finish_times,
        "explanation": f"Final selection: {len(selected)} activities selected: {', '.join(map(str, selected))}"
    })
    
    return steps

def fractional_knapsack(values, weights, capacity):
    """Fractional Knapsack Problem using greedy approach"""
    if not values or not weights:
        raise ValueError("Values and weights cannot be empty")
    
    if len(values) != len(weights):
        raise ValueError("Number of values must match number of weights")
    
    if capacity <= 0:
        raise ValueError("Capacity must be positive")
    
    n = len(values)
    items = list(range(n))
    ratios = [(i, values[i] / weights[i]) for i in items]
    ratios.sort(key=lambda x: x[1], reverse=True)
    sorted_items = [i for i, _ in ratios]
    
    steps = []
    selected = []
    total_value = 0
    remaining_capacity = capacity
    
    # Add initial state
    steps.append({
        "items": sorted_items,
        "selected": [],
        "current_item": None,
        "values": values,
        "weights": weights,
        "remaining_capacity": capacity,
        "total_value": 0,
        "explanation": "Initial state: Items sorted by value/weight ratio"
    })
    
    for item, ratio in ratios:
        if weights[item] <= remaining_capacity:
            # Take whole item
            selected.append([item, 1.0])
            total_value += values[item]
            remaining_capacity -= weights[item]
            steps.append({
                "items": sorted_items,
                "selected": selected.copy(),
                "current_item": item,
                "values": values,
                "weights": weights,
                "remaining_capacity": remaining_capacity,
                "total_value": total_value,
                "explanation": f"Take whole item {item} (Value: {values[item]}, Weight: {weights[item]}, Ratio: {ratio:.2f})"
            })
        else:
            # Take fraction of item
            fraction = remaining_capacity / weights[item]
            selected.append([item, fraction])
            total_value += values[item] * fraction
            remaining_capacity = 0
            steps.append({
                "items": sorted_items,
                "selected": selected.copy(),
                "current_item": item,
                "values": values,
                "weights": weights,
                "remaining_capacity": remaining_capacity,
                "total_value": total_value,
                "explanation": f"Take {fraction:.2f} fraction of item {item} (Value: {values[item]}, Weight: {weights[item]}, Ratio: {ratio:.2f})"
            })
            break
    
    # Add final state
    steps.append({
        "items": sorted_items,
        "selected": selected.copy(),
        "current_item": None,
        "values": values,
        "weights": weights,
        "remaining_capacity": remaining_capacity,
        "total_value": total_value,
        "explanation": f"Final value: {total_value:.2f}, Remaining capacity: {remaining_capacity}"
    })
    
    return steps

def huffman_coding(chars, freqs):
    """Huffman Coding for data compression"""
    from heapq import heappush, heappop
    
    if not chars or not freqs:
        raise ValueError("Characters and frequencies cannot be empty")
    
    if len(chars) != len(freqs):
        raise ValueError("Number of characters must match number of frequencies")
    
    steps = []
    heap = []
    nodes = {}  # To store all nodes for visualization
    node_id = 0  # To generate unique IDs for nodes
    
    # Calculate total width and initial x positions
    total_width = len(chars) * 100  # pixels
    x_spacing = total_width / (len(chars) + 1)
    y_spacing = 80  # pixels between levels
    current_x = x_spacing
    
    # Create leaf nodes and add to heap
    for char, freq in zip(chars, freqs):
        node = {
            'id': node_id,
            'char': char,
            'freq': freq,
            'left': None,
            'right': None,
            'is_leaf': True,
            'x': current_x,
            'y': 400  # Start at bottom
        }
        nodes[node_id] = node
        heappush(heap, (freq, node_id))
        node_id += 1
        current_x += x_spacing
        
        steps.append({
            "nodes": [nodes[nid].copy() for nid in sorted(nodes.keys())],
            "edges": [],
            "current": node['id'],
            "explanation": f"Add leaf node '{char}' with frequency {freq}"
        })
    
    current_level = len(chars)
    # Build the tree
    while len(heap) > 1:
        left_freq, left_id = heappop(heap)
        right_freq, right_id = heappop(heap)
        left_node = nodes[left_id]
        right_node = nodes[right_id]
        
        # Calculate position for new internal node
        new_x = (left_node['x'] + right_node['x']) / 2
        new_y = 400 - (y_spacing * (len(chars) - current_level + 1))
        
        # Create internal node
        internal_node = {
            'id': node_id,
            'char': None,
            'freq': left_freq + right_freq,
            'left': left_id,
            'right': right_id,
            'is_leaf': False,
            'x': new_x,
            'y': new_y
        }
        nodes[node_id] = internal_node
        heappush(heap, (internal_node['freq'], node_id))
        
        # Update edges for visualization
        edges = []
        for nid, node in nodes.items():
            if node['left'] is not None:
                edges.append({
                    'from': nid,
                    'to': node['left'],
                    'from_x': nodes[nid]['x'],
                    'from_y': nodes[nid]['y'],
                    'to_x': nodes[node['left']]['x'],
                    'to_y': nodes[node['left']]['y']
                })
            if node['right'] is not None:
                edges.append({
                    'from': nid,
                    'to': node['right'],
                    'from_x': nodes[nid]['x'],
                    'from_y': nodes[nid]['y'],
                    'to_x': nodes[node['right']]['x'],
                    'to_y': nodes[node['right']]['y']
                })
        
        steps.append({
            "nodes": [nodes[nid].copy() for nid in sorted(nodes.keys())],
            "edges": edges,
            "current": node_id,
            "explanation": f"Merge nodes with frequencies {left_freq} and {right_freq} into internal node with frequency {left_freq + right_freq}"
        })
        
        node_id += 1
        current_level -= 1
    
    # Add final state
    if nodes:
        edges = []
        for nid, node in nodes.items():
            if node['left'] is not None:
                edges.append({
                    'from': nid,
                    'to': node['left'],
                    'from_x': nodes[nid]['x'],
                    'from_y': nodes[nid]['y'],
                    'to_x': nodes[node['left']]['x'],
                    'to_y': nodes[node['left']]['y']
                })
            if node['right'] is not None:
                edges.append({
                    'from': nid,
                    'to': node['right'],
                    'from_x': nodes[nid]['x'],
                    'from_y': nodes[nid]['y'],
                    'to_x': nodes[node['right']]['x'],
                    'to_y': nodes[node['right']]['y']
                })
                
        steps.append({
            "nodes": [nodes[nid].copy() for nid in sorted(nodes.keys())],
            "edges": edges,
            "current": None,
            "explanation": "Final Huffman tree constructed"
        })
    
    return steps

def coin_change_greedy(coins, amount):
    """Coin Change Problem using greedy approach (not always optimal)"""
    if not coins:
        raise ValueError("Coins list cannot be empty")
    
    if amount < 0:
        raise ValueError("Amount cannot be negative")
    
    coins.sort(reverse=True)
    steps = []
    selected = []
    remaining = amount
    
    # Add initial state
    steps.append({
        "coins": coins,
        "selected": selected.copy(),
        "current_coin": None,
        "remaining": remaining,
        "explanation": "Initial state: Coins sorted in descending order"
    })
    
    for coin in coins:
        while remaining >= coin:
            selected.append(coin)
            remaining -= coin
            steps.append({
                "coins": coins,
                "selected": selected.copy(),
                "current_coin": coin,
                "remaining": remaining,
                "explanation": f"Take coin {coin}, remaining amount: {remaining}"
            })
        
        if remaining > 0:
            steps.append({
                "coins": coins,
                "selected": selected.copy(),
                "current_coin": coin,
                "remaining": remaining,
                "explanation": f"Skip coin {coin} as it's larger than remaining amount {remaining}"
            })
    
    # Add final state
    steps.append({
        "coins": coins,
        "selected": selected.copy(),
        "current_coin": None,
        "remaining": remaining,
        "explanation": f"Final state: Used {len(selected)} coins, remaining amount: {remaining}"
    })
    
    return steps

def job_scheduling(jobs, deadlines, profits):
    """Job Scheduling with Deadlines using greedy approach"""
    if not jobs or not deadlines or not profits:
        raise ValueError("Jobs, deadlines, and profits cannot be empty")
    
    if len(jobs) != len(deadlines) or len(jobs) != len(profits):
        raise ValueError("Number of jobs must match number of deadlines and profits")
    
    n = len(jobs)
    job_info = list(zip(jobs, deadlines, profits))
    job_info.sort(key=lambda x: x[2], reverse=True)  # Sort by profit
    
    max_deadline = max(deadlines)
    schedule = [-1] * max_deadline
    selected = []
    steps = []
    
    # Add initial state
    steps.append({
        "jobs": [x[0] for x in job_info],
        "deadlines": [x[1] for x in job_info],
        "profits": [x[2] for x in job_info],
        "selected": selected.copy(),
        "current_job": None,
        "explanation": "Initial state: Jobs sorted by profit in descending order"
    })
    
    for job, deadline, profit in job_info:
        # Find last available slot before deadline
        slot_found = False
        for j in range(min(max_deadline, deadline) - 1, -1, -1):
            if schedule[j] == -1:
                schedule[j] = job
                selected.append(job)
                slot_found = True
                steps.append({
                    "jobs": [x[0] for x in job_info],
                    "deadlines": [x[1] for x in job_info],
                    "profits": [x[2] for x in job_info],
                    "selected": selected.copy(),
                    "current_job": job,
                    "explanation": f"Schedule Job {job} (Profit: {profit}) at time slot {j+1}"
                })
                break
        
        if not slot_found:
            steps.append({
                "jobs": [x[0] for x in job_info],
                "deadlines": [x[1] for x in job_info],
                "profits": [x[2] for x in job_info],
                "selected": selected.copy(),
                "current_job": job,
                "explanation": f"Skip Job {job} (Profit: {profit}) as no valid time slot available"
            })
    
    # Add final state
    total_profit = sum(profit for job, _, profit in job_info if job in selected)
    steps.append({
        "jobs": [x[0] for x in job_info],
        "deadlines": [x[1] for x in job_info],
        "profits": [x[2] for x in job_info],
        "selected": selected.copy(),
        "current_job": None,
        "explanation": f"Final schedule: {len(selected)} jobs scheduled with total profit {total_profit}"
    })
    
    return steps

def minimum_platforms(arrivals, departures):
    """Minimum Platforms needed for railway station"""
    if not arrivals or not departures:
        raise ValueError("Arrivals and departures cannot be empty")
    
    if len(arrivals) != len(departures):
        raise ValueError("Number of arrivals must match number of departures")
    
    n = len(arrivals)
    events = []
    for i in range(n):
        events.append((arrivals[i], 1))  # Arrival is +1
        events.append((departures[i], -1))  # Departure is -1
    
    events.sort()
    platforms = 0
    max_platforms = 0
    steps = []
    
    # Add initial state
    steps.append({
        "time": events[0][0],
        "event_type": "start",
        "current_platforms": 0,
        "max_platforms": 0,
        "explanation": "Initial state: No trains at station"
    })
    
    for time, change in events:
        platforms += change
        max_platforms = max(max_platforms, platforms)
        
        steps.append({
            "time": time,
            "event_type": "arrival" if change == 1 else "departure",
            "current_platforms": platforms,
            "max_platforms": max_platforms,
            "explanation": f"{'Arrival' if change == 1 else 'Departure'} at time {time}, platforms needed: {platforms}"
        })
    
    # Add final state
    steps.append({
        "time": events[-1][0],
        "event_type": "end",
        "current_platforms": platforms,
        "max_platforms": max_platforms,
        "explanation": f"Final state: Maximum platforms needed: {max_platforms}"
    })
    
    return steps 