// Sorting Visualization
async function visualizeSort() {
    const input = document.getElementById('sortInput').value;
    const array = input.split(',').map(Number);
    
    const response = await fetch('/api/bubble_sort', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ array })
    });
    const { steps } = await response.json();
    
    const visualization = document.getElementById('sortVisualization');
    visualization.innerHTML = '';
    
    // Create container for all steps
    const stepContainer = document.createElement('div');
    stepContainer.className = 'step-container';
    visualization.appendChild(stepContainer);

    // Add initial state
    const initialStep = document.createElement('div');
    initialStep.className = 'step-row';
    initialStep.innerHTML = `
        <div class="step-number">0</div>
        <div class="step-description">Initial array</div>
        <div class="number-row">
            ${array.map(num => `
                <div class="number-box">${num}</div>
            `).join('')}
        </div>
    `;
    stepContainer.appendChild(initialStep);
    await new Promise(resolve => setTimeout(resolve, 500));

    // Process each step
    for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        const stepElement = document.createElement('div');
        stepElement.className = 'step-row';
        
        // Create description based on comparison
        let description = '';
        if (step.comparing.length === 2) {
            const [first, second] = step.comparing;
            const firstNum = step.array[first];
            const secondNum = step.array[second];
            if (steps[i - 1] && steps[i - 1].array[first] !== firstNum) {
                description = `Swap ${steps[i - 1].array[first]} and ${steps[i - 1].array[second]} because ${steps[i - 1].array[first]} > ${steps[i - 1].array[second]}`;
            } else {
                description = `Compare ${firstNum} and ${secondNum}${firstNum <= secondNum ? ' (no swap needed)' : ''}`;
            }
        }

        stepElement.innerHTML = `
            <div class="step-number">${i + 1}</div>
            <div class="step-description">${description}</div>
            <div class="number-row">
                ${step.array.map((num, index) => `
                    <div class="number-box${step.comparing.includes(index) ? ' comparing' : ''}">${num}</div>
                `).join('')}
            </div>
        `;
        
        stepContainer.appendChild(stepElement);
        await new Promise(resolve => setTimeout(resolve, 800));
    }

    // Add final state
    const finalStep = document.createElement('div');
    finalStep.className = 'step-row';
    finalStep.innerHTML = `
        <div class="step-number">✓</div>
        <div class="step-description">Array sorted!</div>
        <div class="number-row">
            ${steps[steps.length - 1].array.map(num => `
                <div class="number-box sorted">${num}</div>
            `).join('')}
        </div>
    `;
    stepContainer.appendChild(finalStep);
}

// Pathfinding Visualization
const GRID_SIZE = 10;
let grid = Array(GRID_SIZE).fill().map(() => Array(GRID_SIZE).fill(0));
let start = [0, 0];
let end = [GRID_SIZE-1, GRID_SIZE-1];

function createGrid() {
    const gridDiv = document.getElementById('grid');
    gridDiv.style.gridTemplateColumns = `repeat(${GRID_SIZE}, 20px)`;
    
    for (let i = 0; i < GRID_SIZE; i++) {
        for (let j = 0; j < GRID_SIZE; j++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = i;
            cell.dataset.col = j;
            
            if (i === start[0] && j === start[1]) cell.classList.add('start');
            if (i === end[0] && j === end[1]) cell.classList.add('end');
            
            cell.addEventListener('click', toggleWall);
            gridDiv.appendChild(cell);
        }
    }
}

function toggleWall(event) {
    const row = parseInt(event.target.dataset.row);
    const col = parseInt(event.target.dataset.col);
    if ((row === start[0] && col === start[1]) || 
        (row === end[0] && col === end[1])) return;
        
    grid[row][col] = grid[row][col] === 0 ? 1 : 0;
    event.target.classList.toggle('wall');
}

async function visualizePath() {
    const response = await fetch('/api/bfs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ grid, start, end })
    });
    const { path, visited } = await response.json();
    
    // Animate visited nodes
    for (let [row, col] of visited) {
        if ((row === start[0] && col === start[1]) || 
            (row === end[0] && col === end[1])) continue;
            
        const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
        cell.classList.add('visited');
        await new Promise(resolve => setTimeout(resolve, 50));
    }
    
    // Animate final path
    for (let [row, col] of path) {
        if ((row === start[0] && col === start[1]) || 
            (row === end[0] && col === end[1])) continue;
            
        const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
        cell.classList.remove('visited');
        cell.classList.add('path');
        await new Promise(resolve => setTimeout(resolve, 100));
    }
}

// Visualization Type Selection
function selectType(type) {
    // Update buttons
    document.querySelectorAll('.type-button').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-type="${type}"]`).classList.add('active');

    // Hide all controls and views
    document.querySelectorAll('.structure-controls, .structure-view').forEach(el => {
        el.style.display = 'none';
    });

    // Show selected controls and view
    document.getElementById(`${type}Controls`).style.display = 'block';
    document.getElementById(type).style.display = 'flex';

    // Reset visualizations
    if (type === 'grid') resetGrid();
    else if (type === 'tree') resetTree();
    else if (type === 'graph') resetGraph();
}

// Tree Building
let treeData = null;
let selectedNode = null;

function addRootNode() {
    const value = parseInt(document.getElementById('treeNodeValue').value);
    if (isNaN(value)) return;

    treeData = {
        value: value,
        children: []
    };
    document.getElementById('treeNodeValue').value = '';
    createTree(treeData);
}

function showAddChildPopup(node, event) {
    selectedNode = node.data;
    const popup = document.getElementById('addChildPopup');
    popup.style.display = 'block';
    popup.style.left = `${event.pageX}px`;
    popup.style.top = `${event.pageY}px`;
}

function closeChildPopup() {
    document.getElementById('addChildPopup').style.display = 'none';
    document.getElementById('childNodeValue').value = '';
    selectedNode = null;
}

function addLeftChild() {
    if (!selectedNode) return;
    const value = parseInt(document.getElementById('childNodeValue').value);
    if (isNaN(value)) return;

    if (!selectedNode.children) selectedNode.children = [];
    selectedNode.children[0] = { value: value, children: [] };
    createTree(treeData);
    closeChildPopup();
}

function addRightChild() {
    if (!selectedNode) return;
    const value = parseInt(document.getElementById('childNodeValue').value);
    if (isNaN(value)) return;

    if (!selectedNode.children) selectedNode.children = [];
    selectedNode.children[1] = { value: value, children: [] };
    createTree(treeData);
    closeChildPopup();
}

// Update createTree function
function createTree(data) {
    const width = 800;
    const height = 500;
    const nodeRadius = 25;

    d3.select('#tree').html('');

    const svg = d3.select('#tree')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const g = svg.append('g')
        .attr('transform', `translate(${width/2},50)`);

    const treeLayout = d3.tree()
        .size([width - 100, height - 100]);

    const root = d3.hierarchy(data);
    const treeData = treeLayout(root);

    g.selectAll('.tree-link')
        .data(treeData.links())
        .enter()
        .append('path')
        .attr('class', 'tree-link')
        .attr('d', d3.linkVertical()
            .x(d => d.x)
            .y(d => d.y));

    const nodes = g.selectAll('.tree-node-group')
        .data(treeData.descendants())
        .enter()
        .append('g')
        .attr('class', 'tree-node-group')
        .attr('transform', d => `translate(${d.x},${d.y})`)
        .on('click', function(event, d) {
            showAddChildPopup(d, event);
        });

    nodes.append('circle')
        .attr('class', 'tree-node')
        .attr('r', nodeRadius)
        .attr('fill', 'white')
        .attr('stroke', '#3498db')
        .attr('stroke-width', 2);

    nodes.append('text')
        .attr('class', 'tree-label')
        .text(d => d.data.value);

    return treeData;
}

async function visualizeTreeBFS() {
    const treeInput = document.getElementById('treeInput').value;
    const startValue = parseInt(document.getElementById('treeStart').value);
    const endValue = parseInt(document.getElementById('treeEnd').value);

    const response = await fetch('/api/tree_bfs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            tree: treeInput,
            start: startValue,
            end: endValue
        })
    });
    const { visited, path } = await response.json();

    // Animate the BFS process
    const treeData = createTree(treeInput);
    const nodes = d3.selectAll('.tree-node');

    // Mark start and end nodes
    nodes.each(function(d) {
        if (d.data.value === startValue) d3.select(this).classed('start', true);
        if (d.data.value === endValue) d3.select(this).classed('end', true);
    });

    // Animate visited nodes
    for (const value of visited) {
        nodes.each(function(d) {
            if (d.data.value === value) {
                d3.select(this).classed('visited', true);
            }
        });
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    // Animate path
    for (const value of path) {
        nodes.each(function(d) {
            if (d.data.value === value) {
                d3.select(this).classed('visited', false).classed('path', true);
            }
        });
        await new Promise(resolve => setTimeout(resolve, 500));
    }
}

// Graph Building
let graphNodes = [];
let graphEdges = [];

function addVertex() {
    const value = document.getElementById('graphNodeValue').value;
    if (!value || graphNodes.includes(value)) return;

    graphNodes.push(value);
    document.getElementById('graphNodeValue').value = '';
    createGraph(graphNodes, graphEdges);
}

function connectNodes() {
    const source = document.getElementById('graphSource').value;
    const target = document.getElementById('graphTarget').value;
    
    if (!source || !target || !graphNodes.includes(source) || !graphNodes.includes(target)) {
        alert('Please enter valid source and target nodes');
        return;
    }

    const edge = `${source}-${target}`;
    const reverseEdge = `${target}-${source}`;
    
    if (!graphEdges.includes(edge) && !graphEdges.includes(reverseEdge)) {
        graphEdges.push(edge);
        createGraph(graphNodes, graphEdges);
        document.getElementById('graphSource').value = '';
        document.getElementById('graphTarget').value = '';
    }
}

function createGraph(nodes, edges) {
    const width = 800;
    const height = 500;
    const nodeRadius = 25;

    d3.select('#graph').html('');

    const svg = d3.select('#graph')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const simulation = d3.forceSimulation()
        .force('link', d3.forceLink().id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-500))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(nodeRadius * 2));

    const nodeData = nodes.map(id => ({ id }));
    const linkData = edges.map(edge => {
        const [source, target] = edge.split('-');
        return { source, target };
    });

    const links = svg.append('g')
        .selectAll('.graph-link')
        .data(linkData)
        .enter()
        .append('line')
        .attr('class', 'graph-link');

    const nodeGroups = svg.append('g')
        .selectAll('.graph-node-group')
        .data(nodeData)
        .enter()
        .append('g')
        .attr('class', 'graph-node-group');

    nodeGroups.append('circle')
        .attr('class', 'graph-node')
        .attr('r', nodeRadius);

    nodeGroups.append('text')
        .attr('class', 'graph-label')
        .text(d => d.id);

    simulation.nodes(nodeData).on('tick', () => {
        // Keep nodes within bounds
        nodeData.forEach(d => {
            d.x = Math.max(nodeRadius, Math.min(width - nodeRadius, d.x));
            d.y = Math.max(nodeRadius, Math.min(height - nodeRadius, d.y));
        });

        links
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        nodeGroups
            .attr('transform', d => `translate(${d.x},${d.y})`);
    });

    simulation.force('link').links(linkData);
}

async function visualizeGraphBFS() {
    const nodes = document.getElementById('graphNodes').value.split(',').map(n => n.trim());
    const edges = document.getElementById('graphEdges').value.split(',').map(e => e.trim());
    const start = document.getElementById('graphStart').value.trim();
    const end = document.getElementById('graphEnd').value.trim();

    createGraph(nodes, edges);

    const response = await fetch('/api/graph_bfs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, edges, start, end })
    });
    const { visited, path } = await response.json();

    const graphNodes = d3.selectAll('.graph-node');

    // Mark start and end nodes
    graphNodes.each(function(d) {
        if (d.id === start) d3.select(this).classed('start', true);
        if (d.id === end) d3.select(this).classed('end', true);
    });

    // Animate visited nodes
    for (const node of visited) {
        graphNodes.each(function(d) {
            if (d.id === node) {
                d3.select(this).classed('visited', true);
            }
        });
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    // Animate path
    for (const node of path) {
        graphNodes.each(function(d) {
            if (d.id === node) {
                d3.select(this).classed('visited', false).classed('path', true);
            }
        });
        await new Promise(resolve => setTimeout(resolve, 500));
    }
}

// Update reset functions
function resetTree() {
    treeData = null;
    selectedNode = null;
    d3.select('#tree').html('');
    document.getElementById('treeNodeValue').value = '';
    document.getElementById('treeStart').value = '';
    document.getElementById('treeEnd').value = '';
    document.getElementById('childNodeValue').value = '';
    document.getElementById('addChildPopup').style.display = 'none';
}

function resetGraph() {
    graphNodes = [];
    graphEdges = [];
    d3.select('#graph').html('');
    document.getElementById('graphNodeValue').value = '';
    document.getElementById('graphStart').value = '';
    document.getElementById('graphEnd').value = '';
    document.getElementById('graphSource').value = '';
    document.getElementById('graphTarget').value = '';
    document.getElementById('addVertexPopup').style.display = 'none';
}

// Initialize with grid view
window.onload = () => {
    selectType('grid');
    createGrid();
};