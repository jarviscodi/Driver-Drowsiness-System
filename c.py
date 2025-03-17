import heapq


def dijkstra(adj_matrix, source):
    """Finds the shortest path distances from the source node to all other nodes using Dijkstra’s algorithm.

    Parameters:
    adj_matrix (list of list of int): Adjacency matrix representation of the graph.
                                      If there is no edge, use float('inf') as weight.
    source (int): Index of the source node.

    Returns:
    list: Shortest distances from the source node to all other nodes.
    """
    n = len(adj_matrix)  # Number of nodes
    distances = [float('inf')] * n  # Initialize distances with infinity
    distances[source] = 0  # Distance to the source is 0
    min_heap = [(0, source)]  # Min-heap storing (distance, node)
    visited = set()  # Track visited nodes

    while min_heap:
        current_dist, current_node = heapq.heappop(min_heap)  # Get node with smallest distance

        if current_node in visited:
            continue  # Skip if already processed

        visited.add(current_node)

        for neighbor in range(n):
            weight = adj_matrix[current_node][neighbor]
            if weight != float('inf') and neighbor not in visited:  # Check for a valid edge
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:  # Update distance if shorter path found
                    distances[neighbor] = new_dist
                    heapq.heappush(min_heap, (new_dist, neighbor))

    return distances
# Example adjacency matrix (Graph with 4 nodes)
# 0   1   2   3
graph = [
    [0,  1,  4,  float('inf')],
    [1,  0,  2,  5],
    [4,  2,  0,  1],
    [float('inf'), 5, 1, 0]
]

source = 0  # Starting node
print(dijkstra(graph, source))
