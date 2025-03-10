import heapq
import time
from collections import deque


def solve_maze_graph(graph, start, end, algorithm="BFS"):
    """
    Solve a maze represented as a graph using the specified algorithm.
    Supported algorithms: BFS, DIJKSTRA, A*.
    Each cell is a tuple (r, c); each edge has weight 1.

    Returns:
      (path, elapsed_time)
      where path is a list of cell coordinates representing the solution path,
      and elapsed_time is the total time (in seconds) taken to solve the maze.
      Returns (None, elapsed_time) if no path exists.
    """
    algo = algorithm.upper()
    start_time = time.perf_counter()

    if algo == "A*":
        # A* search using Manhattan distance as heuristic.
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        g_score = {node: float('inf') for node in graph}
        f_score = {node: float('inf') for node in graph}
        came_from = {}
        g_score[start] = 0
        f_score[start] = heuristic(start, end)
        open_set = [(f_score[start], start)]

        while open_set:
            current_f, current = heapq.heappop(open_set)
            if current == end:
                break
            # Skip outdated entries.
            if current_f > f_score[current]:
                continue
            for neighbor in graph[current]:
                tentative_g = g_score[current] + 1  # each edge cost is 1
                if tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        if f_score[end] == float('inf'):
            elapsed_time = time.perf_counter() - start_time
            return None, elapsed_time
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = came_from.get(cur)
        path.reverse()
        elapsed_time = time.perf_counter() - start_time
        return path, elapsed_time

    elif algo == "BFS":
        queue = deque([start])
        came_from = {start: None}
        while queue:
            current = queue.popleft()
            if current == end:
                break
            for neighbor in graph[current]:
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    queue.append(neighbor)
        if end not in came_from:
            elapsed_time = time.perf_counter() - start_time
            return None, elapsed_time
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        elapsed_time = time.perf_counter() - start_time
        return path, elapsed_time

    elif algo == "DIJKSTRA":
        dist = {node: float('inf') for node in graph}
        dist[start] = 0
        came_from = {}
        heap = [(0, start)]
        while heap:
            d, current = heapq.heappop(heap)
            if current == end:
                break
            if d > dist[current]:
                continue
            for neighbor in graph[current]:
                alt = d + 1  # each edge cost is 1
                if alt < dist[neighbor]:
                    dist[neighbor] = alt
                    came_from[neighbor] = current
                    heapq.heappush(heap, (alt, neighbor))
        if dist[end] == float('inf'):
            elapsed_time = time.perf_counter() - start_time
            return None, elapsed_time
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = came_from.get(cur)
        path.reverse()
        elapsed_time = time.perf_counter() - start_time
        return path, elapsed_time

    else:
        raise NotImplementedError(f"Algorithm {algorithm} is not implemented.")


def solve_maze(maze):
    """
    Solve the cell-based maze using BFS.
    Maze is a 2D list with 0 for passages and 1 for walls.

    Returns:
      (path, elapsed_time)
      where path is a list of (row, col) tuples representing the solution,
      and elapsed_time is the solving time in seconds.
    """
    if not maze:
        return None, 0
    start_time = time.perf_counter()
    rows = len(maze)
    cols = len(maze[0])
    start = (0, 0)
    end = (rows - 1, cols - 1)
    queue = deque([start])
    came_from = {start: None}
    while queue:
        current = queue.popleft()
        if current == end:
            break
        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (r + dr, c + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if maze[r + dr][c + dc] == 0 and neighbor not in came_from:
                    came_from[neighbor] = current
                    queue.append(neighbor)
    if end not in came_from:
        elapsed_time = time.perf_counter() - start_time
        return None, elapsed_time
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    elapsed_time = time.perf_counter() - start_time
    return path, elapsed_time
