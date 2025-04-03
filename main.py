import random
import sys
import os
from collections import deque
from PyQt5 import QtWidgets, QtGui, QtCore
import solve_path as solve  # Your module remains unchanged
import vision_test as vision
import time

# =========================
# Maze Generation Functions
# =========================

def generate_maze(rows, cols):
    """
    Generates a maze (cell grid) using a DFS/Prim hybrid.
    Each cell is 0 (passage) or 1 (wall).
    """
    maze = [[1 for _ in range(cols)] for _ in range(rows)]

    def get_neighbors(r, c):
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 1:
                neighbors.append((nr, nc))
        return neighbors

    maze[0][0] = 0
    walls = [(dr, dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
             if 0 <= 0 + dr < rows and 0 <= 0 + dc < cols]
    while walls:
        wr, wc = random.choice(walls)
        walls.remove((wr, wc))
        if maze[wr][wc] == 1:
            neighbors = [(wr + dr, wc + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                         if 0 <= wr + dr < rows and 0 <= wc + dc < cols and maze[wr + dr][wc + dc] == 0]
            if len(neighbors) == 1:
                maze[wr][wc] = 0
                walls.extend(get_neighbors(wr, wc))
    # Ensure end cell is open.
    end_r, end_c = rows - 1, cols - 1
    if maze[end_r][end_c] == 1:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = end_r + dr, end_c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                maze[end_r][end_c] = 0
                break
        else:
            if end_r - 1 >= 0:
                maze[end_r - 1][end_c] = 0
                maze[end_r][end_c] = 0
            elif end_c - 1 >= 0:
                maze[end_r][end_c - 1] = 0
                maze[end_r][end_c] = 0
    return maze


def generate_maze_graph(rows, cols):
    """
    Generates a maze as a graph using Prim's algorithm.
    Each cell (r,c) is a node; an edge between two nodes means a passage.
    Returns a dictionary mapping each cell to a set of connected neighbors.
    """
    nodes = [(r, c) for r in range(rows) for c in range(cols)]
    graph = {node: set() for node in nodes}
    visited = set()
    frontier = []
    start = (0, 0)
    visited.add(start)
    for neighbor in get_neighbors(start, rows, cols):
        frontier.append((start, neighbor))
    while frontier:
        cell, neighbor = random.choice(frontier)
        frontier.remove((cell, neighbor))
        if neighbor not in visited:
            graph[cell].add(neighbor)
            graph[neighbor].add(cell)
            visited.add(neighbor)
            for n in get_neighbors(neighbor, rows, cols):
                if n not in visited:
                    frontier.append((neighbor, n))
    print(graph)
    return graph


def get_neighbors(cell, rows, cols):
    """Helper for generate_maze_graph; same as before."""
    r, c = cell
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors


# ---------------------------
# BFS Animation Generator for Graph Maze (Modified)
# ---------------------------
def bfs_animate_graph(graph, start, end):
    """
    A generator that performs BFS on the maze graph and yields a triple:
      (visited, in_queue, result)
    - visited: set of nodes that have been processed (popped)
    - in_queue: set of nodes currently in the queue (pushed but not yet processed)
    - result: during traversal, yields the last node added; at the end, yields the full solution path as a list.
    """
    queue = deque([start])
    visited = {start}
    in_queue = {start}
    prev = {start: None}
    yield visited.copy(), in_queue.copy(), None  # initial state
    while queue:
        current = queue.popleft()
        if current in in_queue:
            in_queue.remove(current)
        visited.add(current)
        if current == end:
            break
        for neighbor in graph[current]:
            if neighbor not in visited and neighbor not in in_queue:
                in_queue.add(neighbor)
                prev[neighbor] = current
                queue.append(neighbor)
                yield visited.copy(), in_queue.copy(), neighbor
    if end not in prev:
        yield visited.copy(), in_queue.copy(), None
    else:
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        yield visited.copy(), in_queue.copy(), path


# ---------------------------
# BFS Animation Generator for Matrix (Cell Maze)
# ---------------------------
def bfs_animate_matrix(maze, start, end):
    """
    A generator that performs BFS on a cell maze (2D grid) and yields:
      (visited, in_queue, result)
    - visited: set of nodes that have been processed (popped)
    - in_queue: set of nodes currently in the queue (pushed but not yet processed)
    - result: yields the last node added until, at the end, yields the full solution path.
    """
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    queue = deque([start])
    visited = {start}
    in_queue = {start}
    prev = {start: None}
    yield visited.copy(), in_queue.copy(), None  # initial state
    while queue:
        current = queue.popleft()
        if current in in_queue:
            in_queue.remove(current)
        visited.add(current)
        if current == end:
            break
        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr][nc] == 0 and neighbor not in visited and neighbor not in in_queue:
                    in_queue.add(neighbor)
                    prev[neighbor] = current
                    queue.append(neighbor)
                    yield visited.copy(), in_queue.copy(), neighbor
    if end not in prev:
        yield visited.copy(), in_queue.copy(), None
    else:
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        yield visited.copy(), in_queue.copy(), path


# =========================
# Maze Widgets
# =========================

class MazeWidget(QtWidgets.QWidget):
    def __init__(self, maze=None, cell_size=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maze = maze
        self.cell_size = cell_size
        self.setMinimumSize(800, 800)
        self.solution = None
        self.traversal = set()
        self.queue_nodes = set()
        self.monster_folder = "Asserts/monster"
        self.monster_files = [os.path.join(self.monster_folder, f)
                              for f in os.listdir(self.monster_folder) if f.endswith(".gif")]
        if not self.monster_files:
            print("Error: No GIF files found in 'monster' folder!")
            sys.exit(1)
        self.monster_movies = {}
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(100)

    def updateFrame(self):
        self.update()

    def setMaze(self, maze):
        self.maze = maze
        self.monster_movies.clear()
        self.traversal = set()
        self.queue_nodes = set()
        rows = len(maze)
        cols = len(maze[0])
        for i in range(rows):
            for j in range(cols):
                if maze[i][j] == 1:
                    monster_gif = random.choice(self.monster_files)
                    movie = QtGui.QMovie(monster_gif)
                    movie.setScaledSize(QtCore.QSize(self.cell_size, self.cell_size))
                    movie.start()
                    self.monster_movies[(i, j)] = movie
        self.solution = None
        self.update()

    def paintEvent(self, event):
        if not self.maze:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rows = len(self.maze)
        cols = len(self.maze[0])
        for i in range(rows):
            for j in range(cols):
                rect = QtCore.QRect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                if self.maze[i][j] == 1:
                    painter.fillRect(rect, QtGui.QColor("#0A3346"))
                    if (i, j) in self.monster_movies:
                        frame = self.monster_movies[(i, j)].currentPixmap()
                        painter.drawPixmap(rect, frame)
                else:
                    painter.setPen(QtGui.QPen(QtGui.QColor("#FFFFFF"), 1))
                    painter.setBrush(QtGui.QColor("#AFDCF1"))
                    painter.drawRect(rect)
        border_pen = QtGui.QPen(QtGui.QColor("#444444"), 12)
        painter.setPen(border_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        outer_rect = QtCore.QRect(0, 0, cols * self.cell_size, rows * self.cell_size)
        painter.drawRect(outer_rect)
        if self.traversal:
            overlay_color = QtGui.QColor(0, 255, 0, 150)  # processed: green
            for (r, c) in self.traversal:
                rect = QtCore.QRect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                painter.fillRect(rect, overlay_color)
        if self.queue_nodes:
            queue_color = QtGui.QColor(255, 255, 0, 150)  # in queue: yellow
            for (r, c) in self.queue_nodes:
                rect = QtCore.QRect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                painter.fillRect(rect, queue_color)
        if self.solution:
            sol_pen = QtGui.QPen(QtGui.QColor("red"), 4)
            painter.setPen(sol_pen)
            points = []
            for (r, c) in self.solution:
                cx = int(c * self.cell_size + self.cell_size / 2)
                cy = int(r * self.cell_size + self.cell_size / 2)
                points.append(QtCore.QPoint(cx, cy))
            if points:
                painter.drawPolyline(QtGui.QPolygon(points))
        painter.end()


class MazeGraphWidget(QtWidgets.QWidget):
    def __init__(self, graph, rows, cols, cell_size=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = graph
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.setMinimumSize(cols * self.cell_size, rows * self.cell_size)
        self.solution = None
        self.traversal = set()
        self.queue_nodes = set()  # Add attribute for nodes in queue

    def setGraph(self, graph):
        self.graph = graph
        self.solution = None
        self.traversal = set()
        self.queue_nodes = set()
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        wall_color = QtGui.QColor("#1A1A1A")
        passage_color = QtGui.QColor("#AFDCF1")
        border_color = QtGui.QColor("#444444")
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * self.cell_size
                y = r * self.cell_size
                rect = QtCore.QRect(x, y, self.cell_size, self.cell_size)
                painter.fillRect(rect, passage_color)
        if self.traversal:
            overlay = QtGui.QColor(0, 255, 0, 150)  # processed: green
            for (r, c) in self.traversal:
                x = c * self.cell_size
                y = r * self.cell_size
                rect = QtCore.QRect(x, y, self.cell_size, self.cell_size)
                painter.fillRect(rect, overlay)
        if self.queue_nodes:
            queue_color = QtGui.QColor(255, 255, 0, 150)  # in queue: yellow
            for (r, c) in self.queue_nodes:
                x = c * self.cell_size
                y = r * self.cell_size
                rect = QtCore.QRect(x, y, self.cell_size, self.cell_size)
                painter.fillRect(rect, queue_color)
        edge_width = 5
        free_edge = 1
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * self.cell_size
                y = r * self.cell_size
                cell = (r, c)
                if r == 0:
                    pen_color = border_color
                    painter.setPen(QtGui.QPen(pen_color, edge_width))
                else:
                    if (r - 1, c) in self.graph[cell]:
                        pen_color = QtGui.QColor("white")
                        painter.setPen(QtGui.QPen(pen_color, free_edge))
                    else:
                        pen_color = wall_color
                        painter.setPen(QtGui.QPen(pen_color, edge_width))
                painter.drawLine(x, y, x + self.cell_size, y)
                if c == 0:
                    pen_color = border_color
                    painter.setPen(QtGui.QPen(pen_color, edge_width))
                else:
                    if (r, c - 1) in self.graph[cell]:
                        pen_color = QtGui.QColor("white")
                        painter.setPen(QtGui.QPen(pen_color, free_edge))
                    else:
                        pen_color = wall_color
                        painter.setPen(QtGui.QPen(pen_color, edge_width))
                painter.drawLine(x, y, x, y + self.cell_size)
        r = self.rows - 1
        for c in range(self.cols):
            x = c * self.cell_size
            y = r * self.cell_size + self.cell_size
            painter.setPen(QtGui.QPen(border_color, 10))
            painter.drawLine(x, y, x + self.cell_size, y)
        c = self.cols - 1
        for r in range(self.rows):
            x = c * self.cell_size + self.cell_size
            y = r * self.cell_size
            painter.setPen(QtGui.QPen(border_color, 10))
            painter.drawLine(x, y, x, y + self.cell_size)
        r = 0
        for c in range(self.cols):
            x = c * self.cell_size
            y = 0
            painter.setPen(QtGui.QPen(border_color, 10))
            painter.drawLine(x, y, x + self.cell_size, y)
        c = 0
        for r in range(self.rows):
            x = 0
            y = r * self.cell_size
            painter.setPen(QtGui.QPen(border_color, 10))
            painter.drawLine(x, y, x, y + self.cell_size)
        if self.solution:
            sol_pen = QtGui.QPen(QtGui.QColor("red"), 4)
            painter.setPen(sol_pen)
            pts = []
            for (r, c) in self.solution:
                cx = int(c * self.cell_size + self.cell_size / 2)
                cy = int(r * self.cell_size + self.cell_size / 2)
                pts.append(QtCore.QPoint(cx, cy))
            if pts:
                painter.drawPolyline(QtGui.QPolygon(pts))
        painter.end()


# =========================
# Main Window with Switch and Animation Buttons
# =========================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Maze Generator")
        bg_pixmap = QtGui.QPixmap("Asserts/background1.jpg")
        palette = self.palette()
        palette.setBrush(QtGui.QPalette.Background, QtGui.QBrush(bg_pixmap))
        self.setPalette(palette)
        self.showMaximized()

        self.default_rows = 16
        self.default_cols = 16

        self.rows = 0
        self.cols = 0
        self.graph = {}
        self.image = False

        self.start = (0, 0)
        self.end = (0, 0)

        if (self.default_rows >= self.default_cols):
            self.cell_size = int((16 * 50) / self.default_rows)
        else:
            self.cell_size = int((16 * 50) / self.default_cols)

        self.maze1 = generate_maze(self.default_rows, self.default_cols)
        self.maze_graph = generate_maze_graph(self.default_rows, self.default_cols)

        self.cell_maze_widget = MazeWidget(self.maze1, self.cell_size)
        self.cell_maze_widget.setMaze(self.maze1)
        self.graph_maze_widget = MazeGraphWidget(self.maze_graph, self.default_rows, self.default_cols, self.cell_size)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.cell_maze_widget)  # index 0: cell maze
        self.stack.addWidget(self.graph_maze_widget)  # index 1: graph maze

        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)

        row_label = QtWidgets.QLabel("Rows:")
        self.row_edit = QtWidgets.QLineEdit()
        self.row_edit.setPlaceholderText("e.g., 16")
        self.row_edit.setText(str(self.default_rows))
        col_label = QtWidgets.QLabel("Columns:")
        self.col_edit = QtWidgets.QLineEdit()
        self.col_edit.setPlaceholderText("e.g., 16")
        self.col_edit.setText(str(self.default_cols))
        controls_layout.addWidget(row_label)
        controls_layout.addWidget(self.row_edit)
        controls_layout.addWidget(col_label)
        controls_layout.addWidget(self.col_edit)

        self.generate_button = QtWidgets.QPushButton("Generate Maze")
        self.generate_button.clicked.connect(lambda: self.generateMaze(1, self.graph))
        controls_layout.addWidget(self.generate_button)

        switch_label = QtWidgets.QLabel("Select Maze Representation:")
        controls_layout.addWidget(switch_label)
        self.btn_cell = QtWidgets.QPushButton("Cell Maze")
        self.btn_graph = QtWidgets.QPushButton("Graph Maze")
        self.btn_cell.clicked.connect(lambda: self.handle_maze_change(0))
        self.btn_graph.clicked.connect(lambda: self.handle_maze_change(1))
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.btn_cell)
        btn_layout.addWidget(self.btn_graph)
        controls_layout.addLayout(btn_layout)
        controls_layout.addStretch()

        algo_label = QtWidgets.QLabel("Algorithm:")
        self.algo_combo = QtWidgets.QComboBox()
        self.algo_combo.addItems(["BFS", "Dijkstra", "A*"])
        controls_layout.addWidget(algo_label)
        controls_layout.addWidget(self.algo_combo)

        # Label to display solving time.
        self.solve_time_label = QtWidgets.QLabel("Solving time: N/A")
        controls_layout.addWidget(self.solve_time_label)

        self.solve_button = QtWidgets.QPushButton("Solve Maze")
        self.solve_button.clicked.connect(self.handleSolve)
        controls_layout.addWidget(self.solve_button)

        self.show_steps_button = QtWidgets.QPushButton("Show Steps")
        self.show_steps_button.clicked.connect(self.startBfsAnimation)
        controls_layout.addWidget(self.show_steps_button)

        self.show_steps_button = QtWidgets.QPushButton("Maze generate from image")
        self.show_steps_button.clicked.connect(self.handleImageButton)
        controls_layout.addWidget(self.show_steps_button)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.stack, stretch=1)
        main_layout.addWidget(controls_widget)
        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.bfs_timer = QtCore.QTimer(self)
        self.bfs_timer.timeout.connect(self.animateBfsStep)
        self.bfs_gen = None

    def handleImageButton(self):
        self.image = True
        graph, self.start, self.end, rows, cols = vision.main()
        self.graph = graph
        print(graph, rows, cols, self.start, self.end)
        self.set_rows_cols(rows, cols)
        self.generateMaze(2, graph)

    def set_rows_cols(self, rows, cols):
        self.rows = rows
        self.cols = cols
        print("set values")
        print(self.rows)
        print(self.cols)

    def get_rows_cols(self):
        print("get values")
        print(self.rows)
        print(self.cols)
        return self.rows, self.cols

    def handle_maze_change(self, index):
        self.stack.setCurrentIndex(index)
        self.clearSolution()


    def handleSolve(self):
        try:
            rows = int(self.row_edit.text())
            cols = int(self.col_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid integers for rows and columns.")
            return
        algo = self.algo_combo.currentText()

        if self.stack.currentIndex() == 0:
            # For cell maze: use graph solver on converted cell maze for all algorithms.
            cell_graph = self.cellMazeToGraph(self.maze1)
            path = solve.solve_maze_graph(cell_graph, (0, 0), (rows - 1, cols - 1), algorithm=algo)
            print("Solution path (cell maze):", path)
            self.cell_maze_widget.solution = path
            self.graph_maze_widget.solution = path
            self.cell_maze_widget.update()
            self.graph_maze_widget.update()
        else:
            if (self.image == False):
                print("handle solve False")
                self.solve_maze(self.maze_graph, (0, 0), (rows - 1, cols - 1), algorithm=algo)
            if (self.image == True):
                print("handle solve True")
                self.solve_maze(self.graph, self.start, self.end, algorithm=algo)


    def solve_maze(self, graph, start, end, algorithm="BFS"):

        path,time = solve.solve_maze_graph(graph, start, end, algorithm=algorithm)

        self.solve_time_label.setText(f"Solving time: {time*1000:.4f} Miliseconds")
        print("Solution path (graph maze):", path)
        self.cell_maze_widget.solution = path
        self.graph_maze_widget.solution = path
        self.cell_maze_widget.update()
        self.graph_maze_widget.update()

    def cellMazeToGraph(self, maze):
        rows = len(maze)
        cols = len(maze[0]) if rows > 0 else 0
        graph = {}
        for r in range(rows):
            for c in range(cols):
                if maze[r][c] == 0:
                    cell = (r, c)
                    neighbors = []
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                            neighbors.append((nr, nc))
                    graph[cell] = set(neighbors)
        return graph

    def generateMaze(self, index, graph):

        if index == 1:
            self.image = False
            try:
                rows = int(self.row_edit.text())
                cols = int(self.col_edit.text())
                if rows < 1 or cols < 1:
                    raise ValueError
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Invalid Input",
                                              "Please enter valid positive integers for rows and columns.")
                return
            self.maze1 = generate_maze(rows, cols)
            self.maze_graph = generate_maze_graph(rows, cols)
            if (rows >= cols):
                self.cell_maze_widget.cell_size = int((16 * 50) / rows)
                self.graph_maze_widget.cell_size = int((16 * 50) / rows)
            else:
                self.cell_maze_widget.cell_size = int((16 * 50) / cols)
                self.graph_maze_widget.cell_size = int((16 * 50) / cols)
            self.cell_maze_widget.setMaze(self.maze1)

            self.graph_maze_widget.rows = rows
            self.graph_maze_widget.cols = cols
            self.graph_maze_widget.setGraph(self.maze_graph)
            self.cell_maze_widget.setMinimumSize(cols * self.cell_maze_widget.cell_size,
                                                 rows * self.cell_maze_widget.cell_size)
            self.graph_maze_widget.setMinimumSize(cols * self.graph_maze_widget.cell_size,
                                                  rows * self.graph_maze_widget.cell_size)
        else:
            try:
                rows, cols = self.get_rows_cols()

                if rows < 1 or cols < 1:
                    raise ValueError
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Invalid Input",
                                              "Please enter valid positive integers for rows and columns.")
                return
            self.maze1 = generate_maze(rows, cols)
            self.maze_graph = generate_maze_graph(rows, cols)
            if (rows >= cols):
                self.cell_maze_widget.cell_size = int((16 * 50) / rows)
                self.graph_maze_widget.cell_size = int((16 * 50) / rows)
            else:
                self.cell_maze_widget.cell_size = int((16 * 50) / cols)
                self.graph_maze_widget.cell_size = int((16 * 50) / cols)
            # self.cell_maze_widget.setMaze(self.maze1)

            self.graph_maze_widget.rows = rows
            self.graph_maze_widget.cols = cols
            self.graph_maze_widget.setGraph(graph)
            self.cell_maze_widget.setMinimumSize(cols * self.cell_maze_widget.cell_size,
                                                 rows * self.cell_maze_widget.cell_size)
            self.graph_maze_widget.setMinimumSize(cols * self.graph_maze_widget.cell_size,
                                                  rows * self.graph_maze_widget.cell_size)

    def startBfsAnimation(self):
        try:
            rows = int(self.row_edit.text())
            cols = int(self.col_edit.text())
        except ValueError:
            return
        if self.image == True:
            start = self.start
            end = self.end
            if self.stack.currentIndex() == 0:
                self.bfs_gen = bfs_animate_matrix(self.maze1, start, end)
                self.cell_maze_widget.traversal = set()
                self.cell_maze_widget.queue_nodes = set()
                self.cell_maze_widget.solution = None
            else:
                self.bfs_gen = bfs_animate_graph(self.graph, start, end)
                self.graph_maze_widget.traversal = set()
                self.graph_maze_widget.queue_nodes = set()
                self.graph_maze_widget.solution = None
            self.bfs_timer.start(50)

        else:
            start = (0, 0)
            end = (rows - 1, cols - 1)
            if self.stack.currentIndex() == 0:
                self.bfs_gen = bfs_animate_matrix(self.maze1, start, end)
                self.cell_maze_widget.traversal = set()
                self.cell_maze_widget.queue_nodes = set()
                self.cell_maze_widget.solution = None
            else:
                self.bfs_gen = bfs_animate_graph(self.maze_graph, start, end)
                self.graph_maze_widget.traversal = set()
                self.graph_maze_widget.queue_nodes = set()
                self.graph_maze_widget.solution = None
            self.bfs_timer.start(50)

    def animateBfsStep(self):
        if self.stack.currentIndex() == 0:
            self.animateBfsStep_matrix()
        else:
            self.animateBfsStep_graph()

    def animateBfsStep_graph(self):
        try:
            visited, in_queue, result = next(self.bfs_gen)
            self.graph_maze_widget.traversal = visited
            self.graph_maze_widget.queue_nodes = in_queue
            if isinstance(result, list):
                self.graph_maze_widget.solution = result
                self.bfs_timer.stop()
        except StopIteration:
            self.bfs_timer.stop()
        self.graph_maze_widget.update()

    def animateBfsStep_matrix(self):
        try:
            processed, in_queue, result = next(self.bfs_gen)
            self.cell_maze_widget.traversal = processed
            self.cell_maze_widget.queue_nodes = in_queue
            if isinstance(result, list):
                self.cell_maze_widget.solution = result
                self.bfs_timer.stop()
        except StopIteration:
            self.bfs_timer.stop()
        self.cell_maze_widget.update()

    def clearSolution(self):
        self.cell_maze_widget.solution = None
        self.cell_maze_widget.traversal = set()
        self.cell_maze_widget.queue_nodes = set()
        self.graph_maze_widget.solution = None
        self.graph_maze_widget.traversal = set()
        self.graph_maze_widget.queue_nodes = set()
        self.cell_maze_widget.update()
        self.graph_maze_widget.update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
