import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Rows = 0
Cols = 0
#############################################
# Preprocessing and Outer Boundary Functions
#############################################

def preprocess_image(image_path):
    """
    Reads an image, converts to grayscale, applies Gaussian blur and a simple binary threshold,
    and returns (original_image, grayscale, binary).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")
    # Resize image if it’s too large
    max_dim = 800
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Simple threshold: pixels below 127 become black (0), above become white (255)
    _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    return img, gray, binary


def compute_outer_boundary(binary_img):
    """
    Scans the binary image (with walls = 0) and finds the minimum and maximum
    x and y coordinates where a black pixel is found.
    Returns (x_min, y_min, x_max, y_max).
    """
    ys, xs = np.where(binary_img == 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = binary_img.shape
        return (0, 0, w - 1, h - 1)
    x_min = int(np.min(xs))
    x_max = int(np.max(xs))
    y_min = int(np.min(ys))
    y_max = int(np.max(ys))
    return (x_min, y_min, x_max, y_max)


def draw_outer_boundary(image, boundary, color=(0, 0, 255), thickness=3):
    """
    Draws a rectangle on the image for the outer boundary.
    """
    x_min, y_min, x_max, y_max = boundary
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image


#############################################
# Hough Transform and Grid Lines Functions
#############################################

def angle_deg(theta):
    """Convert theta (radians) to degrees in [0, 180)."""
    return (theta * 180.0 / np.pi) % 180


def is_near_horizontal(theta, threshold_deg=10):
    """Return True if line angle is near 90° (horizontal)."""
    return abs(angle_deg(theta) - 90) < threshold_deg


def is_near_vertical(theta, threshold_deg=10):
    """Return True if line angle is near 0° or 180° (vertical)."""
    a_deg = angle_deg(theta)
    return a_deg < threshold_deg or a_deg > (180 - threshold_deg)


def group_lines(lines, rho_threshold, theta_threshold_deg):
    """
    Groups lines (each as (rho, theta)) that are close in both rho and theta.
    Returns a list of merged (rho, theta) tuples.
    """
    groups = []
    for (rho, theta) in lines:
        added = False
        for group in groups:
            rho0, theta0 = group[0]
            if abs(rho - rho0) < rho_threshold and abs(angle_deg(theta) - angle_deg(theta0)) < theta_threshold_deg:
                group.append((rho, theta))
                added = True
                break
        if not added:
            groups.append([(rho, theta)])
    merged = []
    for g in groups:
        avg_rho = sum(r for r, t in g) / len(g)
        avg_theta = sum(t for r, t in g) / len(g)
        merged.append((avg_rho, avg_theta))
    return merged


def filter_vertical_lines(lines, x_threshold=20):
    """
    For vertical lines, computes the x coordinate (x = |rho*cos(theta)|) and groups similar values.
    Returns a list of averaged x positions.
    """
    if not lines:
        return []
    x_vals = [abs(rho * np.cos(theta)) for rho, theta in lines]
    x_vals.sort()
    groups = [[x_vals[0]]]
    for x in x_vals[1:]:
        if x - groups[-1][-1] < x_threshold:
            groups[-1].append(x)
        else:
            groups.append([x])
    averaged_x = [sum(g) / len(g) for g in groups]
    return averaged_x


def filter_horizontal_lines(lines, y_threshold=20):
    """
    For horizontal lines, computes the y coordinate (y = |rho*sin(theta)|) and groups similar values.
    Returns a list of averaged y positions.
    """
    if not lines:
        return []
    y_vals = [abs(rho * np.sin(theta)) for rho, theta in lines]
    y_vals.sort()
    groups = [[y_vals[0]]]
    for y in y_vals[1:]:
        if y - groups[-1][-1] < y_threshold:
            groups[-1].append(y)
        else:
            groups.append([y])
    averaged_y = [sum(g) / len(g) for g in groups]
    return averaged_y


def detect_grid_lines(binary_img, outer_rect):
    """
    Applies Hough transform on the ROI defined by the outer boundary.
    Returns:
      - merged_horiz: merged horizontal lines (as (rho, theta)) – used mainly for reference.
      - averaged_x: list of vertical line x positions (global coordinates).
      - averaged_y: list of horizontal line y positions (global coordinates).
    """
    x_min, y_min, x_max, y_max = outer_rect
    roi = binary_img[y_min:y_max + 1, x_min:x_max + 1]
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(roi, kernel, iterations=1)
    edges = cv2.Canny(dilated, 30, 100, apertureSize=7)
    raw_lines = cv2.HoughLines(edges, 1, np.pi / 180, 90)
    if raw_lines is None:
        return [], [], []
    lines = [tuple(line[0]) for line in raw_lines]
    horiz_lines = []
    vert_lines = []
    for rho, theta in lines:
        if is_near_horizontal(theta, threshold_deg=10):
            horiz_lines.append((rho, theta))
        elif is_near_vertical(theta, threshold_deg=10):
            vert_lines.append((rho, theta))
    merged_horiz = group_lines(horiz_lines, rho_threshold=30, theta_threshold_deg=10)
    averaged_x = filter_vertical_lines(vert_lines, x_threshold=20)
    averaged_y = filter_horizontal_lines(horiz_lines, y_threshold=20)
    # Shift positions from ROI to global coordinates.
    averaged_x = [x + x_min for x in averaged_x]
    averaged_y = [y + y_min for y in averaged_y]
    return merged_horiz, averaged_x, averaged_y


############################################
# Cell Numbering and Extraction
############################################

def draw_cell_numbers(image, vert_boundaries, horiz_boundaries):
    """
    Overlays cell numbers on the image for each cell defined by vertical and horizontal boundaries.
    """
    cell_img = image.copy()
    num_rows = len(horiz_boundaries) - 1
    num_cols = len(vert_boundaries) - 1
    cell_num = 1
    for i in range(num_rows):
        for j in range(num_cols):
            x1 = vert_boundaries[j]
            x2 = vert_boundaries[j + 1]
            y1 = horiz_boundaries[i]
            y2 = horiz_boundaries[i + 1]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.putText(cell_img, str(cell_num), (cx - 10, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cell_num += 1
    return cell_img


def extract_cells(binary_img, vert_boundaries, horiz_boundaries):
    """
    Iterates over each cell (defined by adjacent vertical and horizontal boundaries)
    and samples the binary image along each cell side to determine if a wall is present.
    Walls are assumed if the average pixel value < 128 (since walls are black, 0).
    Returns a list of cell data in the form:
      [cell number, left, right, up, down]
    where left, right, up, down are booleans (1 for wall, 0 for open).
    """
    cells = []
    cell_num = 1
    num_rows = len(horiz_boundaries) - 1
    num_cols = len(vert_boundaries) - 1
    for i in range(num_rows):
        for j in range(num_cols):
            # Get cell boundaries.
            x_left = int(vert_boundaries[j])
            x_right = int(vert_boundaries[j + 1])
            y_top = int(horiz_boundaries[i])
            y_bottom = int(horiz_boundaries[i + 1])
            # Sample one column for left wall.
            left_region = binary_img[y_top:y_bottom, x_left:x_left + 1]
            avg_left = np.mean(left_region)
            left_wall = 1 if avg_left < 128 else 0
            # Sample one column for right wall.
            right_region = binary_img[y_top:y_bottom, x_right - 1:x_right]
            avg_right = np.mean(right_region)
            right_wall = 1 if avg_right < 128 else 0
            # Sample one row for top wall.
            top_region = binary_img[y_top:y_top + 1, x_left:x_right]
            avg_top = np.mean(top_region)
            up_wall = 1 if avg_top < 128 else 0
            # Sample one row for bottom wall.
            bottom_region = binary_img[y_bottom - 1:y_bottom, x_left:x_right]
            avg_bottom = np.mean(bottom_region)
            down_wall = 1 if avg_bottom < 128 else 0
            cells.append([i, j, cell_num, left_wall, right_wall, up_wall, down_wall])
            cell_num += 1
    return cells


def extract_start_end(cells, vert_boundaries, horiz_boundaries):
    num_rows = len(horiz_boundaries) - 2
    num_cols = len(vert_boundaries) - 2
    border_cells = []
    start_end = []
    print(num_rows)
    for cell in cells:
        if cell[0] == 0 or cell[0] == num_rows or cell[1] == 0 or cell[1] == num_cols:
            border_cells.append(cell)
            # print(border_cells)

    for cell in border_cells:

        # 0- row num
        # 1 - col num
        # 2 - cell num
        # 3 - left
        # 4 -right
        # 5 - top
        # 6 - bottom

        if cell[0] == 0:
            if cell[0] == 0 and cell[1] == 0:
                if cell[3] == 0 or cell[5] == 0:
                    start_end.append((cell[0], cell[1]))
                    print(cell[2])
            if cell[0] == 0 and cell[1] == num_cols:
                if cell[4] == 0 or cell[5] == 0:
                    start_end.append((cell[0], cell[1]))
                    print(cell[2])
            else:
                if cell[5] == 0:
                    start_end.append((cell[0], cell[1]))
                    print(cell[2])

        if cell[0] == num_rows:
            if cell[0] == num_rows and cell[1] == 0:
                if cell[3] == 0 or cell[6] == 0:
                    start_end.append((cell[0], cell[1]))
                    print(cell[2])
            if cell[0] == num_rows and cell[1] == num_cols:
                if cell[4] == 0 or cell[6] == 0:
                    start_end.append((cell[0], cell[1]))
                    print(cell[2])
            else:
                if cell[6] == 0:
                    start_end.append((cell[0], cell[1]))
                    print(cell[2])

        if cell[0] != num_rows and cell[0] != 0:
            if cell[1] == 0:
                if cell[3] == 0:
                    start_end.append((cell[0], cell[1]))
                    print("cell[1] == 0 and (cell[0] != 0 or cell[0] != num_rows):")
                    print(cell[2])
            if cell[1] == num_cols:
                if cell[4] == 0:
                    start_end.append((cell[0], cell[1]))
                    print("cell[1] == num_cols and (cell[0] != 0 or cell[0] != num_rows):")
                    print(cell[2])

    return start_end


def generate_graph_from_cells(cells, num_rows, num_cols, start_cell, end_cell):
    """
    Generates a graph from the cell data.
    Each cell is a node (represented as (row, col)), and an edge exists between adjacent cells
    if there is no wall separating them. Neighbors are stored in a set.

    The starting cell and end cell are provided as inputs (as (row, col) tuples).

    Parameters:
      cells: list of tuples in the format (row, column, cell_num, left, right, up, down)
      num_rows: total number of rows in the grid.
      num_cols: total number of columns in the grid.
      start_cell: tuple (row, col) for the starting cell.
      end_cell: tuple (row, col) for the ending cell.

    Returns:
      A tuple (graph, start_node, end_node) where:
        - graph is a dictionary mapping (row, col) to a set of neighbor (row, col) tuples.
        - start_node is the starting cell node.
        - end_node is the ending cell node.
    """
    graph = {}
    start_node = None
    end_node = None

    for cell in cells:
        # Unpack the cell tuple.
        # Expected format: (row, column, cell_num, left, right, up, down)
        row, column, cell_num, left, right, up, down = cell
        node = (row, column)
        neighbors = set()
        # Check left neighbor (if no wall on left and not on leftmost column)
        if left == 0 and column > 0:
            neighbors.add((row, column - 1))
        # Check right neighbor (if no wall on right and not on rightmost column)
        if right == 0 and column < num_cols - 1:
            neighbors.add((row, column + 1))
        # Check up neighbor (if no wall on up and not in the top row)
        if up == 0 and row > 0:
            neighbors.add((row - 1, column))
        # Check down neighbor (if no wall on down and not in the bottom row)
        if down == 0 and row < num_rows - 1:
            neighbors.add((row + 1, column))
        graph[node] = neighbors

        # Identify starting and ending nodes if they match the input.
        if node == start_cell:
            start_node = node
        if node == end_cell:
            end_node = node

    # Second pass: ensure bidirectionality. For every node, add the reverse connection.
    for node, neighbors in graph.items():
        for nbr in neighbors:
            if nbr in graph:
                graph[nbr].add(node)
            else:
                graph[nbr] = {node}

    # If the provided starting or ending cell wasn't found, add it as an isolated node.
    if start_node is None:
        graph[start_cell] = set()
        start_node = start_cell
    if end_node is None:
        graph[end_cell] = set()
        end_node = end_cell

    return graph, start_node, end_node


############################################
# Combined Processing Function
############################################

def process_maze_image(image_path):
    """
    Combines the entire pipeline:
      1. Preprocess the image to get binary.
      2. Compute the outer boundary.
      3. Detect grid lines within the outer boundary.
      4. Filter out grid lines that duplicate outer boundaries.
      5. Build complete vertical and horizontal boundary lists.
      6. Draw grid lines and number cells.
      7. Extract each cell's wall information.
    Returns:
      - numbered_img: final image with grid and cell numbers.
      - cell_width, cell_height: average cell sizes.
      - vert_boundaries, horiz_boundaries: boundary lists.
      - cells: list of cell data [cell no, left, right, up, down].
    """
    original, gray, binary = preprocess_image(image_path)
    outer_rect = compute_outer_boundary(binary)  # (x_min, y_min, x_max, y_max)
    x_min, y_min, x_max, y_max = outer_rect
    processed_img = original.copy()
    processed_img = draw_outer_boundary(processed_img, outer_rect, color=(0, 0, 255), thickness=3)

    # Detect grid lines in ROI.
    merged_horiz, averaged_x, averaged_y = detect_grid_lines(binary, outer_rect)

    # Skip grid lines that are too close to the outer boundary.
    boundary_threshold = 10
    final_x = [x for x in averaged_x if abs(x - x_min) >= boundary_threshold and abs(x - x_max) >= boundary_threshold]
    final_y = [y for y in averaged_y if abs(y - y_min) >= boundary_threshold and abs(y - y_max) >= boundary_threshold]

    # Build full boundaries.
    vert_boundaries = [x_min] + sorted(final_x) + [x_max]
    horiz_boundaries = [y_min] + sorted(final_y) + [y_max]

    # Draw grid lines on processed image.
    for x in vert_boundaries:
        cv2.line(processed_img, (int(x), y_min), (int(x), y_max), (0, 255, 0), 2)
    for y in horiz_boundaries:
        cv2.line(processed_img, (x_min, int(y)), (x_max, int(y)), (0, 255, 0), 2)

    # Number the cells.
    numbered_img = draw_cell_numbers(processed_img, vert_boundaries, horiz_boundaries)

    # Compute average cell sizes.
    cell_width = np.mean(np.diff(vert_boundaries)) if len(vert_boundaries) > 1 else None
    cell_height = np.mean(np.diff(horiz_boundaries)) if len(horiz_boundaries) > 1 else None

    # Extract cell wall information.
    cells = extract_cells(binary, vert_boundaries, horiz_boundaries)
    start_end_cells = extract_start_end(cells, vert_boundaries, horiz_boundaries)

    return numbered_img, cell_width, cell_height, vert_boundaries, horiz_boundaries, cells, start_end_cells


# def openImage(self):
#     global graph
#     options = QtWidgets.QFileDialog.Options()
#     options |= QtWidgets.QFileDialog.ReadOnly
#     filePath, _ = QtWidgets.QFileDialog.getOpenFileName(
#         self, "Select Maze Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)", options=options)
#     if filePath:
#         graph = self.processImage(filePath)
#     return graph


def processImage(filePath):
    try:
        numbered_img, cell_width, cell_height, vert_boundaries, horiz_boundaries, cells, start_end_cells = process_maze_image(
            filePath)
        num_rows = len(horiz_boundaries) - 1
        num_cols = len(vert_boundaries) - 1
        set_rows_cols(num_rows,num_cols)
        graph = generate_graph_from_cells(cells, num_rows, num_cols, start_end_cells[0], start_end_cells[1])
        # print(graph[0])
        # h_img, w_img, ch = numbered_img.shape
        # bytesPerLine = ch * w_img
        # qImg = QtGui.QImage(numbered_img.data, w_img, h_img, bytesPerLine, QtGui.QImage.Format_BGR888)
        # pixmap = QtGui.QPixmap.fromImage(qImg)
        # self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), QtCore.Qt.KeepAspectRatio))
        # info_text = ""
        # if cell_width is not None and cell_height is not None:
        #     info_text += f"Avg cell width: {cell_width:.1f}px, height: {cell_height:.1f}px\n"
        # info_text += f"starting Cell: {start_end_cells[0]}\n end cell: {start_end_cells[1]}"
        #
        # self.infoLabel.setText(info_text)
        return graph[0], graph[1], graph[2], num_rows, num_cols
    except Exception as e:
        print(e)
        # QtWidgets.QMessageBox.warning(self, "Error", str(e))
def set_rows_cols(rows,cols):
    Rows = rows
    Cols = cols


def get_rows_cols():
    print(Rows,Cols)
    return Rows, Cols


def main():
    """Main function to process the maze image and display the graph structure."""
    # Open file dialog to choose an image
    Tk().withdraw()  # Hide the root window
    image_path = askopenfilename(title="Select a Maze Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])

    if not image_path:
        print("No file selected. Exiting...")
        return

    try:
        graph1,start,end, rows, cols = processImage(image_path)
        print("graph from vision")
        print(graph1)
        return graph1, start, end, rows, cols

    except ValueError as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
# ############################################
# # PyQt5 Main Window
# ############################################
#
# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Maze Grid, Cell Numbering & Cell Extraction")
#         self.resize(1000, 800)
#         centralWidget = QtWidgets.QWidget()
#         self.setCentralWidget(centralWidget)
#         layout = QtWidgets.QVBoxLayout(centralWidget)
#
#         self.openButton = QtWidgets.QPushButton("Select Maze Image")
#         self.openButton.clicked.connect(self.openImage)
#         layout.addWidget(self.openButton)
#
#         self.imageLabel = QtWidgets.QLabel("Processed maze image will appear here")
#         self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
#         layout.addWidget(self.imageLabel, stretch=1)
#
#         self.infoLabel = QtWidgets.QLabel("Cell size, boundaries, and cell data will be shown here")
#         self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)
#         layout.addWidget(self.infoLabel)
#
#
# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())
