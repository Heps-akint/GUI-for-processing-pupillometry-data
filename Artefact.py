# Import necessary libraries and modules
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QTabWidget, QVBoxLayout, QWidget,
                             QPushButton, QComboBox, QLabel, QLineEdit, QTableWidget, QTableWidgetItem,
                             QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt
from sklearn.cluster import KMeans
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.colors as mcolors

# Define a custom DataTable widget that inherits from QWidget
class DataTable(QWidget):
    # Constructor method for DataTable
    def __init__(self):
        # Call the constructor of the parent class (QWidget)
        super().__init__()
        # Initialize the user interface (UI) for this DataTable widget
        self.initUI()

    # Method to create the UI for DataTable widget
    def initUI(self):
        # Create a vertical box layout
        layout = QVBoxLayout()

        # Create a table widget and add it to the layout
        self.table = QTableWidget()
        layout.addWidget(self.table)

        # Set the layout for this DataTable widget
        self.setLayout(layout)
        
    # Method to set the data for the table widget
    def set_data(self, data):
        # Set the number of rows and columns for the table widget based on the data shape
        self.table.setRowCount(data.shape[0])
        self.table.setColumnCount(data.shape[1])

        # Set the horizontal header labels for the table widget using the column names from the data
        self.table.setHorizontalHeaderLabels(data.columns)

        # Iterate through the rows and columns of the data and populate the table widget with the data
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # Create a table widget item with the data value and add it to the table widget
                item = QTableWidgetItem(str(data.iloc[i, j]))
                self.table.setItem(i, j, item)

class DataGraph(QWidget):
    # Initialize the DataGraph object
    def __init__(self):
        super().__init__()
        self.initUI()

    # Set up the user interface for the DataGraph
    def initUI(self):
        layout = QVBoxLayout()

        # Create and add X-Axis dropdown menu to layout
        self.x_axis_combo = QComboBox()
        layout.addWidget(QLabel("X-Axis Column:"))
        layout.addWidget(self.x_axis_combo)

        # Create and add Y-Axis dropdown menu to layout
        self.y_axis_combo = QComboBox()
        layout.addWidget(QLabel("Y-Axis Column:"))
        layout.addWidget(self.y_axis_combo)

        # Create and add "Remove Outliers" checkbox to layout
        self.remove_outliers_cb = QCheckBox("Remove Outliers")
        layout.addWidget(self.remove_outliers_cb)

        # Create and add "Remove Zero Values" checkbox to layout
        self.remove_zeros_cb = QCheckBox("Remove Zero Values")
        layout.addWidget(self.remove_zeros_cb)

        # Create and add Range Filter dropdown menu to layout
        self.range_column_combo = QComboBox()
        layout.addWidget(QLabel("Select Column for Range Filter:"))
        layout.addWidget(self.range_column_combo)

        # Create and add minimum value input field for Range Filter
        self.min_value_input = QLineEdit()
        layout.addWidget(QLabel("Minimum Value:"))
        layout.addWidget(self.min_value_input)

        # Create and add maximum value input field for Range Filter
        self.max_value_input = QLineEdit()
        layout.addWidget(QLabel("Maximum Value:"))
        layout.addWidget(self.max_value_input)

        # Create and add "Apply Filters" button to layout
        self.apply_filters_btn = QPushButton("Apply Filters")
        self.apply_filters_btn.clicked.connect(self.apply_filters)
        layout.addWidget(self.apply_filters_btn)

        # Create a Figure and FigureCanvas, add it to the layout
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Set the layout for the DataGraph
        self.setLayout(layout)

    # Set the data for the DataGraph and initialize the UI elements
    def set_data(self, data):
        self.data = data  # Store the original data
        self.filtered_data = self.data.copy()  # Make a copy of the data for filtering
        self.x_axis_combo.clear()  # Clear the X-Axis dropdown menu
        self.y_axis_combo.clear()  # Clear the Y-Axis dropdown menu
        self.range_column_combo.clear()  # Clear the Range Filter dropdown menu
        self.x_axis_combo.addItems(self.data.columns)  # Add column names to the X-Axis dropdown menu
        self.y_axis_combo.addItems(self.data.columns)  # Add column names to the Y-Axis dropdown menu
        self.range_column_combo.addItems(self.data.columns)  # Add column names to the Range Filter dropdown menu
        self.plot()  # Plot the data using the current settings


        # Apply the selected filters to the data and update the plot
    def apply_filters(self):
        # Make a copy of the original data for filtering
        self.filtered_data = self.data.copy()

        # If "Remove Outliers" checkbox is checked, remove outliers based on the Interquartile Range (IQR)
        if self.remove_outliers_cb.isChecked():
            y_column = self.y_axis_combo.currentText()
            Q1 = self.data[y_column].quantile(0.25)
            Q3 = self.data[y_column].quantile(0.75)
            IQR = Q3 - Q1
            self.filtered_data = self.filtered_data[(self.filtered_data[y_column] >= (Q1 - 1.5 * IQR)) & (self.filtered_data[y_column] <= (Q3 + 1.5 * IQR))]

        # If "Remove Zero Values" checkbox is checked, remove rows with zero values
        if self.remove_zeros_cb.isChecked():
            self.filtered_data = self.filtered_data[(self.filtered_data != 0).all(axis=1)]

        # Apply the range filter based on the selected column and input values
        try:
            column = self.range_column_combo.currentText()
            min_value = self.min_value_input.text()
            max_value = self.max_value_input.text()

            # Print the filter settings for debugging
            print(f"Applying range filter: column={column}, min_value={min_value}, max_value={max_value}")

            # If both minimum and maximum values are provided, apply the range filter
            if min_value and max_value:
                min_value = float(min_value)
                max_value = float(max_value)
                self.filtered_data = self.filtered_data[(self.filtered_data[column] >= min_value) & (self.filtered_data[column] <= max_value)]

            # Print the filtered data for debugging
            print(f"Filtered data:\n{self.filtered_data}")

        # If a ValueError occurs, show a warning message for invalid input
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Please enter valid minimum and maximum values for the range filter.")
        # If any other exception occurs, show an error message with the exception details
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

        # Update the plot with the filtered data
        self.plot()

    # Plot the data on the graph
    def plot(self):
        # Clear the current figure
        self.figure.clear()
        # Create a new subplot
        ax = self.figure.add_subplot(111)
        # Get the selected X and Y columns
        x_column = self.x_axis_combo.currentText()
        y_column = self.y_axis_combo.currentText()

        # If both X and Y columns are selected, plot the data
        if x_column and y_column:
            x_data = self.filtered_data[x_column]
            y_data = self.filtered_data[y_column]

            # Subtract the mean of the y-axis data from each data point
            y_mean = y_data.mean()
            y_data_centered = y_data - y_mean

            # Plot the scatter plot with centered y-data
            ax.scatter(x_data, y_data_centered)

            # Set x and y-axis labels
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column + " (Centered)")

            # Set y-axis limits
            min_y = min(y_data_centered)
            max_y = max(y_data_centered)
            if min_y > 0:
                min_y = -min_y
            if max_y < 0:
                max_y = -max_y
            ax.set_ylim(min_y, max_y)

        self.canvas.draw()

class MachineLearningTab(QWidget):
    def __init__(self, parent=None):
        super(MachineLearningTab, self).__init__(parent)
        self.initUI()

    # Initialize the user interface for the Machine Learning tab
    def initUI(self):
        layout = QVBoxLayout()

        # Create a button to apply K-means clustering
        self.cluster_button = QPushButton("Apply K-means Clustering")
        self.cluster_button.clicked.connect(self.apply_kmeans_clustering)
        layout.addWidget(self.cluster_button)

        # Create a label to display the result of clustering
        self.result_label = QLabel()
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    # Apply K-means clustering when the button is clicked
    def apply_kmeans_clustering(self):
        try:
            result = self.run_clustering()
            # Show a smiley face if the clustering is successful, otherwise show a sad face
            if result:
                pixmap = QPixmap("smiley_face.png")
            else:
                pixmap = QPixmap("sad_face.png")

            self.result_label.setPixmap(pixmap)
            self.result_label.setScaledContents(True)

        # If any exception occurs, show an error message with the exception details
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    # Function to handle the clustering results
    def on_clustering_finished(self, result):
        # Enable the cluster button
        self.cluster_button.setEnabled(True)

        # Show a smiley face if the clustering is successful, otherwise show a sad face
        if result:
            pixmap = QPixmap("smiley_face.png")
        else:
            pixmap = QPixmap("sad_face.png")

        # Set the size of the result label and display the result image
        self.result_label.setFixedSize(200, 200)
        self.result_label.setPixmap(pixmap.scaled(self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # Run K-means clustering on the filtered data
    def run_clustering(self):
        # Get the main window and the graph viewer
        main_window = self.window()
        graph_viewer = main_window.get_graph_viewer()
        data = graph_viewer.filtered_data
        x_column = graph_viewer.x_axis_combo.currentText()
        y_column = graph_viewer.y_axis_combo.currentText()

        # Get the values from the selected X and Y columns
        X = data[[x_column, y_column]].values
        # Run K-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        labels = kmeans.labels_

        # Get the size of each cluster
        cluster_sizes = [sum(labels == 0), sum(labels == 1)]
        # Determine which cluster has the highest value on the y-axis
        higher_value_cluster = labels[X[:, 1].argmax()]
        # Determine which cluster is larger
        larger_cluster = cluster_sizes.index(max(cluster_sizes))

        # Return True if the higher value cluster and the larger cluster are the same, otherwise return False
        return higher_value_cluster == larger_cluster


class KMeansClusteringThread(QThread):
    # Signal to emit when the clustering thread is finished
    finished = pyqtSignal(bool)

    # Initialize the KMeansClusteringThread with data and column names for X and Y axes
    def __init__(self, data, x_column, y_column):
        super().__init__()
        self.data = data
        self.x_column = x_column
        self.y_column = y_column

    # The main function that runs when the thread is started
    def run(self):
        try:
            print("Clustering thread started.")
            # Get the values from the selected X and Y columns
            X = self.data[[self.x_column, self.y_column]].values
            # Run K-means clustering with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
            labels = kmeans.labels_

            # Get the size of each cluster
            cluster_sizes = [sum(labels == 0), sum(labels == 1)]
            # Determine which cluster has the highest value on the y-axis
            higher_value_cluster = labels[X[:, 1].argmax()]
            # Determine which cluster is larger
            larger_cluster = cluster_sizes.index(max(cluster_sizes))

            # Check if the higher value cluster and the larger cluster are the same
            result = higher_value_cluster == larger_cluster
            print("Clustering thread finished.")
            # Emit the signal with the clustering result
            self.finished.emit(result)

        # If any exception occurs, print the error and emit the signal with a False result
        except Exception as e:
            print(f"Error in clustering thread: {e}")
            self.finished.emit(False)


class CustomGraph(QWidget):
    # Initialize the CustomGraph widget
    def __init__(self):
        super().__init__()
        self.initUI()

    # Set up the user interface for the CustomGraph widget
    def initUI(self):
        layout = QVBoxLayout()

        self.x_axis_combo = QComboBox()
        layout.addWidget(QLabel("X-Axis Column:"))
        layout.addWidget(self.x_axis_combo)

        self.y_axis_combo = QComboBox()
        layout.addWidget(QLabel("Y-Axis Column:"))
        layout.addWidget(self.y_axis_combo)

        self.plot_btn = QPushButton("Plot Grid")
        self.plot_btn.clicked.connect(self.plot)
        layout.addWidget(self.plot_btn)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    # Set the data for the CustomGraph widget
    def set_data(self, data):
        self.data = data
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        self.x_axis_combo.addItems(self.data.columns)
        self.y_axis_combo.addItems(self.data.columns)

    # Plot the custom graph
    def plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        x_column = self.x_axis_combo.currentText()
        y_column = self.y_axis_combo.currentText()

        if x_column and y_column:
            x_data = self.data[x_column]
            y_data = self.data[y_column]

            # Normalize the data to a pixel range (e.g., 0-255)
            x_data_normalized = (x_data - x_data.min()) / (x_data.max() - x_data.min()) * 255
            y_data_normalized = (y_data - y_data.min()) / (y_data.max() - y_data.min()) * 255

            # Create a 256x256 grid with zeros (white pixels)
            pixel_grid = np.zeros((256, 256))

            # Set the corresponding grid positions to 1 (black pixels)
            for x, y in zip(x_data_normalized.astype(int), y_data_normalized.astype(int)):
                pixel_grid[y, x] = 1

            # Show the pixel grid
            ax.imshow(pixel_grid, cmap=mcolors.ListedColormap(['white', 'black']), origin='lower')

            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

        self.canvas.draw()
            
class MainWindow(QMainWindow):
    # Initialize the main window
    def __init__(self):
        super().__init__()
        self.initUI()

    # Set up the user interface for the main window
    def initUI(self):
        self.setWindowTitle("Data Analysis Tool")

        # Create the menu bar and the File menu
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        # Add the "Open" action to the File menu
        open_action = file_menu.addAction("Open")
        open_action.triggered.connect(self.open_file)

        # Create the central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Initialize the tab widget and add tabs for data viewer, graph viewer, machine learning, and custom graph
        self.tab_widget = QTabWidget()
        self.data_viewer = DataTable()
        self.graph_viewer = DataGraph()
        self.machine_learning_tab = MachineLearningTab()
        self.custom_graph = CustomGraph()

        self.tab_widget.addTab(self.data_viewer, "Data")
        self.tab_widget.addTab(self.graph_viewer, "Graph")
        self.tab_widget.addTab(self.machine_learning_tab, "Machine Learning")
        self.tab_widget.addTab(self.custom_graph, "Custom Graph")

        # Add the tab widget to the layout and set the layout for the central widget
        layout.addWidget(self.tab_widget)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    # Open a file and load the data into the data viewer, graph viewer, and custom graph
    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;All Files (*)")

        if file_name:
            try:
                data = pd.read_csv(file_name)
                self.data_viewer.set_data(data)
                self.graph_viewer.set_data(data)
                self.custom_graph.set_data(data)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while opening the file: {str(e)}")

    # Getter function to return the graph viewer
    def get_graph_viewer(self):
        return self.graph_viewer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
