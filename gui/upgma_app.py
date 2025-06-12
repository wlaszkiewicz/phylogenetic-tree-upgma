import traceback
from typing import List, Tuple, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QColor, QCursor
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QFileDialog, QMessageBox, QSpinBox, QGroupBox, QFormLayout, QTableWidget,
    QTableWidgetItem, QTabWidget, QListWidget, QListWidgetItem,
    QApplication, QDialog, QDialogButtonBox, QLineEdit, QRadioButton,
    QGraphicsScene, QGraphicsView, QLabel
)

from alignment.center_star import (
    build_msa, calculate_all_pairwise_alignments, reduce_msa,
    calculate_distance_matrix_from_msa, check_ultrametric_conditions, calculate_sp_score
)
from alignment.upgma import upgma_algorithm
from gui.add_sequence_dialog import AddSequenceDialog
from gui.photo_viewer import PhotoViewer
from gui.tree_renderer import render_tree
from utils.fasta_parser import parse_fasta
from utils.file_saver import save_msa_data, save_table_data


class UPGMAApp(QMainWindow):
    """
    Main application window for the UPGMA Phylogenetic Tree Tool.

    Provides a GUI for loading sequence or distance matrix data, running UPGMA calculations,
    displaying results (scoring matrix, MSA, distance matrix, Newick tree, graphical tree),
    and saving outputs.
    """

    def __init__(self):
        """
        Initialize the UPGMAApp window and set up the UI components.
        """
        super().__init__()
        self.current_sequences_with_ids = []
        self.displayed_tree_pixmap = None
        self._init_ui()
        self._setup_tree_view()
        self._toggle_input_sections()
        self._clear_results(clear_input=True)

    def _init_ui(self):
        """
        Set up the main window UI, including all input and output sections.
        """
        self.setWindowTitle("UPGMA Phylogenetic Tree Tool")
        self.setWindowIcon(QIcon("icon.png"))
        self._set_window_geometry()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self._create_input_type_group()
        self._create_test_data_buttons()
        self._create_sequence_input_group()
        self._create_scoring_group()
        self._create_matrix_input_group()
        self._create_run_button()
        self._create_output_tabs()

    def _set_window_geometry(self):
        """
        Center and resize the main window on the screen.
        """
        window_width, window_height = 2000, 1500
        screen = QApplication.primaryScreen().availableGeometry()
        x = (screen.width() - window_width) // 2 + screen.left()
        y = (screen.height() - window_height) // 2 + screen.top()
        self.move(x, y)
        self.resize(window_width, window_height)

    def _create_input_type_group(self):
        """
        Create radio buttons for selecting input data type (sequences or distance matrix).
        """
        input_type_group = QGroupBox("Input Data Type")
        input_type_layout = QHBoxLayout()

        self.seq_input_radio = QRadioButton("Sequences")
        self.seq_input_radio.setChecked(True)
        self.matrix_input_radio = QRadioButton("Distance Matrix")
        self.seq_input_radio.toggled.connect(self._toggle_input_sections)

        input_type_layout.addStretch(2)
        input_type_layout.addWidget(self.seq_input_radio)
        input_type_layout.addStretch(2)
        input_type_layout.addWidget(self.matrix_input_radio)
        input_type_layout.addStretch(2)

        input_type_layout.setAlignment(Qt.AlignHCenter)
        input_type_group.setLayout(input_type_layout)
        self.main_layout.addWidget(input_type_group)

    def _create_test_data_buttons(self):
        """
        Create buttons for loading test matrix or test sequences.
        """
        test_data_buttons_group = QGroupBox("Load Test Data")
        test_data_layout = QHBoxLayout()

        self.load_test_matrix_button = QPushButton("Load Test Matrix")
        self.load_test_matrix_button.clicked.connect(self.load_test_matrix_data)

        self.load_test_sequences_button = QPushButton("Load Test Sequences")
        self.load_test_sequences_button.clicked.connect(self.load_test_sequences_data)

        test_data_layout.addStretch(2)
        test_data_layout.addWidget(self.load_test_matrix_button)
        test_data_layout.addStretch(1)
        test_data_layout.addWidget(self.load_test_sequences_button)
        test_data_layout.addStretch(2)

        test_data_buttons_group.setLayout(test_data_layout)
        self.main_layout.addWidget(test_data_buttons_group)

    def _create_sequence_input_group(self):
        """
        Create the sequence input group, including the list widget and sequence management buttons.
        """
        self.seq_input_group = QGroupBox("Input Sequences")
        seq_layout = QVBoxLayout()

        self.sequence_list_widget = QListWidget()
        self.sequence_list_widget.setMinimumHeight(120)
        self.sequence_list_widget.setSelectionMode(QListWidget.ExtendedSelection)

        btn_layout = QHBoxLayout()
        btns = {
            "Load FASTA": self.load_fasta_file,
            "Add Manually": self.add_manual_sequence,
            "Remove Selected": self.remove_selected_sequences,
            "Clear All": self.clear_all_sequences_interactive
        }

        for name, func in btns.items():
            btn = QPushButton(name)
            btn.clicked.connect(func)
            btn_layout.addWidget(btn)

        seq_layout.addWidget(self.sequence_list_widget)
        seq_layout.addLayout(btn_layout)
        self.seq_input_group.setLayout(seq_layout)
        self.main_layout.addWidget(self.seq_input_group)

    def _create_scoring_group(self):
        """
        Create the scoring parameter input group for sequence alignment.
        """
        self.scoring_group = QGroupBox("Alignment Scoring (Center Star)")
        scoring_layout = QHBoxLayout()

        self.match_score = self._create_spinbox(0, 100, 1, 300)
        self.mismatch_penalty = self._create_spinbox(-100, 0, -1, 300)
        self.gap_penalty = self._create_spinbox(-100, 0, -2, 300)

        scoring_layout.addStretch(1)
        scoring_layout.addWidget(QLabel("Match:"))
        scoring_layout.addWidget(self.match_score)
        scoring_layout.addStretch(2)
        scoring_layout.addWidget(QLabel("Mismatch:"))
        scoring_layout.addWidget(self.mismatch_penalty)
        scoring_layout.addStretch(2)
        scoring_layout.addWidget(QLabel("Gap:"))
        scoring_layout.addWidget(self.gap_penalty)
        scoring_layout.addStretch(1)

        self.scoring_group.setLayout(scoring_layout)
        self.main_layout.addWidget(self.scoring_group)

    def _create_spinbox(self, min_val, max_val, default, width):
        """
        Helper to create a QSpinBox with given parameters.

        Args:
            min_val (int): Minimum value.
            max_val (int): Maximum value.
            default (int): Default value.
            width (int): Fixed width.

        Returns:
            QSpinBox: Configured spinbox.
        """
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        spinbox.setFixedWidth(width)
        return spinbox

    def _create_matrix_input_group(self):
        """
        Create the input group for entering a distance matrix and item names.
        """
        self.matrix_input_group = QGroupBox("Input Distance Matrix")
        matrix_layout = QFormLayout()

        self.matrix_item_names_input = QLineEdit()
        self.matrix_item_names_input.setPlaceholderText("e.g. A,B,C,D...")

        self.matrix_data_input = QTextEdit()
        self.matrix_data_input.setPlaceholderText("e.g. \n0.0 1.0 2.0\n1.0 0.0 3.0\n2.0 3.0 0.0")

        matrix_layout.addRow("Names:", self.matrix_item_names_input)
        matrix_layout.addRow("Matrix:", self.matrix_data_input)
        self.matrix_input_group.setLayout(matrix_layout)
        self.main_layout.addWidget(self.matrix_input_group)

    def _create_run_button(self):
        """
        Create the button to run the UPGMA calculation.
        """
        self.run_button = QPushButton("▶️ Run UPGMA Calculation")
        self.run_button.setStyleSheet("font-size: 35px; padding: 8px;")
        self.run_button.clicked.connect(self.run_calculation)
        self.main_layout.addWidget(self.run_button)

    def _create_output_tabs(self):
        """
        Create the output tabs for displaying results (scoring matrix, MSA, distance matrix, trees).
        """
        self.output_tabs = QTabWidget()
        self.tabs = {
            "Scoring Matrix": QTableWidget(),
            "Full MSA": QTableWidget(),
            "Reduced MSA": QTableWidget(),
            "Distance Matrix": QTableWidget(),
            "Newick Tree": QTextEdit(),
            "Graphical Tree": PhotoViewer()
        }

        icons = {
            "Scoring Matrix": "🔢",
            "Full MSA": "📜",
            "Reduced MSA": "✂️",
            "Distance Matrix": "📏",
            "Newick Tree": "🌳",
            "Graphical Tree": "📊"
        }

        for name, widget in self.tabs.items():
            tab = QWidget()
            layout = QVBoxLayout(tab)

            if name == "Full MSA":
                score_layout = QHBoxLayout()
                score_layout.addStretch()
                self.sp_score_label = QLabel("Score (Sum-of-Pairs): N/A")
                score_layout.addWidget(self.sp_score_label)
                score_layout.addStretch()
                layout.addLayout(score_layout)

            layout.addWidget(widget)
            controls_layout = QHBoxLayout()
            self._add_tab_controls(name, widget, controls_layout)

            if controls_layout.count() > 0:
                layout.addLayout(controls_layout)

            tab.setLayout(layout)
            self.output_tabs.addTab(tab, f"{icons[name]} {name}")

        self.main_layout.addWidget(self.output_tabs)

    def _add_tab_controls(self, name, widget, controls_layout):
        """
        Add save and control buttons to each output tab.

        Args:
            name (str): Tab name.
            widget (QWidget): Tab widget.
            controls_layout (QHBoxLayout): Layout to add controls to.
        """
        save_actions = {
            "Scoring Matrix": self.save_scoring_matrix_file,
            "Full MSA": lambda: self.save_msa_file("Full MSA"),
            "Reduced MSA": lambda: self.save_msa_file("Reduced MSA"),
            "Distance Matrix": self.save_distance_matrix_file,
            "Newick Tree": self.save_newick_tree_file,
            "Graphical Tree": self.save_graphical_tree_file,
        }

        if name in save_actions:
            btn = QPushButton(f"💾 Save {name}")
            btn.clicked.connect(save_actions[name])
            setattr(self, f"save_{name.lower().replace(' ', '_')}_button", btn)
            controls_layout.addWidget(btn)

        if name == "Graphical Tree":
            zoom_in_btn = QPushButton("➕ Zoom In")
            zoom_in_btn.clicked.connect(widget.zoom_in)
            zoom_out_btn = QPushButton("➖ Zoom Out")
            zoom_out_btn.clicked.connect(widget.zoom_out)
            reset_view_btn = QPushButton("🔄 Reset View")
            reset_view_btn.clicked.connect(widget.reset_view)

            controls_layout.addStretch()
            controls_layout.addWidget(zoom_in_btn)
            controls_layout.addWidget(zoom_out_btn)
            controls_layout.addWidget(reset_view_btn)
            controls_layout.addStretch()

    def _setup_tree_view(self):
        """
        Set up the QGraphicsScene for the graphical tree view.
        """
        self.tree_view = self.tabs["Graphical Tree"]
        self.original_tree_scene = QGraphicsScene()
        self.tree_view.setScene(self.original_tree_scene)

    def _toggle_input_sections(self):
        """
        Show/hide input sections based on the selected input type (sequences or matrix).
        """
        is_seq = self.seq_input_radio.isChecked()
        self.seq_input_group.setVisible(is_seq)
        self.scoring_group.setVisible(is_seq)
        self.matrix_input_group.setVisible(not is_seq)

        self.output_tabs.setTabEnabled(0, is_seq)
        self.output_tabs.setTabEnabled(1, is_seq)
        self.output_tabs.setTabEnabled(2, is_seq)

        if not is_seq and self.output_tabs.currentIndex() < 3:
            self.output_tabs.setCurrentIndex(3)

        self._clear_results()

    def _update_sequence_list_widget(self):
        """
        Update the sequence list widget to reflect the current sequences.
        """
        self.sequence_list_widget.clear()
        for i, (sid, sdata) in enumerate(self.current_sequences_with_ids):
            item = QListWidgetItem(f"{sid} ({len(sdata)}bp): {sdata[:40]}...")
            item.setData(Qt.UserRole, i)
            self.sequence_list_widget.addItem(item)
        self._clear_results()

    def _clear_results(self, clear_input=False):
        """
        Clear all result displays and optionally input fields.

        Args:
            clear_input (bool): If True, also clear input fields.
        """
        if clear_input:
            self.current_sequences_with_ids.clear()
            self.sequence_list_widget.clear()
            self.matrix_item_names_input.clear()
            self.matrix_data_input.clear()

        if hasattr(self, 'sp_score_label'):
            self.sp_score_label.setText("Score (Sum-of-Pairs): N/A")

        self.displayed_tree_pixmap = None

        for name, widget in self.tabs.items():
            if name in ["Scoring Matrix", "Full MSA", "Reduced MSA"] and not self.seq_input_radio.isChecked():
                continue

            if isinstance(widget, QTableWidget):
                widget.clearContents()
                widget.setRowCount(0)
                widget.setColumnCount(0)
            elif isinstance(widget, QTextEdit):
                widget.clear()
            elif isinstance(widget, QGraphicsView) and hasattr(self, 'original_tree_scene'):
                self.original_tree_scene.clear()

        for name in self.tabs:
            btn_attr = f"save_{name.lower().replace(' ', '_')}_button"
            if hasattr(self, btn_attr):
                getattr(self, btn_attr).setEnabled(False)

        QApplication.processEvents()

    def load_test_matrix_data(self):
        """
        Load a predefined test distance matrix into the input fields.
        """
        if self.current_sequences_with_ids or self.matrix_data_input.toPlainText().strip():
            if QMessageBox.question(
                    self, "Confirm", "Clear current input and load test matrix?",
                    QMessageBox.Yes | QMessageBox.No
            ) == QMessageBox.No:
                return

        self._clear_results(clear_input=True)
        self.matrix_input_radio.setChecked(True)
        self.matrix_item_names_input.setText("A,B,C,D,E,F,G")
        self.matrix_data_input.setText(
            "0.0 19.0 27.0 8.0 33.0 18.0 13.0\n"
            "19.0 0.0 31.0 18.0 36.0 1.0 13.0\n"
            "27.0 31.0 0.0 26.0 41.0 32.0 29.0\n"
            "8.0 18.0 26.0 0.0 31.0 17.0 14.0\n"
            "33.0 36.0 41.0 31.0 0.0 35.0 28.0\n"
            "18.0 1.0 32.0 17.0 35.0 0.0 12.0\n"
            "13.0 13.0 29.0 14.0 28.0 12.0 0.0"
        )
        QMessageBox.information(self, "Test Data Loaded", "Test distance matrix has been loaded.")

    def load_test_sequences_data(self):
        """
        Load a predefined set of test sequences into the input fields.
        """
        if self.current_sequences_with_ids or self.matrix_data_input.toPlainText().strip():
            if QMessageBox.question(
                    self, "Confirm", "Clear current input and load test sequences?",
                    QMessageBox.Yes | QMessageBox.No
            ) == QMessageBox.No:
                return

        self._clear_results(clear_input=True)
        self.seq_input_radio.setChecked(True)
        self.current_sequences_with_ids = [
            ("s1_test", "ATTGCCATT"),
            ("s2_test", "ATGGCCATT"),
            ("s3_test", "ATCCATTTTT"),
            ("s4_test", "ATCTTCTT"),
            ("s5_test", "ACTGACC")
        ]
        self._update_sequence_list_widget()
        QMessageBox.information(self, "Test Data Loaded", "Test sequences have been loaded.")

    def load_fasta_file(self):
        """
        Open a file dialog to load sequences from a FASTA file.
        """
        fp, _ = QFileDialog.getOpenFileName(self, "Open FASTA", "", "FASTA (*.fa *.fasta)")
        if not fp:
            return

        try:
            with open(fp, 'r') as f:
                new_seqs = parse_fasta(f.read())
            self.current_sequences_with_ids.extend(new_seqs)
            self._update_sequence_list_widget()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def add_manual_sequence(self):
        """
        Open a dialog to manually add a new sequence.
        """
        existing_ids = [sid for sid, _ in self.current_sequences_with_ids]
        dialog = AddSequenceDialog(existing_ids=existing_ids, parent=self)

        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            self.current_sequences_with_ids.append(data)
            self._update_sequence_list_widget()

    def remove_selected_sequences(self):
        """
        Remove the selected sequences from the list.
        """
        rows = {item.data(Qt.UserRole) for item in self.sequence_list_widget.selectedItems()}
        if rows:
            self.current_sequences_with_ids = [
                s for i, s in enumerate(self.current_sequences_with_ids) if i not in rows
            ]
            self._update_sequence_list_widget()

    def clear_all_sequences_interactive(self):
        """
        Prompt the user to confirm and clear all sequences if confirmed.
        """
        if self.current_sequences_with_ids and QMessageBox.question(
                self, "Confirm", "Clear all sequences?", QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes:
            self._clear_results(clear_input=True)

    def run_calculation(self):
        """
        Run the UPGMA calculation based on the current input type and data.
        """
        self._clear_results()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            if self.seq_input_radio.isChecked():
                self._run_sequence_based_calculation()
            else:
                self._run_matrix_based_calculation()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{e}\n\n{traceback.format_exc()}")
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.setOverrideCursor(Qt.ArrowCursor)
            QApplication.changeOverrideCursor(QCursor(Qt.ArrowCursor))
            QApplication.processEvents()

    def _run_sequence_based_calculation(self):
        """
        Run the calculation pipeline for sequence-based input.
        """
        if len(self.current_sequences_with_ids) < 2:
            QMessageBox.warning(self, "Input Error", "Please provide at least two sequences.")
            return

        item_names = [sid for sid, _ in self.current_sequences_with_ids]
        center_idx, scoring_matrix, all_alignments = calculate_all_pairwise_alignments(
            self.current_sequences_with_ids,
            self.match_score.value(),
            self.mismatch_penalty.value(),
            self.gap_penalty.value()
        )

        self._display_scoring_matrix(scoring_matrix, item_names, center_idx)
        self.save_scoring_matrix_button.setEnabled(True)

        full_msa = build_msa(center_idx, self.current_sequences_with_ids, all_alignments)
        sp_score = calculate_sp_score(
            full_msa,
            self.match_score.value(),
            self.mismatch_penalty.value(),
            self.gap_penalty.value()
        )

        self._display_msa("Full MSA", full_msa, item_names, center_idx)
        self.sp_score_label.setText(f"Score (Sum-of-Pairs): {sp_score}")
        self.save_full_msa_button.setEnabled(True)

        reduced = reduce_msa(full_msa)
        self._display_msa("Reduced MSA", reduced, item_names, center_idx)
        self.save_reduced_msa_button.setEnabled(True)

        dist_matrix = calculate_distance_matrix_from_msa(reduced)
        if dist_matrix is None:
            return

        self._process_distance_matrix(dist_matrix, item_names)

    def _run_matrix_based_calculation(self):
        """
        Run the calculation pipeline for matrix-based input.
        """
        dist_matrix, item_names = parse_matrix_input(
            self.matrix_item_names_input.text(),
            self.matrix_data_input.toPlainText()
        )

        if dist_matrix is None:
            QMessageBox.warning(self, "Input Error", "Matrix and item names cannot be empty.")
            return

        self._process_distance_matrix(dist_matrix, item_names)

    def _process_distance_matrix(self, dist_matrix, item_names):
        """
        Display the distance matrix, check ultrametricity, and run UPGMA.

        Args:
            dist_matrix (np.ndarray): Distance matrix.
            item_names (List[str]): Names of items.
        """
        self._display_distance_matrix(dist_matrix, item_names)
        self.save_distance_matrix_button.setEnabled(True)

        is_ultrametric, violations = check_ultrametric_conditions(dist_matrix, item_names)
        if not is_ultrametric and not self._confirm_ultrametric_violation(violations):
            return

        newick_tree = upgma_algorithm(dist_matrix, item_names)
        self.tabs["Newick Tree"].setText(newick_tree)
        self.save_newick_tree_button.setEnabled(True)

        if not newick_tree.startswith("ERROR:"):
            self._draw_tree(newick_tree)
            if self.displayed_tree_pixmap and not self.displayed_tree_pixmap.isNull():
                self.save_graphical_tree_button.setEnabled(True)
            self.output_tabs.setCurrentWidget(self.tabs["Graphical Tree"].parentWidget())

    def _confirm_ultrametric_violation(self, violations):
        """
        Show a dialog warning about ultrametric condition violations and ask user to continue.

        Args:
            violations (list): List of ultrametric violations.

        Returns:
            bool: True if user wants to continue, False otherwise.
        """
        msg = (
            "The results might not be as reliable because the distance matrix "
            "does not meet the strict ultrametric condition.\n\n"
            "For UPGMA to be perfectly accurate, for any three items, the two largest "
            "distances between them must be equal. This condition has been violated.\n\n"
            f"Example violations found:\n{violations[:1]}\n\n"
            "Do you want to continue with the calculation anyway?"
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("Matrix Condition Warning")
        layout = QVBoxLayout(dialog)

        label = QLabel(msg)
        label.setWordWrap(True)
        layout.addWidget(label)

        button_box = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        return dialog.exec_() == QDialog.Accepted

    def _display_msa(self, tab_name: str, msa: List[str], names: List[str], center_idx: int):
        """
        Display a multiple sequence alignment in the specified tab.

        Args:
            tab_name (str): Name of the tab.
            msa (List[str]): List of aligned sequences.
            names (List[str]): Sequence names.
            center_idx (int): Index of the center sequence.
        """
        table = self.tabs[tab_name]
        if not msa or not msa[0]:
            table.clearContents()
            table.setRowCount(0)
            return

        table.clear()
        table.setUpdatesEnabled(False)
        table.setRowCount(len(names))
        table.setColumnCount(len(msa[0]))
        table.setVerticalHeaderLabels(names)

        center_font = QFont()
        center_font.setBold(True)
        table.verticalHeaderItem(center_idx).setFont(center_font)

        for r, row_str in enumerate(msa):
            for c, char in enumerate(row_str):
                table.setItem(r, c, QTableWidgetItem(char))
                if r == center_idx:
                    item = table.item(r, c)
                    if item:
                        item.setBackground(QColor("#FFE6EF"))

        table.resizeColumnsToContents()
        table.setUpdatesEnabled(True)

    def _display_scoring_matrix(self, matrix: np.ndarray, names: List[str], center_idx: int):
        """
        Display the scoring matrix in the corresponding tab.

        Args:
            matrix (np.ndarray): Scoring matrix.
            names (List[str]): Sequence names.
            center_idx (int): Index of the center sequence.
        """
        table = self.tabs["Scoring Matrix"]
        table.clear()
        table.setUpdatesEnabled(False)

        n = len(names)
        table.setColumnCount(n + 1)
        table.setRowCount(n)
        table.setHorizontalHeaderLabels(names + ["Sum"])
        table.setVerticalHeaderLabels(names)

        center_font = QFont()
        center_font.setBold(True)
        table.verticalHeaderItem(center_idx).setFont(center_font)
        table.horizontalHeaderItem(center_idx).setFont(center_font)

        highlight = QColor("#F5F5FF")
        highlight_sum = QColor("#EAF6FF")
        highlight_seq = QColor("#FFE6EF")
        bold_font = QFont()
        bold_font.setBold(True)

        row_sums = np.sum(matrix, axis=1)

        for r in range(n):
            for c in range(n):
                item = QTableWidgetItem(f"{matrix[r, c]:.0f}")
                if r == center_idx:
                    item.setBackground(highlight_seq)
                table.setItem(r, c, item)

        for i in range(n):
            sum_item_row = QTableWidgetItem(f"{row_sums[i]:.0f}")
            sum_item_row.setBackground(highlight_sum)
            if i == center_idx:
                sum_item_row.setFont(bold_font)
                sum_item_row.setBackground(highlight)
            table.setItem(i, n, sum_item_row)

        table.resizeColumnsToContents()
        table.setUpdatesEnabled(True)

    def _display_distance_matrix(self, matrix: np.ndarray, names: List[str]):
        """
        Display the distance matrix in the corresponding tab.

        Args:
            matrix (np.ndarray): Distance matrix.
            names (List[str]): Item names.
        """
        table = self.tabs["Distance Matrix"]
        table.clear()
        table.setUpdatesEnabled(False)

        n = len(names)
        table.setRowCount(n)
        table.setColumnCount(n)
        table.setHorizontalHeaderLabels(names)
        table.setVerticalHeaderLabels(names)

        for r in range(n):
            for c in range(n):
                table.setItem(r, c, QTableWidgetItem(f"{matrix[r, c]:.4f}"))

        table.resizeColumnsToContents()
        table.setUpdatesEnabled(True)

    def _draw_tree(self, newick_string: str):
        """
        Render and display the phylogenetic tree from a Newick string.

        Args:
            newick_string (str): Newick tree string.
        """
        view = self.tabs["Graphical Tree"]
        self.original_tree_scene.clear()
        self.displayed_tree_pixmap = None

        if not newick_string or newick_string.startswith("ERROR:"):
            error_text = newick_string if newick_string.startswith("ERROR:") else 'No valid tree data to display.'
            self.original_tree_scene.addText(error_text)
            return

        pixmap = render_tree(newick_string)
        if pixmap:
            self.displayed_tree_pixmap = pixmap
            self.original_tree_scene.addPixmap(pixmap)
            view.reset_view()
        else:
            self.original_tree_scene.addText("Failed to render tree image.")

    def save_msa_file(self, msa_type: str):
        """
        Save the MSA table to a file.

        Args:
            msa_type (str): Type of MSA ("Full MSA" or "Reduced MSA").
        """
        save_msa_data(self.tabs.get(msa_type), msa_type, self)

    def save_scoring_matrix_file(self):
        """
        Save the scoring matrix to a CSV file.
        """
        save_table_data(self.tabs["Scoring Matrix"], "Scoring Matrix", "scoring_matrix.csv", self)

    def save_distance_matrix_file(self):
        """
        Save the distance matrix to a CSV file.
        """
        save_table_data(self.tabs["Distance Matrix"], "Distance Matrix", "distance_matrix.csv", self)

    def save_newick_tree_file(self):
        """
        Save the Newick tree string to a file.
        """
        text = self.tabs["Newick Tree"].toPlainText()
        if not text:
            QMessageBox.warning(self, "Warning", "No data to save.")
            return

        fp, _ = QFileDialog.getSaveFileName(self, "Save Newick Tree", "output/newick_tree.nwk", "Newick (*.nwk)")
        if fp:
            try:
                with open(fp, 'w') as f:
                    f.write(text)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save file:\n{e}")

    def save_graphical_tree_file(self):
        """
        Save the graphical tree image to a PNG file.
        """
        if not self.displayed_tree_pixmap or self.displayed_tree_pixmap.isNull():
            QMessageBox.warning(self, "Warning", "No image to save.")
            return

        fp, _ = QFileDialog.getSaveFileName(self, "Save Tree Image", "output/tree.png", "PNG (*.png)")
        if fp and not self.displayed_tree_pixmap.save(fp):
            QMessageBox.critical(self, "Save Error", "An error occurred while saving the image.")


def parse_matrix_input(names_str: str, matrix_str: str) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
     Parses matrix input from user text input fields.

     Args:
         names_str (str): Comma-separated names of items.
         matrix_str (str): String representation of the matrix (rows separated by newlines).

     Returns:
         Tuple[Optional[np.ndarray], Optional[List[str]]]: Tuple of parsed matrix and list of names,
         or (None, None) if parsing fails.
     """
    if not names_str or not matrix_str:
        return None, None

    names = [n.strip() for n in names_str.split(',')]
    try:
        matrix_lines = matrix_str.strip().split('\n')
        matrix = np.array([[float(v) for v in line.strip().replace(',', ' ').split()] for line in matrix_lines])

        if matrix.shape[0] != len(names) or matrix.shape[1] != len(names):
            raise ValueError(f"Matrix dimensions ({matrix.shape}) do not match number of names ({len(names)}).")

        return matrix, names
    except Exception:
        return None, None
