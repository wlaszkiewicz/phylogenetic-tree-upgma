import os
import tempfile
import traceback

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QCursor
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QFileDialog, QMessageBox, QSpinBox, QGroupBox, QFormLayout, QTableWidget,
    QTableWidgetItem, QTabWidget, QListWidget, QListWidgetItem,
    QApplication, QDialog, QDialogButtonBox, QLineEdit, QRadioButton, QGraphicsScene, QGraphicsView, QLabel
)
from ete3 import Tree, TreeStyle, faces, NodeStyle

from alignment.center_star import build_msa, calculate_all_pairwise_alignments, reduce_msa, \
    calculate_distance_matrix_from_msa, \
    check_ultrametric_conditions
from alignment.upgma import upgma_algorithm
from gui.add_sequence_dialog import AddSequenceDialog
from gui.photo_viewer import PhotoViewer
from utils.fasta_parser import parse_fasta


class UPGMAApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UPGMA Phylogenetic Tree Tool")
        self.setWindowIcon(QIcon("icon.png"))
        window_width, window_height = 2000, 1500
        screen = QApplication.primaryScreen().availableGeometry()
        x = (screen.width() - window_width) // 2 + screen.left()
        y = (screen.height() - window_height) // 2 + screen.top()
        self.move(x, y)
        self.resize(window_width, window_height)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.current_sequences_with_ids = []
        self._setup_ui()
        self.tree_view = self.tabs["Graphical Tree"]
        self.original_tree_scene = QGraphicsScene()
        self.tree_view.setScene(self.original_tree_scene)
        self._toggle_input_sections()
        self._clear_results(clear_input=True)

    def _setup_ui(self):
        # Input Type
        input_type_group = QGroupBox("Input Data Type")
        input_type_layout = QHBoxLayout()
        self.seq_input_radio = QRadioButton("Sequences")
        self.seq_input_radio.setChecked(True)
        self.matrix_input_radio = QRadioButton("Distance Matrix")
        self.seq_input_radio.toggled.connect(self._toggle_input_sections)
        input_type_layout.addWidget(self.seq_input_radio)
        input_type_layout.addWidget(self.matrix_input_radio)
        input_type_group.setLayout(input_type_layout)
        self.main_layout.addWidget(input_type_group)

        # Test Data Buttons
        test_data_buttons_group = QGroupBox("Load Test Data")
        test_data_layout = QHBoxLayout()
        self.load_test_matrix_button = QPushButton("Load Test Matrix")
        self.load_test_matrix_button.clicked.connect(self.load_test_matrix_data)
        self.load_test_sequences_button = QPushButton("Load Test Sequences")
        self.load_test_sequences_button.clicked.connect(self.load_test_sequences_data)
        test_data_layout.addStretch(1)
        test_data_layout.addWidget(self.load_test_matrix_button)
        test_data_layout.addWidget(self.load_test_sequences_button)
        test_data_layout.addStretch(1)
        test_data_buttons_group.setLayout(test_data_layout)
        self.main_layout.addWidget(test_data_buttons_group)

        # Sequence Input
        self.seq_input_group = QGroupBox("Input Sequences")
        seq_layout = QVBoxLayout()
        self.sequence_list_widget = QListWidget()
        self.sequence_list_widget.setMinimumHeight(120)
        self.sequence_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        btn_layout = QHBoxLayout()
        btns = {"Load FASTA": self.load_fasta_file, "Add Manually": self.add_manual_sequence,
                "Remove Selected": self.remove_selected_sequences, "Clear All": self.clear_all_sequences_interactive}
        for name, func in btns.items():
            btn = QPushButton(name)
            btn.clicked.connect(func)
            btn_layout.addWidget(btn)
        seq_layout.addWidget(self.sequence_list_widget)
        seq_layout.addLayout(btn_layout)
        self.seq_input_group.setLayout(seq_layout)
        self.main_layout.addWidget(self.seq_input_group)

        self.scoring_group = QGroupBox("Alignment Scoring (Center Star)")
        scoring_layout = QHBoxLayout()
        self.scoring_group.setLayout(scoring_layout)

        self.match_score = QSpinBox()
        self.match_score.setRange(0, 100)
        self.match_score.setValue(1)
        self.match_score.setFixedWidth(300)
        self.mismatch_penalty = QSpinBox()
        self.mismatch_penalty.setRange(-100, 0)
        self.mismatch_penalty.setValue(-1)
        self.mismatch_penalty.setFixedWidth(300)
        self.gap_penalty = QSpinBox()
        self.gap_penalty.setRange(-100, 0)
        self.gap_penalty.setValue(-2)
        self.gap_penalty.setFixedWidth(300)

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
        self.main_layout.addWidget(self.scoring_group)

        # Matrix Input
        self.matrix_input_group = QGroupBox("Input Distance Matrix")
        matrix_layout = QFormLayout()
        self.matrix_item_names_input = QLineEdit()
        self.matrix_item_names_input.setPlaceholderText("A,B,C,D")
        self.matrix_data_input = QTextEdit()
        self.matrix_data_input.setFont(QFont("Courier New"))
        matrix_layout.addRow("Names:", self.matrix_item_names_input)
        matrix_layout.addRow("Matrix:", self.matrix_data_input)
        self.matrix_input_group.setLayout(matrix_layout)
        self.main_layout.addWidget(self.matrix_input_group)

        # Run Button
        self.run_button = QPushButton("▶️ Run UPGMA Calculation")
        self.run_button.setStyleSheet("font-size: 35px; padding: 8px;")
        self.run_button.clicked.connect(self.run_calculation)
        # ensure the cursor is reset (if returned earlier due to an error)
        QApplication.restoreOverrideCursor()
        QApplication.setOverrideCursor(Qt.ArrowCursor)
        QApplication.changeOverrideCursor(QCursor(Qt.ArrowCursor))
        QApplication.processEvents()
        self.main_layout.addWidget(self.run_button)

        # Output Tabs
        self.output_tabs = QTabWidget()
        self.tabs = {"Scoring Matrix": QTableWidget(), "Full MSA": QTableWidget(), "Reduced MSA": QTableWidget(),
                     "Distance Matrix": QTableWidget(), "Newick Tree": QTextEdit(), "Graphical Tree": PhotoViewer()}
        icons = {"Scoring Matrix": "🔢", "Full MSA": "📜", "Reduced MSA": "✂️", "Distance Matrix": "📏",
                 "Newick Tree": "🌳", "Graphical Tree": "📊"}

        for name, widget in self.tabs.items():
            tab = QWidget()
            layout = QVBoxLayout(tab)

            if name == "Full MSA":
                score_layout = QHBoxLayout()
                score_layout.addStretch()
                # Sum of Pairs Score
                self.sp_score_label = QLabel("Score (Sum-of-Pairs): N/A")
                score_layout.addWidget(self.sp_score_label)

                score_layout.addStretch()
                layout.addLayout(score_layout)

            layout.addWidget(widget)
            controls_layout = QHBoxLayout()

            saveable_tabs = {
                "Scoring Matrix": self.save_scoring_matrix_file, "Full MSA": lambda: self.save_msa_file("Full MSA"),
                "Reduced MSA": lambda: self.save_msa_file("Reduced MSA"),
                "Distance Matrix": self.save_distance_matrix_file,
                "Newick Tree": self.save_newick_tree_file, "Graphical Tree": self.save_graphical_tree_file,
            }

            if name in saveable_tabs:
                btn = QPushButton(f"💾 Save {name}")
                btn.clicked.connect(saveable_tabs[name])
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

            if controls_layout.count() > 0: layout.addLayout(controls_layout)
            tab.setLayout(layout)
            self.output_tabs.addTab(tab, f"{icons[name]} {name}")
        self.main_layout.addWidget(self.output_tabs)

    def _toggle_input_sections(self):
        is_seq = self.seq_input_radio.isChecked()
        self.seq_input_group.setVisible(is_seq)
        self.scoring_group.setVisible(is_seq)
        self.matrix_input_group.setVisible(not is_seq)

        self.output_tabs.setTabEnabled(0, is_seq)
        self.output_tabs.setTabEnabled(1, is_seq)
        self.output_tabs.setTabEnabled(2, is_seq)
        if not is_seq and self.output_tabs.currentIndex() < 3: self.output_tabs.setCurrentIndex(3)
        self._clear_results()

    def _update_sequence_list_widget(self):
        self.sequence_list_widget.clear()
        for i, (sid, sdata) in enumerate(self.current_sequences_with_ids):
            item = QListWidgetItem(f"{sid} ({len(sdata)}bp): {sdata[:40]}...")
            item.setData(Qt.UserRole, i)
            self.sequence_list_widget.addItem(item)
        self._clear_results()

    def _clear_results(self, clear_input=False):
        if clear_input:
            self.current_sequences_with_ids.clear()
            self.sequence_list_widget.clear()
            self.matrix_item_names_input.clear()
            self.matrix_data_input.clear()

        if hasattr(self, 'sp_score_label'):
            self.sp_score_label.setText("Score (Sum-of-Pairs): N/A")

        self.displayed_tree_pixmap = None
        for name, widget in self.tabs.items():
            if name in ["Scoring Matrix", "Full MSA", "Reduced MSA"] and not self.seq_input_radio.isChecked(): continue
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
            if hasattr(self, btn_attr): getattr(self, btn_attr).setEnabled(False)
        QApplication.processEvents()

    def _clear_sequence_input_data_only(self):
        self.current_sequences_with_ids.clear()
        self.sequence_list_widget.clear()

    def load_test_matrix_data(self):
        if self.current_sequences_with_ids or self.matrix_data_input.toPlainText().strip():
            if QMessageBox.question(self, "Confirm", "Clear current input and load test matrix?",
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.No: return
        self._clear_results(clear_input=True)
        self.matrix_input_radio.setChecked(True)
        self.matrix_item_names_input.setText("A,B,C,D,E,F,G")
        self.matrix_data_input.setText(
            "0.0 19.0 27.0 8.0 33.0 18.0 13.0\n19.0 0.0 31.0 18.0 36.0 1.0 13.0\n27.0 31.0 0.0 26.0 41.0 32.0 29.0\n8.0 18.0 26.0 0.0 31.0 17.0 14.0\n33.0 36.0 41.0 31.0 0.0 35.0 28.0\n18.0 1.0 32.0 17.0 35.0 0.0 12.0\n13.0 13.0 29.0 14.0 28.0 12.0 0.0")
        QMessageBox.information(self, "Test Data Loaded", "Test distance matrix has been loaded.")

    def load_test_sequences_data(self):
        if self.current_sequences_with_ids or self.matrix_data_input.toPlainText().strip():
            if QMessageBox.question(self, "Confirm", "Clear current input and load test sequences?",
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.No: return
        self._clear_results(clear_input=True)
        self.seq_input_radio.setChecked(True)
        self.current_sequences_with_ids = [("s1_test", "ATTGCCATT"), ("s2_test", "ATGGCCATT"),
                                           ("s3_test", "ATCCATTTTT"), ("s4_test", "ATCTTCTT"), ("s5_test", "ACTGACC")]
        self._update_sequence_list_widget()
        QMessageBox.information(self, "Test Data Loaded", "Test sequences have been loaded.")

    def load_fasta_file(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Open FASTA", "", "FASTA (*.fa *.fasta)")
        if not fp: return
        try:
            with open(fp, 'r') as f:
                new_seqs = parse_fasta(f.read())
            self.current_sequences_with_ids.extend(new_seqs)
            self._update_sequence_list_widget()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def add_manual_sequence(self):
        """
        Opens a dialog to manually add a new sequence.
        """
        existing_ids = [sid for sid, _ in self.current_sequences_with_ids]

        dialog = AddSequenceDialog(existing_ids=existing_ids, parent=self)

        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            self.current_sequences_with_ids.append(data)
            self._update_sequence_list_widget()

    def remove_selected_sequences(self):
        rows = {item.data(Qt.UserRole) for item in self.sequence_list_widget.selectedItems()}
        if rows: self.current_sequences_with_ids = [s for i, s in enumerate(self.current_sequences_with_ids) if
                                                    i not in rows]; self._update_sequence_list_widget()

    def clear_all_sequences_interactive(self):
        if self.current_sequences_with_ids and QMessageBox.question(self, "Confirm", "Clear all sequences?",
                                                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self._clear_results(clear_input=True)

    def run_calculation(self):
        self._clear_results()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.seq_input_radio.isChecked():
                if len(self.current_sequences_with_ids) < 2:
                    QMessageBox.warning(self, "Input Error", "Please provide at least two sequences.")
                    return
                item_names = [sid for sid, _ in self.current_sequences_with_ids]

                # 1. Perform all alignments ONCE and find the center
                center_idx, scoring_matrix, all_alignments = calculate_all_pairwise_alignments(
                    self.current_sequences_with_ids, self.match_score.value(),
                    self.mismatch_penalty.value(), self.gap_penalty.value()
                )
                self._display_scoring_matrix(scoring_matrix, item_names, center_idx)

                # 2. Build the MSA from the pre-computed alignments
                full_msa = build_msa(
                    center_idx, self.current_sequences_with_ids, all_alignments
                )

                sp_score = self.calculate_sp_score(full_msa)

                self._display_msa("Full MSA", full_msa, item_names, center_idx)
                self.sp_score_label.setText(f"Score (Sum-of-Pairs): {sp_score}")

                reduced = reduce_msa(full_msa)
                self._display_msa("Reduced MSA", reduced, item_names, center_idx)
                dist_matrix = calculate_distance_matrix_from_msa(reduced)
                if dist_matrix is None: return
            else:
                dist_matrix, item_names = self._parse_matrix_input()
                if dist_matrix is None: return

            is_ultrametric, violations = check_ultrametric_conditions(dist_matrix, item_names)
            if not is_ultrametric:
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
                result = dialog.exec_()
                if result != QDialog.Accepted:
                    return

            self._display_distance_matrix(dist_matrix, item_names)
            newick_tree = upgma_algorithm(dist_matrix, item_names)
            self.tabs["Newick Tree"].setText(newick_tree)

            if not newick_tree.startswith("ERROR:"):
                self._draw_tree(newick_tree)
                self.output_tabs.setCurrentWidget(self.tabs["Graphical Tree"].parentWidget())
            else:
                self._draw_tree(newick_tree)  # Display the error message in the tree view

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{e}\n\n{traceback.format_exc()}")
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.setOverrideCursor(Qt.ArrowCursor)
            QApplication.changeOverrideCursor(QCursor(Qt.ArrowCursor))
            QApplication.processEvents()

    def _parse_matrix_input(self):
        names_str, matrix_str = self.matrix_item_names_input.text(), self.matrix_data_input.toPlainText()
        if not names_str or not matrix_str:
            QMessageBox.warning(self, "Input Error", "Matrix and item names cannot be empty.")
            return None, None
        names = [n.strip() for n in names_str.split(',')]
        try:
            matrix_lines = matrix_str.strip().split('\n')
            matrix = np.array([[float(v) for v in line.strip().replace(',', ' ').split()] for line in matrix_lines])
            if matrix.shape[0] != len(names) or matrix.shape[1] != len(names):
                raise ValueError(f"Matrix dimensions ({matrix.shape}) do not match number of names ({len(names)}).")
            return matrix, names
        except Exception as e:
            QMessageBox.critical(self, "Matrix Error", f"Could not parse matrix: {e}")
            return None, None

    def _display_msa(self, tab_name: str, msa: list[str], names: list[str], center_idx: int):
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
                        item.setBackground(QColor("#FFE6EF"))  # Lighter pastel pink for center sequence
        table.resizeColumnsToContents()
        table.setUpdatesEnabled(True)

    def _display_scoring_matrix(self, matrix: np.ndarray, names: list[str], center_idx: int):
        """Displays the pairwise alignment score matrix with sums and highlighting."""
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

        highlight = QColor("#F5F5FF")  # Lighter purple (lavender)
        highlight_sum = QColor("#EAF6FF")  # Lighter blue
        highlight_seq = QColor("#FFE6EF")  # Lighter pastel pink

        bold_font = QFont()
        bold_font.setBold(True)

        row_sums = np.sum(matrix, axis=1)

        # Fill main matrix and highlight center row/column
        for r in range(n):
            for c in range(n):
                item = QTableWidgetItem(f"{matrix[r, c]:.0f}")
                if r == center_idx:
                    item.setBackground(highlight_seq)
                table.setItem(r, c, item)

        # Fill Sum column
        for i in range(n):
            sum_item_row = QTableWidgetItem(f"{row_sums[i]:.0f}")
            sum_item_row.setBackground(highlight_sum)
            if i == center_idx:
                sum_item_row.setFont(bold_font)
                sum_item_row.setBackground(highlight)
            table.setItem(i, n, sum_item_row)

        table.resizeColumnsToContents()
        table.setUpdatesEnabled(True)

    def _display_distance_matrix(self, matrix: np.ndarray, names: list[str]):
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
        view = self.tabs["Graphical Tree"]
        self.original_tree_scene.clear()
        self.displayed_tree_pixmap = None

        if not newick_string or newick_string.startswith("ERROR:"):
            error_text = newick_string if newick_string.startswith("ERROR:") else 'No valid tree data to display.'
            self.original_tree_scene.addText(error_text)
            return

        temp_image_path = None
        try:
            t = Tree(newick_string, format=1)
            ts = TreeStyle()
            ts.show_leaf_name = False
            ts.mode = "r"
            ts.show_branch_length = True
            leaves = t.get_leaves()
            if leaves:
                _, tree_height = t.get_farthest_leaf()
                target_pixel_width = 700
                ts.scale = (target_pixel_width / tree_height) if tree_height > 0 else 100
            else:
                ts.scale = 100
            ts.show_scale = True
            ts.branch_vertical_margin = 10
            colors = ["#F6C1E7", "#B5AEE4", "#A7D3F6", "#D1B3E0", "#C1C8F6", "#E3B6F7", "#B3E0F7"]

            def color_branches(node, color_idx=0):
                nstyle = NodeStyle()
                nstyle["hz_line_color"] = colors[color_idx % len(colors)]
                nstyle["vt_line_color"] = colors[color_idx % len(colors)]
                nstyle["hz_line_width"] = 3
                nstyle["vt_line_width"] = 3
                node.set_style(nstyle)
                for i, child in enumerate(node.children): color_branches(child, color_idx + i + 1)

            color_branches(t)
            leaf_font_size = 12 if len(t) <= 10 else 10
            for leaf in t.iter_leaves():
                leaf_name_face = faces.TextFace(leaf.name, fsize=leaf_font_size, fgcolor="black")
                leaf_name_face.margin_left = 6
                leaf.add_face(leaf_name_face, column=0, position="aligned")
                leaf_style = NodeStyle()
                leaf_style["size"] = 5
                leaf_style["fgcolor"] = "#606060"
                leaf.set_style(leaf_style)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_image_path = tmp.name
            t.render(temp_image_path, tree_style=ts, units="px")
            self.displayed_tree_pixmap = QPixmap(temp_image_path)
            if not self.displayed_tree_pixmap.isNull():
                self.original_tree_scene.addPixmap(self.displayed_tree_pixmap)
                view.reset_view()
            else:
                self.original_tree_scene.addText("Failed to load rendered tree image.")
        except Exception as e:
            self.original_tree_scene.addText(f'Error rendering tree:\n{str(e)}')
            traceback.print_exc()
        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except OSError:
                    pass

    def save_msa_file(self, msa_type: str):
        """
        Saves the specified MSA (Multiple Sequence Alignment) table to a file.
        The user can choose between FASTA format or a plain text matrix format.

        Args:
            msa_type (str): The key for the tab, e.g., "Full MSA" or "Reduced MSA".
        """
        table = self.tabs.get(msa_type)
        if not table or table.rowCount() == 0:
            QMessageBox.warning(self, "Warning", f"No {msa_type} data to save.")
            return

        default_filename = f"{msa_type.lower().replace(' ', '_')}.fasta"
        filter_str = "FASTA Format (*.fa *.fasta);;Text Matrix (*.txt);;All Files (*)"
        fp, selected_filter = QFileDialog.getSaveFileName(self, f"Save {msa_type}", f"output/{default_filename}",
                                                          filter_str)

        if not fp:
            return

        try:
            with open(fp, 'w') as f:
                if 'FASTA' in selected_filter:
                    for r in range(table.rowCount()):
                        name = table.verticalHeaderItem(r).text()
                        sequence = "".join([table.item(r, c).text() for c in range(table.columnCount())])
                        f.write(f">{name}\n{sequence}\n")
                else:
                    names = [table.verticalHeaderItem(r).text() for r in range(table.rowCount())]
                    max_name_len = max(len(name) for name in names) if names else 0

                    for r in range(table.rowCount()):
                        name = names[r]
                        sequence = "".join([table.item(r, c).text() for c in range(table.columnCount())])
                        f.write(f"{name.ljust(max_name_len + 2)}{sequence}\n")

            QMessageBox.information(self, "Success", f"{msa_type} saved successfully to\n{fp}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save file:\n{e}")

    def _save_table_data(self, table_name: str, default_filename: str):
        """
        A generic helper to save data from a QTableWidget to a file.
        The user can choose between CSV format or a plain text matrix format.

        Args:
            table_name (str): The key for the tab, e.g., "Scoring Matrix".
            default_filename (str): The suggested filename, e.g., "scoring_matrix.csv".
        """
        table = self.tabs[table_name]
        if table.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No data to save.")
            return

        filter_str = "CSV (Comma-separated) (*.csv);;Text Matrix (*.txt);;All Files (*)"
        fp, selected_filter = QFileDialog.getSaveFileName(self, f"Save {table_name}", f"output/{default_filename}",
                                                          filter_str)

        if not fp:
            return

        try:
            with open(fp, 'w', newline='') as f:
                col_headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
                row_headers = [table.verticalHeaderItem(i).text() for i in range(table.rowCount())]

                if 'CSV' in selected_filter:
                    # Write header row (with an empty top-left cell)
                    f.write(',' + ','.join(col_headers) + '\n')
                    for r in range(table.rowCount()):
                        row_data = [table.item(r, c).text() for c in range(table.columnCount())]
                        # Write row header, followed by the row's data
                        f.write(row_headers[r] + ',' + ','.join(row_data) + '\n')
                else:
                    # Determine column widths for text alignment
                    col_widths = [len(h) for h in col_headers]
                    max_row_header_width = max(len(h) for h in row_headers) if row_headers else 0

                    for c in range(table.columnCount()):
                        max_col_content = max((len(table.item(r, c).text()) for r in range(table.rowCount())),
                                              default=0)
                        col_widths[c] = max(col_widths[c], max_col_content)

                    # Write header
                    f.write(' ' * (max_row_header_width + 2))
                    for c, header in enumerate(col_headers):
                        f.write(header.ljust(col_widths[c] + 2))
                    f.write('\n')

                    # Write data rows
                    for r in range(table.rowCount()):
                        f.write(row_headers[r].ljust(max_row_header_width + 2))
                        for c in range(table.columnCount()):
                            item_text = table.item(r, c).text()
                            f.write(item_text.ljust(col_widths[c] + 2))
                        f.write('\n')

            QMessageBox.information(self, "Success", f"{table_name} saved successfully to\n{fp}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save file:\n{e}")

    def save_scoring_matrix_file(self):
        self._save_table_data("Scoring Matrix", "scoring_matrix.csv")

    def save_distance_matrix_file(self):
        self._save_table_data("Distance Matrix", "distance_matrix.csv")

    def save_newick_tree_file(self):
        text = self.tabs["Newick Tree"].toPlainText()
        if not text: QMessageBox.warning(self, "Warning", "No data to save."); return
        fp, _ = QFileDialog.getSaveFileName(self, "Save Newick Tree", "output/newick_tree.nwk", "Newick (*.nwk)")
        if fp:
            try:
                with open(fp, 'w') as f:
                    f.write(text)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save file:\n{e}")

    def save_graphical_tree_file(self):
        if not self.displayed_tree_pixmap or self.displayed_tree_pixmap.isNull():
            QMessageBox.warning(self, "Warning", "No image to save.")
            return
        fp, _ = QFileDialog.getSaveFileName(self, "Save Tree Image", "output/tree.png", "PNG (*.png)")
        if fp:
            if not self.displayed_tree_pixmap.save(fp):
                QMessageBox.critical(self, "Save Error", "An error occurred while saving the image.")

    def calculate_sp_score(self, msa: list[str]) -> int:
        if not msa or not msa[0]: return 0
        match_score, mismatch_penalty, gap_penalty = self.match_score.value(), self.mismatch_penalty.value(), self.gap_penalty.value()
        num_seqs, aln_len = len(msa), len(msa[0])
        total_score = 0
        for i in range(num_seqs):
            for j in range(i + 1, num_seqs):
                pair_score = 0
                for k in range(aln_len):
                    char1, char2 = msa[i][k], msa[j][k]
                    if char1 == '-' and char2 == '-':
                        continue
                    elif char1 == '-' or char2 == '-':
                        pair_score += gap_penalty
                    elif char1 == char2:
                        pair_score += match_score
                    else:
                        pair_score += mismatch_penalty
                total_score += pair_score
        return total_score
