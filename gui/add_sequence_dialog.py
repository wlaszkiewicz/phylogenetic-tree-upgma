from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QTextEdit, QDialogButtonBox, QMessageBox


class AddSequenceDialog(QDialog):
    """
    A dialog for manually adding a sequence with an optional ID.
    """

    def __init__(self, default_id_prefix="ManualSeq", existing_ids=None, parent=None):
        """
        Initializes the AddSequenceDialog.

        Args:
            default_id_prefix (str): Prefix for suggesting a default sequence ID.
            existing_ids (list[str], optional): A list of already existing sequence IDs
                                                 to avoid conflicts. Defaults to None.
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("Add Sequence Manually")
        self.setMinimumWidth(400)

        layout = QFormLayout(self)

        self.id_edit = QLineEdit()
        self.seq_edit = QTextEdit()
        self.seq_edit.setAcceptRichText(False)
        self.seq_edit.setPlaceholderText("Enter sequence data (e.g., ATGCGTCAG)...")
        self.seq_edit.setMinimumHeight(100)

        self.default_id_prefix = default_id_prefix
        self.existing_ids_for_dialog = existing_ids if existing_ids else []

        count = 1
        suggested_id = f"{self.default_id_prefix}{count}"
        while suggested_id in self.existing_ids_for_dialog:
            count += 1
            suggested_id = f"{self.default_id_prefix}{count}"
        self.id_edit.setPlaceholderText(f"Optional, e.g., {suggested_id}")

        layout.addRow("Sequence ID:", self.id_edit)
        layout.addRow("Sequence Data:", self.seq_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept_input)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.input_id = ""
        self.input_sequence = ""

    def accept_input(self):
        """
        Validates the input and, if valid, accepts the dialog.
        Otherwise, shows a warning message.
        """
        seq_id = self.id_edit.text().strip().replace(" ", "_") # Allow underscores
        seq_data = self.seq_edit.toPlainText().strip().upper().replace(" ", "").replace("\n", "").replace("\r", "")

        if not seq_data:
            QMessageBox.warning(self, "Input Error", "Sequence data cannot be empty.")
            return

        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ-")
        invalid_found = [char for char in seq_data if char not in valid_chars]
        if invalid_found:
            QMessageBox.warning(self, "Invalid Characters",
                                  f"Sequence data contains invalid characters: {', '.join(set(invalid_found))}. "
                                  "Only A-Z and '-' are allowed.")
            return

        if not seq_id:
            count = 1
            seq_id = f"{self.default_id_prefix}{count}"
            while seq_id in self.existing_ids_for_dialog:
                count += 1
                seq_id = f"{self.default_id_prefix}{count}"
        elif seq_id in self.existing_ids_for_dialog:
            QMessageBox.warning(self, "ID Conflict",
                                  f"The sequence ID '{seq_id}' already exists. Please choose a different ID.")
            return

        self.input_id = seq_id
        self.input_sequence = seq_data
        self.accept()

    def get_data(self) -> tuple[str, str]:
        """
        Returns the entered sequence ID and data if the dialog was accepted.

        Returns:
            A tuple (sequence_id, sequence_data). This method is called after the
            dialog has been successfully accepted and validated.
        """
        return self.input_id, self.input_sequence