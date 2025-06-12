from PyQt5.QtWidgets import QTableWidget, QMessageBox, QFileDialog


def save_table_data(table: QTableWidget, table_name: str, default_filename: str, parent) -> bool:
    """Save the data from a QTableWidget to a file in CSV or text matrix format.

    Args:
        table (QTableWidget): The table widget containing the data to save.
        table_name (str): The name of the table, used in the save dialog.
        default_filename (str): The default filename suggested in the save dialog.
        parent: The parent widget for dialogs.

    Returns:
        bool: True if the data was saved successfully, False otherwise.
    """
    if table.rowCount() == 0:
        QMessageBox.warning(parent, "Warning", "No data to save.")
        return False

    filter_str = "CSV (Comma-separated) (*.csv);;Text Matrix (*.txt);;All Files (*)"
    fp, selected_filter = (QFileDialog.getSaveFileName(
        parent, f"Save {table_name}", f"output/{default_filename}", filter_str
    ))

    if not fp:
        return False

    try:
        with open(fp, 'w', newline='') as f:
            col_headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
            row_headers = [table.verticalHeaderItem(i).text() for i in range(table.rowCount())]

            if 'CSV' in selected_filter:
                f.write(',' + ','.join(col_headers) + '\n')
                for r in range(table.rowCount()):
                    row_data = [table.item(r, c).text() for c in range(table.columnCount())]
                    f.write(row_headers[r] + ',' + ','.join(row_data) + '\n')
            else:
                col_widths = [len(h) for h in col_headers]
                max_row_header_width = max(len(h) for h in row_headers) if row_headers else 0

                for c in range(table.columnCount()):
                    max_col_content = max((len(table.item(r, c).text()) for r in range(table.rowCount())),
                                          default=0)
                    col_widths[c] = max(col_widths[c], max_col_content)

                f.write(' ' * (max_row_header_width + 2))
                for c, header in enumerate(col_headers):
                    f.write(header.ljust(col_widths[c] + 2))
                f.write('\n')

                for r in range(table.rowCount()):
                    f.write(row_headers[r].ljust(max_row_header_width + 2))
                    for c in range(table.columnCount()):
                        item_text = table.item(r, c).text()
                        f.write(item_text.ljust(col_widths[c] + 2))
                    f.write('\n')

        return True
    except Exception as e:
        QMessageBox.critical(parent, "Save Error", f"Could not save file:\n{e}")
        return False


def save_msa_data(table: QTableWidget, msa_type: str, parent) -> bool:
    """Save MSA (Multiple Sequence Alignment) data from a QTableWidget to a file in FASTA or text format.

    Args:
        table (QTableWidget): The table widget containing the MSA data.
        msa_type (str): The type of MSA (used for dialog and filename).
        parent: The parent widget for dialogs.

    Returns:
        bool: True if the data was saved successfully, False otherwise.
    """
    if not table or table.rowCount() == 0:
        QMessageBox.warning(parent, "Warning", f"No {msa_type} data to save.")
        return False

    default_filename = f"{msa_type.lower().replace(' ', '_')}.fasta"
    filter_str = "FASTA Format (*.fa *.fasta);;Text Matrix (*.txt);;All Files (*)"
    fp, selected_filter = QFileDialog.getSaveFileName(
        parent, f"Save {msa_type}", f"output/{default_filename}", filter_str
    )

    if not fp:
        return False

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

        return True
    except Exception as e:
        QMessageBox.critical(parent, "Save Error", f"Could not save file:\n{e}")
        return False
