import sys

from PyQt5.QtWidgets import QApplication

from gui.upgma_app import UPGMAApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = UPGMAApp()
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec_())
