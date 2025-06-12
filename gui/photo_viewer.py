from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QGraphicsView


class PhotoViewer(QGraphicsView):
    """A custom QGraphicsView that handles zooming and panning.

    Inherits from:
        QGraphicsView: Provides a widget for displaying the contents of a QGraphicsScene.

    Methods:
        zoom_in(): Zooms into the view by scaling up.
        zoom_out(): Zooms out of the view by scaling down.
        reset_view(): Resets the view to fit the scene's bounding rectangle.
    """

    def __init__(self, parent=None):
        """Initializes the PhotoViewer.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

    def zoom_in(self):
        """Zooms into the view by scaling up."""
        self.scale(1.25, 1.25)

    def zoom_out(self):
        """Zooms out of the view by scaling down."""
        self.scale(0.8, 0.8)

    def reset_view(self):
        """Resets the view to fit the scene's bounding rectangle, keeping aspect ratio."""
        if self.scene() and self.scene().items():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
