import os
import tempfile
from typing import Optional

from PyQt5.QtGui import QPixmap
from ete3 import Tree, TreeStyle, NodeStyle, faces


def render_tree(newick_string: str) -> Optional[QPixmap]:
    """Render a phylogenetic tree from a Newick string to a QPixmap.

    Args:
        newick_string (str): The Newick string representing the tree.

    Returns:
        Optional[QPixmap]: The rendered tree as a QPixmap, or None if rendering fails.
    """
    if not newick_string or newick_string.startswith("ERROR:"):
        return None

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

        _style_tree(t, colors)
        _add_leaf_faces(t)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_image_path = tmp.name

        t.render(temp_image_path, tree_style=ts, units="px")
        pixmap = QPixmap(temp_image_path)
        return pixmap if not pixmap.isNull() else None

    except Exception:
        return None
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except OSError:
                pass


def _style_tree(node, colors, color_idx=0):
    """Recursively style the tree nodes with different colors.

      Args:
          node: The current tree node.
          colors (list): List of color hex strings.
          color_idx (int, optional): Current color index. Defaults to 0.
      """
    nstyle = NodeStyle()
    nstyle["hz_line_color"] = colors[color_idx % len(colors)]
    nstyle["vt_line_color"] = colors[color_idx % len(colors)]
    nstyle["hz_line_width"] = 3
    nstyle["vt_line_width"] = 3
    node.set_style(nstyle)
    for i, child in enumerate(node.children):
        _style_tree(child, colors, color_idx + i + 1)


def _add_leaf_faces(tree):
    """Add text faces and styles to the leaves of the tree.

        Args:
            tree: The tree object whose leaves will be styled.
    """
    leaf_font_size = 12 if len(tree) <= 10 else 10
    for leaf in tree.iter_leaves():
        leaf_name_face = faces.TextFace(leaf.name, fsize=leaf_font_size, fgcolor="black")
        leaf_name_face.margin_left = 6
        leaf.add_face(leaf_name_face, column=0, position="aligned")
        leaf_style = NodeStyle()
        leaf_style["size"] = 5
        leaf_style["fgcolor"] = "#606060"
        leaf.set_style(leaf_style)
