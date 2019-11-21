from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore

from PIL import Image
import os


def explore(pool, in_app=False):
    if not in_app:
        app = QtWidgets.QApplication([])
    else:
        app = None
    dialog = QtWidgets.QWidget()
    dialog.setWindowTitle(f"Pool Explorer ({len(pool):d} samples)")

    current_sample = None
    current_resource_ids = None
    current_resource = None

    # Widgets
    sample_list = QtWidgets.QListWidget()
    resource_id_list = QtWidgets.QListWidget()
    history_list = QtWidgets.QListWidget()
    delete_button = QtWidgets.QPushButton("delete")
    image_preview = QtWidgets.QLabel()

    # Update functions
    def update_sample_list():
        sample_list.clear()
        sample_list.clearSelection()
        for sample in pool:
            sample_list.addItem(str(sample.get_resource("uid")))

    def sample_list_item_clicked(currentRow):
        nonlocal current_sample
        if currentRow >= 0:
            selected_sample = currentRow
            current_sample = pool[currentRow]
            update_resource_id_list()
            resource_id_list.setCurrentRow(0)
            update_history_list()

    def update_resource_id_list():
        nonlocal current_sample
        nonlocal current_resource_ids

        current_resource_ids = list(sorted(current_sample.get_resource_ids()))

        resource_id_list.clear()
        resource_id_list.clearSelection()
        for resource_id in current_resource_ids:

            resource = current_sample.get_resource(resource_id)

            if resource_id == "uid":
                resource_id_list.addItem(f"UID: {resource}")
                continue

            if hasattr(resource, "shape"):
                resource_id_list.addItem(f"{resource_id} (preview)")
            elif hasattr(resource, "__str__"):
                resource_id_list.addItem(f"{resource_id} ({resource})")
            else:
                resource_id_list.addItem(f"{resource_id} (Unknown)")

    def update_history_list():
        nonlocal current_sample
        history_list.clear()
        history_list.clearSelection()
        for item in current_sample.history:
            history_list.addItem(str(item))

    def resource_id_list_item_clicked(currentRow):
        nonlocal current_sample
        nonlocal current_resource
        nonlocal current_resource_ids

        if currentRow >= 0:
            resource_id = current_resource_ids[currentRow]
            current_resource = current_sample.get_resource(resource_id)
            update_image_preview()

    def update_image_preview():
        nonlocal current_resource

        image_preview.clear()
        if current_resource is None:
            return

        if hasattr(current_resource, "shape") and hasattr(current_resource, "tobytes"):
            if len(current_resource.shape) == 3:
                try:
                    assert current_resource.shape[2] == 3
                    image_preview.setPixmap(
                        QtGui.QPixmap(
                            QtGui.QImage(
                                current_resource.tobytes(),
                                current_resource.shape[1],
                                current_resource.shape[0],
                                3 * current_resource.shape[1],
                                QtGui.QImage.Format_RGB888,
                            )
                        ).scaled(
                            image_preview.width(),
                            image_preview.height(),
                            QtCore.Qt.KeepAspectRatio,
                            QtCore.Qt.SmoothTransformation,
                        )
                    )
                except Exception as ex:
                    image_preview.setText(f"Error during preview creation: {ex}")
            else:
                image_preview.setText("Unsupported image data.")
        elif type(current_resource) is Image.Image:
            try:
                assert current_resource.mode == "RGB"
                image_preview.setPixmap(
                    QtGui.QPixmap(
                        QtGui.QImage(
                            current_resource.tobytes(),
                            current_resource.width,
                            current_resource.height,
                            3 * current_resource.width,
                            QtGui.QImage.Format_RGB888,
                        )
                    ).scaled(
                        image_preview.width(),
                        image_preview.height(),
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )
                )
            except Exception as ex:
                image_preview.setText(f"Error during preview creation: {ex}")
        elif type(current_resource) is str and os.path.exists(current_resource):
            image_preview.setPixmap(
                QtGui.QPixmap(QtGui.QImage(current_resource)).scaled(
                    image_preview.width(),
                    image_preview.height(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )
        else:
            image_preview.setText("No image data.")

    def delete_button_clicked():
        nonlocal current_sample
        if current_sample is not None:
            pool.remove(current_sample)

        update_sample_list()

    # Events
    sample_list.currentRowChanged.connect(sample_list_item_clicked)
    resource_id_list.currentRowChanged.connect(resource_id_list_item_clicked)
    delete_button.clicked.connect(delete_button_clicked)

    dialog.resizeEvent = lambda newSize: (
        QtWidgets.QWidget.resizeEvent(dialog, newSize),
        update_image_preview(),
    )[0]

    # Layout
    layout = QtWidgets.QHBoxLayout()

    layout.addWidget(sample_list)

    middle_widget = QtWidgets.QWidget()
    middle_layout = QtWidgets.QVBoxLayout()
    middle_widget.setLayout(middle_layout)
    middle_layout.addWidget(resource_id_list)
    middle_layout.addWidget(history_list)
    middle_layout.addWidget(delete_button)
    layout.addWidget(middle_widget)

    layout.addWidget(image_preview)

    update_sample_list()

    dialog.setLayout(layout)

    sample_list.setMinimumWidth(150)
    sample_list.setMaximumWidth(250)
    middle_widget.setMinimumWidth(150)
    middle_widget.setMaximumWidth(250)
    image_preview.setMinimumWidth(150)

    # Go!
    dialog.show()
    if not in_app:
        app.exec_()
