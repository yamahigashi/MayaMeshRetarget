# pyright: reportCallIssue=false,reportArgumentType=false,reportAttributeAccessIssue=false,reportMissingImports=false
"""This module contains the UI for the application."""
from maya import cmds
from maya.app.general.mayaMixin import MayaQWidgetBaseMixin

from Qt.QtWidgets import (  # type: ignore
    QApplication,
    QWidget,
    QButtonGroup,
    QLineEdit,
    QListWidget,
    QPushButton,
    QLabel,
    QSlider,
    QHBoxLayout,
    QVBoxLayout,
    QRadioButton,
    QGroupBox,
    QSpacerItem,
    QSizePolicy,
    QAbstractItemView,
    QListWidgetItem,
    QScrollArea,
    QFrame,
    QGridLayout,
    QToolButton,
)

from Qt.QtCore import (
    Qt,
    QSize,
    Signal,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QAbstractAnimation,
)

from . import (
    logic,
    util,
    inpaint,
)

WINDOW_NAME = "MeshRetargetingToolWindow"
TITLE = "Mesh Retargeting Tool"
LABEL_WIDTH = 90

########################################################################################################################
class IntSlider(QWidget):
    def __init__(self, label, minimum=0, maximum=100, interval=5, initial_value=5):
        # type: (str, int, int, int, int) -> None

        super(IntSlider, self).__init__()

        self.sizePolicy().setHorizontalPolicy(QSizePolicy.MinimumExpanding)
        self.sizePolicy().setVerticalPolicy(QSizePolicy.MinimumExpanding)
        self.sizeHint()

        self.minimum = minimum
        self.maximum = maximum
        self.interval = interval

        self.initUI(label, initial_value)

    def initUI(self, label, initial_value):
        # Create the slider and the label
        self.label = QLabel(label, self)
        self.label.setFixedWidth(LABEL_WIDTH - 7)
        self.label.setAlignment(Qt.AlignRight)
        self.value_display = QLabel(str(initial_value), self)
        self.value_display.setFixedWidth(40)
        self.value_display.setAlignment(Qt.AlignRight)
        self.slider = QSlider(Qt.Horizontal, self)
        
        # Set the range of the slider
        self.slider.setMinimum(self.minimum)
        self.slider.setMaximum(self.maximum)
        self.slider.setTickInterval(self.interval)
        self.slider.setValue(initial_value)
        
        # Connect the valueChanged signal to the slot
        self.slider.valueChanged.connect(self.updateValueDisplay)

        # Create the layout and add the widgets
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_display)
        self.setLayout(layout)

    def sizeHint(self):
        # type: () -> QSize
        """Return the size hint of the widget."""
        return QSize(300, 30)

    def updateValueDisplay(self, value):
        self.value_display.setText(str(value))

    def value(self):
        return self.slider.value()

    def setValue(self, value):
        self.slider.setValue(value)


class FloatSlider(QWidget):
    def __init__(self, label, minimum=0.0, maximum=1.0, interval=0.05, step=0.001, initial_value=0.05):
        # type: (str, float, float, float, float, float) -> None

        super(FloatSlider, self).__init__()

        self.sizePolicy().setHorizontalPolicy(QSizePolicy.MinimumExpanding)
        self.sizePolicy().setVerticalPolicy(QSizePolicy.MinimumExpanding)
        self.sizeHint()

        self.minimum = minimum
        self.maximum = maximum
        self.interval = interval
        self.step = step
        self.value_multiplier = 1.0 / step

        self.initUI(label, initial_value)

    def initUI(self, label, initial_value):
        # Create the slider and the label
        self.label = QLabel(label, self)
        self.label.setFixedWidth(LABEL_WIDTH - 7)
        self.label.setAlignment(Qt.AlignRight)
        self.value_display = QLabel("{:.3f}".format(initial_value), self)
        self.value_display.setFixedWidth(40)
        self.value_display.setAlignment(Qt.AlignRight)
        self.slider = QSlider(Qt.Horizontal, self)
        
        # Set the range and step of the slider
        self.slider.setMinimum(self.minimum * self.value_multiplier)
        self.slider.setMaximum(self.maximum * self.value_multiplier)
        self.slider.setTickInterval(self.interval * self.value_multiplier)
        self.slider.setSingleStep(self.step * self.value_multiplier)
        self.slider.setValue(initial_value * self.value_multiplier)
        
        # Connect the valueChanged signal to the slot
        self.slider.valueChanged.connect(self.updateValueDisplay)

        # Create the layout and add the widgets
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_display)
        self.setLayout(layout)

    def sizeHint(self):
        # type: () -> QSize
        """Return the size hint of the widget."""
        return QSize(300, 30)

    def updateValueDisplay(self, value):
        # Calculate the float value based on the slider's integer value
        float_value = value / self.value_multiplier
        self.value_display.setText("{:.3f}".format(float_value))

    def value(self):
        # Get the current float value of the slider
        return self.slider.value() / self.value_multiplier

    def setValue(self, float_value):
        # Set the value of the slider using a float
        self.slider.setValue(int(float_value * self.value_multiplier))


class ToggleGroupBox(QWidget):
    def __init__(self, parent=None, title='', animationDuration=300):
        """
        References:
            # Adapted from c++ version
            http://stackoverflow.com/questions/32476006/how-to-make-an-expandable-collapsable-section-widget-in-qt
        """
        super(ToggleGroupBox, self).__init__()

        self.animationDuration = animationDuration
        self.toggleAnimation = QParallelAnimationGroup()
        self.contentArea = QScrollArea()
        self.headerLine = QFrame()
        self.toggleButton = QToolButton()
        self.mainLayout = QGridLayout()

        toggleButton = self.toggleButton
        toggleButton.setStyleSheet("QToolButton { border: none; }")
        toggleButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggleButton.setArrowType(Qt.RightArrow)
        toggleButton.setText(str(title))
        toggleButton.setCheckable(True)
        toggleButton.setChecked(False)

        headerLine = self.headerLine
        headerLine.setFrameShape(QFrame.HLine)
        headerLine.setFrameShadow(QFrame.Sunken)
        headerLine.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        self.contentArea.setStyleSheet("QScrollArea { background-color: white; border: none; }")
        self.contentArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # start out collapsed
        self.contentArea.setMaximumHeight(0)
        self.contentArea.setMinimumHeight(0)
        # let the entire widget grow and shrink with its content
        toggleAnimation = self.toggleAnimation
        toggleAnimation.addAnimation(QPropertyAnimation(self, b"minimumHeight"))
        toggleAnimation.addAnimation(QPropertyAnimation(self, b"maximumHeight"))
        toggleAnimation.addAnimation(QPropertyAnimation(self.contentArea, b"maximumHeight"))
        # don't waste space
        mainLayout = self.mainLayout
        mainLayout.setVerticalSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        row = 0
        mainLayout.addWidget(self.toggleButton, row, 0, 1, 1, Qt.AlignLeft)
        mainLayout.addWidget(self.headerLine, row, 2, 1, 1)
        row += 1
        mainLayout.addWidget(self.contentArea, row, 0, 1, 3)
        self.setLayout(self.mainLayout)

        def start_animation(checked):
            arrow_type = Qt.DownArrow if checked else Qt.RightArrow
            direction = QAbstractAnimation.Forward if checked else QAbstractAnimation.Backward
            toggleButton.setArrowType(arrow_type)
            self.toggleAnimation.setDirection(direction)
            self.toggleAnimation.start()

        self.toggleButton.clicked.connect(start_animation)

    def setContentLayout(self, contentLayout):
        # Not sure if this is equivalent to self.contentArea.destroy()
        self.contentArea.destroy()
        self.contentArea.setLayout(contentLayout)
        collapsedHeight = self.sizeHint().height() - self.contentArea.maximumHeight()
        contentHeight = contentLayout.sizeHint().height()
        for i in range(self.toggleAnimation.animationCount()-1):
            spoilerAnimation = self.toggleAnimation.animationAt(i)
            spoilerAnimation.setDuration(self.animationDuration)
            spoilerAnimation.setStartValue(collapsedHeight)
            spoilerAnimation.setEndValue(collapsedHeight + contentHeight)
        contentAnimation = self.toggleAnimation.animationAt(self.toggleAnimation.animationCount() - 1)
        contentAnimation.setDuration(self.animationDuration)
        contentAnimation.setStartValue(0)
        contentAnimation.setEndValue(contentHeight)

class ClickableLineEdit(QLineEdit):

    double_clicked = Signal()
    clicked = Signal()

    def __init__(self, parent=None):
        super(ClickableLineEdit, self).__init__(parent)

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        super(ClickableLineEdit, self).mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super(ClickableLineEdit, self).mousePressEvent(event)


class MeshRetargetingToolWindow(MayaQWidgetBaseMixin, QWidget):

    def __init__(self,
                 parent=None,
                 src=None,
                 dst=None,
                 meshes=None,
                 inpaint=False,
                 inpaint_mode="distance",
                 distance=0.1,
                 angle=180.0,
                 sampling_stride=1,
                 apply_rigid_transform=True
    ):
        # type: (QWidget|None, str|None, str|None, list[str]|None, bool, str, float, float, int, bool) -> None
        super(MeshRetargetingToolWindow, self).__init__(parent)

        self.initUI()
        if src:
            self.src_line_edit.setText(src)
        if dst:
            self.dst_line_edit.setText(dst)
        if meshes:
            for mesh in meshes:
                self.ret_list_widget.addItem(mesh)
        if inpaint:
            self.inpaint_on.setChecked(True)
        else:
            self.inpaint_off.setChecked(True)
        if inpaint_mode == "distance":
            self.inpaint_mode_dist.setChecked(True)
        else:
            self.inpaint_mode_selection.setChecked(True)
        self.dist_slider.setValue(distance)
        self.angle_slider.setValue(angle)
        self.stride_slider.setValue(sampling_stride)
        if apply_rigid_transform:
            self.rigid_on.setChecked(True)
        else:
            self.rigid_off.setChecked

        isReady = self.checkToExecute()
        self.execute_button.setEnabled(isReady)

    def initUI(self):
        # type: () -> None

        # -----------------------------------------------
        # source and destination meshes
        self.meshes_group_box = QGroupBox("Meshes", self)
        self.src_label = QLabel("Source:", self)
        self.src_label.setAlignment(Qt.AlignRight)
        self.src_label.setFixedWidth(LABEL_WIDTH)
        # self.src_line_edit = QLineEdit(self)
        self.src_line_edit = ClickableLineEdit(self)
        self.src_line_edit.setReadOnly(True)
        self.src_line_edit.setFocusPolicy(Qt.NoFocus)
        # self.src_line_edit.double_clicked.connect(self.selectSourceMesh)
        self.src_line_edit.clicked.connect(self.selectSourceMesh)

        self.src_button = QPushButton("set", self)
        self.src_button.clicked.connect(self.setMesh)
        self.src_button.setFixedWidth(50)
        
        self.dst_label = QLabel("Target:", self)
        self.dst_label.setAlignment(Qt.AlignRight)
        self.dst_label.setFixedWidth(LABEL_WIDTH)
        # self.dst_line_edit = QLineEdit(self)
        self.dst_line_edit = ClickableLineEdit(self)
        self.dst_line_edit.setReadOnly(True)
        self.dst_line_edit.setFocusPolicy(Qt.NoFocus)
        # self.dst_line_edit.double_clicked.connect(self.selectTargetMesh)
        self.dst_line_edit.clicked.connect(self.selectTargetMesh)
        self.dst_button = QPushButton("set", self)
        self.dst_button.clicked.connect(self.setMesh)
        self.dst_button.setFixedWidth(50)

        self.ret_label = QLabel("To Retarget:", self)
        self.ret_label.setAlignment(Qt.AlignRight)
        self.ret_label.setFixedWidth(LABEL_WIDTH)
        self.ret_list_widget = QListWidget(self)
        # self.ret_list_widget.setReadOnly(True)
        self.ret_list_widget.setFocusPolicy(Qt.NoFocus)
        self.ret_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.ret_add_button = QPushButton("set", self)
        self.ret_add_button.clicked.connect(self.addRetargetMesh)
        self.ret_add_button.setFixedWidth(50)
        self.ret_remove_button = QPushButton("remove", self)
        self.ret_remove_button.clicked.connect(self.removeSelectedMeshes)
        self.ret_remove_button.setFixedWidth(50)
        self.ret_clear_button = QPushButton("clear", self)
        self.ret_clear_button.clicked.connect(self.clearSelectedMeshes)
        self.ret_clear_button.setFixedWidth(50)
        self.ret_list_widget.itemClicked.connect(self.selectRetargetMesh)

        # -----------------------------------------------
        # search settings
        self.settings_group_box = QGroupBox("Settings", self)

        self.rigid_mode_label = QLabel("Maintain Rigid:", self)
        self.rigid_mode_label.setAlignment(Qt.AlignRight)
        self.rigid_mode_label.setFixedWidth(LABEL_WIDTH)
        self.rigid_on = QRadioButton("On", self)
        self.rigid_on.toggled.connect(self.rigidModeToggled)
        self.rigid_on.setChecked(True)
        self.rigid_off = QRadioButton("Off", self)
        self.rigid_off.toggled.connect(self.rigidModeToggled)
        self.rigid_button_group = QButtonGroup(self.settings_group_box)
        self.rigid_button_group.addButton(self.rigid_on)
        self.rigid_button_group.addButton(self.rigid_off)

        self.inpaint_onoff_label = QLabel("Inpaint:", self)
        self.inpaint_onoff_label.setAlignment(Qt.AlignRight)
        self.inpaint_onoff_label.setFixedWidth(LABEL_WIDTH)
        self.inpaint_on = QRadioButton("On", self)
        self.inpaint_on.toggled.connect(self.inpaintOnOffToggled)
        self.inpaint_off = QRadioButton("Off", self)
        self.inpaint_off.toggled.connect(self.inpaintOnOffToggled)
        self.inpaint_button_group = QButtonGroup(self.settings_group_box)
        self.inpaint_button_group.addButton(self.inpaint_on)
        self.inpaint_button_group.addButton(self.inpaint_off)

        self.stride_slider = IntSlider("Sampling Stride:", minimum=1, maximum=100, interval=1, initial_value=1)

        # -----------------------------------------------
        self.inpaint_settings_group_box = QGroupBox("Inpaint Settings")
        self.inpaint_mode_label = QLabel("Mode:", self)
        self.inpaint_mode_label.setFixedWidth(LABEL_WIDTH + 4)
        self.inpaint_mode_label.setAlignment(Qt.AlignRight)
        self.inpaint_mode_dist = QRadioButton("Distance and Angle", self)
        self.inpaint_mode_dist.setChecked(True)
        self.inpaint_mode_dist.toggled.connect(self.inpaintModeToggled)
        self.inpaint_mode_selection = QRadioButton("Selection", self)
        self.inpaint_mode_selection.toggled.connect(self.inpaintModeToggled)
        self.inpaint_button_group2 = QButtonGroup(self.settings_group_box)
        self.inpaint_button_group2.addButton(self.inpaint_mode_dist)
        self.inpaint_button_group2.addButton(self.inpaint_mode_selection)
        self.dist_slider = FloatSlider("distance:", minimum=0.000001, maximum=0.3, interval=0.01, step=0.001, initial_value=0.1)
        self.angle_slider = FloatSlider("angle:", minimum=0.0, maximum=180.0, interval=1.0, step=0.5, initial_value=180.0)

        # -----------------------------------------------
        self.utility_group_box = QGroupBox("Utility")
        self.utility_group_box.setCheckable(True)
        self.utility_group_box.setChecked(False)
        self.restructure_button = QPushButton("Restructure Selected", self)
        self.restructure_button.clicked.connect(self.restructureButtonClicked)
        self.select_inpaint_area_button = QPushButton("Select inpaint area", self)
        self.select_inpaint_area_button.clicked.connect(self.selectInpaintArea)

        # -----------------------------------------------
        self.execute_button = QPushButton("Execute!", self)
        self.execute_button.clicked.connect(self.executeButtonClicked)
        self.execute_button.setEnabled(False)

        # -----------------------------------------------
        # Create layouts
        src_layout = QHBoxLayout()
        src_layout.addWidget(self.src_label)
        src_layout.addWidget(self.src_line_edit)
        src_layout.addWidget(self.src_button)

        dst_layout = QHBoxLayout()
        dst_layout.addWidget(self.dst_label)
        dst_layout.addWidget(self.dst_line_edit)
        dst_layout.addWidget(self.dst_button)

        ret_layout = QHBoxLayout()
        ret_layout.addWidget(self.ret_label)
        ret_layout.addWidget(self.ret_list_widget)
        ret_buttons_layout = QVBoxLayout()
        ret_buttons_layout.addWidget(self.ret_add_button)
        ret_buttons_layout.addWidget(self.ret_remove_button)
        ret_buttons_layout.addWidget(self.ret_clear_button)
        ret_buttons_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        ret_layout.addLayout(ret_buttons_layout)

        meshes_group_box_layout = QVBoxLayout()
        meshes_group_box_layout.addLayout(src_layout)
        meshes_group_box_layout.addLayout(dst_layout)
        meshes_group_box_layout.addItem(QSpacerItem(0, 13, QSizePolicy.Minimum, QSizePolicy.Minimum))
        meshes_group_box_layout.addLayout(ret_layout)
        self.meshes_group_box.setLayout(meshes_group_box_layout)

        rigid_mode_layout = QHBoxLayout()
        rigid_mode_layout.addWidget(self.rigid_mode_label)
        rigid_mode_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        rigid_mode_layout.addWidget(self.rigid_on)
        rigid_mode_layout.addWidget(self.rigid_off)
        rigid_mode_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        inpaint_onoff_layout = QHBoxLayout()
        inpaint_onoff_layout.addWidget(self.inpaint_onoff_label)
        inpaint_onoff_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        inpaint_onoff_layout.addWidget(self.inpaint_on)
        inpaint_onoff_layout.addWidget(self.inpaint_off)
        inpaint_onoff_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        stride_layout = QHBoxLayout()
        stride_layout.addWidget(self.stride_slider)

        search_group_box_layout = QVBoxLayout()
        search_group_box_layout.addLayout(rigid_mode_layout)
        search_group_box_layout.addLayout(inpaint_onoff_layout)
        search_group_box_layout.addLayout(stride_layout)
        self.settings_group_box.setLayout(search_group_box_layout)

        inpaint_mode_layout = QHBoxLayout()
        inpaint_mode_layout.addWidget(self.inpaint_mode_label)
        inpaint_mode_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        inpaint_mode_layout.addWidget(self.inpaint_mode_dist)
        inpaint_mode_layout.addWidget(self.inpaint_mode_selection)
        inpaint_mode_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(self.dist_slider)
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(self.angle_slider)

        inpaint_settings_group_box_layout = QVBoxLayout()
        inpaint_settings_group_box_layout.addLayout(inpaint_mode_layout)
        inpaint_settings_group_box_layout.addLayout(dist_layout)
        inpaint_settings_group_box_layout.addLayout(angle_layout)
        self.inpaint_settings_group_box.setLayout(inpaint_settings_group_box_layout)

        utility_group_box_layout = QHBoxLayout()
        utility_group_box_layout.addWidget(self.restructure_button)
        utility_group_box_layout.addWidget(self.select_inpaint_area_button)
        self.utility_group_box.setLayout(utility_group_box_layout)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.execute_button)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.meshes_group_box)
        main_layout.addWidget(self.settings_group_box)
        main_layout.addWidget(self.inpaint_settings_group_box)
        main_layout.addWidget(self.utility_group_box)
        # main_layout.addLayout(count_layout)
        spacer = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        main_layout.addItem(spacer)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

        # Set window properties
        self.setGeometry(300, 300, 450, 250)
        self.setWindowTitle(TITLE)
        self.show()

    def setMesh(self):
        # type: () -> None
        """Insert text into the line edit."""

        selection = cmds.ls(sl=True, objectsOnly=True)
        if not selection:
            cmds.warning("Nothing is selected")
            return

        sel = selection[0]

        sender = self.sender()
        if sender == self.src_button:

            if self.src_line_edit.text() != sel:
                self.clear()

            self.src_line_edit.setText(sel)

        elif sender == self.dst_button:

            if self.dst_line_edit.text() != sel:
                self.clear()

            self.dst_line_edit.setText(sel)

        isReady = self.checkToExecute()
        self.execute_button.setEnabled(isReady)

    def selectSourceMesh(self):
        # type: () -> None
        """Select the source mesh."""

        self.selectMesh(self.src_line_edit.text())

    def selectTargetMesh(self):
        # type: () -> None
        """Select the target mesh."""

        self.selectMesh(self.dst_line_edit.text())

    def selectRetargetMesh(self, item):
        # type: (QListWidgetItem) -> None
        """Select the target mesh."""
        if isinstance(item, QListWidgetItem):
            name = item.text()
        else:
            name = item
        self.selectMesh(name)

    def selectMesh(self, mesh):
        # type: (str) -> None
        """Select the mesh from the line edit."""

        if not mesh:
            cmds.warning("No mesh selected.")
            return

        shift = QApplication.keyboardModifiers() == Qt.ShiftModifier
        ctrl = QApplication.keyboardModifiers() == Qt.ControlModifier

        if shift and ctrl:
            cmds.select(mesh, toggle=True, add=True)

        elif shift:
            cmds.select(mesh, add=True)

        elif ctrl:
            cmds.select(mesh, toggle=True)

        else:
            cmds.select(mesh)

    def addRetargetMesh(self):
        # type: () -> None
        """Add a mesh to the list of meshes to retarget."""

        selected = cmds.ls(selection=True, type="transform")
        if not selected:
            cmds.warning("No meshes selected to add.")
            return

        for mesh in selected:
            if not self.hasMesh(mesh):
                continue

            if not self.ret_list_widget.findItems(mesh, Qt.MatchExactly):
                self.ret_list_widget.addItem(mesh)

        isReady = self.checkToExecute()
        self.execute_button.setEnabled(isReady)

    def removeSelectedMeshes(self):
        # リスト内で選択されているメッシュを削除
        selected_items = self.ret_list_widget.selectedItems()
        if not selected_items:
            cmds.warning("No meshes selected to remove.")
            return

        for item in selected_items:
            self.ret_list_widget.takeItem(self.ret_list_widget.row(item))

        isReady = self.checkToExecute()
        self.execute_button.setEnabled(isReady)

    def clearSelectedMeshes(self):
        # リスト内の全てのメッシュを削除
        self.ret_list_widget.clear()
        isReady = self.checkToExecute()
        self.execute_button.setEnabled(isReady)

    def clear(self):
        # type: () -> None
        pass

    def hasMesh(self, mesh):
        # type: (str) -> bool
        """Check if the mesh exists in the scene."""
        m = cmds.listRelatives(mesh, children=True, shapes=True)
        if not m:
            return False

        return True

    def updateValueDisplay(self):
        # type: () -> None
        """Update the value display label."""

        sender = self.sender()
        if sender == self.dist_slider:
            self.dist_value_display.setText(str(self.dist_slider.value()))

        elif sender == self.angle_slider:
            self.angle_value_display.setText(str(self.angle_slider.value()))

    def rigidModeToggled(self):
        if self.rigid_on.isChecked():
            pass
        elif self.rigid_off.isChecked():
            pass

    def inpaintModeToggled(self):
        if self.inpaint_mode_dist.isChecked():
            self.dist_slider.setEnabled(True)
            self.angle_slider.setEnabled(True)

        elif self.inpaint_mode_selection.isChecked():
            self.dist_slider.setEnabled(False)
            self.angle_slider.setEnabled(False)

    def inpaintOnOffToggled(self):
        if self.inpaint_on.isChecked():
            self.inpaint_settings_group_box.setEnabled(True)
        elif self.inpaint_off.isChecked():
            self.inpaint_settings_group_box.setEnabled(False)

    def restructureButtonClicked(self):
        # type: () -> None
        """Restructure the selected meshes."""
        suffixs = []
        if self.rigid_on.isChecked():
            suffixs.append("rigid")
        if self.inpaint_on.isChecked():
            suffixs.append("inpaint")
        util.restructure_meshes_hierarchy(suffix="_".join(suffixs))

    def selectInpaintArea(self):
        # type: () -> None
        """Restructure the selected meshes."""
        src = self.src_line_edit.text()
        dsts = [self.ret_list_widget.item(i).text() for i in range(self.ret_list_widget.count())]
        dist = self.dist_slider.value()
        angle = self.angle_slider.value()

        if not src:
            cmds.warning("Please set the source mesh.")
            return

        if not dsts:
            cmds.warning("Please set the target mesh.")
            return

        inpaint.select_inpaint_area(src, dsts, dist, angle)

    def checkToExecute(self):
        # type: () -> bool
        """Check if the tool is ready to execute."""
        if not self.src_line_edit.text():
            return False

        if not self.dst_line_edit.text():
            return False

        if self.ret_list_widget.count() == 0:
            return False

        # check vertex count
        src_vertex_count = cmds.polyEvaluate(self.src_line_edit.text(), vertex=True)
        dst_vertex_count = cmds.polyEvaluate(self.dst_line_edit.text(), vertex=True)
        if src_vertex_count != dst_vertex_count:
            cmds.warning("Source and target meshes have different vertex counts.")
            return False

        return True

    def executeButtonClicked(self):
        # type: () -> None
        """Search for vertices to transfer weights from."""

        if not self.checkToExecute():
            cmds.warning("Please set source, target, and retarget meshes.")
            return

        src = self.src_line_edit.text()
        dst = self.dst_line_edit.text()
        retarget_meshes = [self.ret_list_widget.item(i).text() for i in range(self.ret_list_widget.count())]
        radius_coeff = self.dist_slider.value()
        angle = self.angle_slider.value()
        sampling_stride = 10
        apply_rigid_transform = self.rigid_on.isChecked()
        inpaint = self.inpaint_on.isChecked()

        meshes = logic.retarget(
            source=src,
            target=dst,
            meshes=retarget_meshes,
            # kernel=kernel_name,
            radius_coefficient=radius_coeff,
            angle=angle,
            sampling_stride=sampling_stride,
            apply_rigid_transform=apply_rigid_transform,
            inpaint=inpaint
        )

        cmds.select(meshes)


def show_ui():
    # type: () -> None
    """Show the UI."""
    global MAIN_WIDGET

    src = None
    dst = None
    meshes = None
    inpaint = False
    inpaint_mode = "distance"
    distance = 0.1
    angle = 180.0
    sampling_stride = 1
    apply_rigid_transform = True
    pos = None

    # close all previous windows
    all_widgets = {w.objectName(): w for w in QApplication.allWidgets()}
    for k, v in all_widgets.items():
        if v.__class__.__name__ == WINDOW_NAME:

            src = v.src_line_edit.text()
            dst = v.dst_line_edit.text()
            meshes = [v.ret_list_widget.item(i).text() for i in range(v.ret_list_widget.count())]
            inpaint = v.inpaint_on.isChecked()
            inpaint_mode = "distance" if v.inpaint_mode_dist.isChecked() else "selection"
            distance = v.dist_slider.value()
            angle = v.angle_slider.value()
            sampling_stride = v.stride_slider.value()
            apply_rigid_transform = v.rigid_on.isChecked()
            pos = v.pos()

            v.close()
            v.deleteLater()

    main_widget = MeshRetargetingToolWindow(
        src=src,
        dst=dst,
        meshes=meshes,
        inpaint=inpaint,
        inpaint_mode=inpaint_mode,
        distance=distance,
        angle=angle,
        sampling_stride=sampling_stride,
        apply_rigid_transform=apply_rigid_transform
    )
    main_widget.show()
    if pos:
        main_widget.move(pos)

    MAIN_WIDGET = main_widget
