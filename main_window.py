from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _configure_qt_runtime_paths() -> None:
    spec = importlib.util.find_spec("PySide6")
    if spec is None or spec.origin is None:
        return

    pyside_dir = Path(spec.origin).resolve().parent
    plugins_dir = pyside_dir / "plugins"
    platforms_dir = plugins_dir / "platforms"

    if platforms_dir.exists():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)

    if plugins_dir.exists():
        existing_plugin_path = os.environ.get("QT_PLUGIN_PATH", "")
        plugin_parts = [str(plugins_dir)]
        if existing_plugin_path:
            plugin_parts.append(existing_plugin_path)
        os.environ["QT_PLUGIN_PATH"] = os.pathsep.join(plugin_parts)

    existing_path = os.environ.get("PATH", "")
    path_parts = [str(pyside_dir)]
    if existing_path:
        path_parts.append(existing_path)
    os.environ["PATH"] = os.pathsep.join(path_parts)


_configure_qt_runtime_paths()

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from calibration_core import (
    CalibrationError,
    CalibrationProject,
    ControlPoint,
    UnitSpec,
    attach_calibration,
    create_project,
    export_api_ready_config,
    load_project_json,
    save_calibration_json,
    save_project_json,
    validate_reference_resolution,
)
from video_canvas import CanvasPoint, VideoCanvas


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Ground-Plane Calibration Tool")
        self.resize(1500, 900)

        self.canvas = VideoCanvas(self)
        self.project: Optional[CalibrationProject] = None
        self.points: list[ControlPoint] = []

        self.current_source_path: str = ""
        self.current_source_type: Optional[str] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_frame_index = 0
        self.video_frame_count = 0
        self.video_fps = 0.0
        self.video_rotation_degrees = 0
        self._raw_video_width: int = 0    # raw OpenCV frame width before any rotation
        self._raw_video_height: int = 0   # raw OpenCV frame height before any rotation
        self.video_capture: Optional[cv2.VideoCapture] = None

        self._table_sync_in_progress = False
        self._frame_sync_in_progress = False
        self._selected_point_id: Optional[str] = None

        self._build_ui()
        self._connect_signals()
        self._set_video_controls_enabled(False)
        self._update_window_state()

    def closeEvent(self, event) -> None:
        self._release_video_capture()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(8, 8, 8, 8)
        central_layout.setSpacing(8)

        source_bar = QHBoxLayout()
        self.open_image_button = QPushButton("Open Image")
        self.open_video_button = QPushButton("Open Video")
        self.use_current_frame_button = QPushButton("Use Current Frame")
        self.current_source_label = QLabel("No source loaded")
        self.current_source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        source_bar.addWidget(self.open_image_button)
        source_bar.addWidget(self.open_video_button)
        source_bar.addWidget(self.use_current_frame_button)
        source_bar.addSpacing(12)
        source_bar.addWidget(self.current_source_label, 1)
        central_layout.addLayout(source_bar)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        central_layout.addWidget(splitter, 1)

        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(self.canvas, 1)

        video_controls_group = QGroupBox("Video Controls", self)
        video_controls_layout = QHBoxLayout(video_controls_group)
        self.prev_frame_button = QPushButton("Previous Frame")
        self.next_frame_button = QPushButton("Next Frame")
        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_slider.setTracking(True)
        self.frame_index_spin = QSpinBox(self)
        self.frame_index_spin.setMinimum(0)
        self.frame_count_label = QLabel("/ 0")
        self.frame_timestamp_label = QLabel("00:00:00.000")

        video_controls_layout.addWidget(self.prev_frame_button)
        video_controls_layout.addWidget(self.frame_slider, 1)
        video_controls_layout.addWidget(self.next_frame_button)
        video_controls_layout.addWidget(QLabel("Frame"))
        video_controls_layout.addWidget(self.frame_index_spin)
        video_controls_layout.addWidget(self.frame_count_label)
        video_controls_layout.addSpacing(12)
        video_controls_layout.addWidget(QLabel("Time"))
        video_controls_layout.addWidget(self.frame_timestamp_label)
        video_controls_layout.addSpacing(16)
        video_controls_layout.addWidget(QLabel("Rotation"))
        self.rotation_combo = QComboBox(self)
        self.rotation_combo.addItems(["0°", "90° CW", "180°", "270° CW"])
        self.rotation_combo.setFixedWidth(88)
        self.rotation_combo.setToolTip(
            "Manually rotate the video frame for display and coordinate mapping.\n"
            "Existing control points are automatically remapped when rotation changes."
        )
        video_controls_layout.addWidget(self.rotation_combo)
        left_layout.addWidget(video_controls_group)

        splitter.addWidget(left_panel)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        unit_group = QGroupBox("Units", self)
        unit_layout = QFormLayout(unit_group)
        self.unit_combo = QComboBox(self)
        self.unit_combo.addItems(["meters", "centimeters", "millimeters", "feet", "custom"])
        self.unit_symbol_edit = QLineEdit(self)
        self.unit_symbol_edit.setPlaceholderText("Optional symbol, e.g. m or ft")
        self.custom_unit_edit = QLineEdit(self)
        self.custom_unit_edit.setPlaceholderText("Custom unit text")
        unit_layout.addRow("Unit", self.unit_combo)
        unit_layout.addRow("Symbol", self.unit_symbol_edit)
        unit_layout.addRow("Custom", self.custom_unit_edit)
        right_layout.addWidget(unit_group)

        self.point_table = QTableWidget(0, 9, self)
        self.point_table.setHorizontalHeaderLabels(
            [
                "Use",
                "Label",
                "Img X",
                "Img Y",
                "World X",
                "World Y",
                "Note",
                "Err (px)",
                "Err (world)",
            ]
        )
        self.point_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.point_table.setSelectionMode(QTableWidget.SingleSelection)
        self.point_table.setAlternatingRowColors(True)
        self.point_table.verticalHeader().setVisible(False)
        header = self.point_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)
        right_layout.addWidget(self.point_table, 1)

        point_buttons_layout = QHBoxLayout()
        self.delete_point_button = QPushButton("Delete Selected Point")
        self.clear_points_button = QPushButton("Clear Points")
        point_buttons_layout.addWidget(self.delete_point_button)
        point_buttons_layout.addWidget(self.clear_points_button)
        right_layout.addLayout(point_buttons_layout)

        action_group = QGroupBox("Calibration Actions", self)
        action_layout = QVBoxLayout(action_group)
        grid_layout = QHBoxLayout()
        self.grid_interval_spin = QDoubleSpinBox(self)
        self.grid_interval_spin.setDecimals(3)
        self.grid_interval_spin.setRange(0.001, 1_000_000_000.0)
        self.grid_interval_spin.setSingleStep(0.5)
        self.grid_interval_spin.setValue(1.0)
        self.grid_interval_spin.setSuffix(" units")
        grid_layout.addWidget(QLabel("Grid Interval"))
        grid_layout.addWidget(self.grid_interval_spin, 1)
        action_layout.addLayout(grid_layout)
        self.compute_button = QPushButton("Compute Calibration")
        self.save_project_button = QPushButton("Save Project")
        self.load_project_button = QPushButton("Load Project")
        self.export_json_button = QPushButton("Export Calibration JSON")
        action_layout.addWidget(self.compute_button)
        action_layout.addWidget(self.save_project_button)
        action_layout.addWidget(self.load_project_button)
        action_layout.addWidget(self.export_json_button)
        right_layout.addWidget(action_group)

        metrics_group = QGroupBox("Calibration Summary", self)
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics_text = QTextEdit(self)
        self.metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_text)
        right_layout.addWidget(metrics_group, 1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 3)

        self.setCentralWidget(central_widget)

        file_menu = self.menuBar().addMenu("File")
        open_image_action = QAction("Open Image", self)
        open_video_action = QAction("Open Video", self)
        save_project_action = QAction("Save Project", self)
        load_project_action = QAction("Load Project", self)
        export_action = QAction("Export Calibration JSON", self)
        file_menu.addAction(open_image_action)
        file_menu.addAction(open_video_action)
        file_menu.addSeparator()
        file_menu.addAction(save_project_action)
        file_menu.addAction(load_project_action)
        file_menu.addAction(export_action)

        open_image_action.triggered.connect(self.open_image)
        open_video_action.triggered.connect(self.open_video)
        save_project_action.triggered.connect(self.save_project)
        load_project_action.triggered.connect(self.load_project)
        export_action.triggered.connect(self.export_calibration_json)

        status = QStatusBar(self)
        self.cursor_label = QLabel("Pixel: -, -")
        self.zoom_label = QLabel("Zoom: 100%")
        self.project_label = QLabel("No project")
        status.addPermanentWidget(self.project_label)
        status.addPermanentWidget(self.cursor_label)
        status.addPermanentWidget(self.zoom_label)
        self.setStatusBar(status)

    def _connect_signals(self) -> None:
        self.open_image_button.clicked.connect(self.open_image)
        self.open_video_button.clicked.connect(self.open_video)
        self.use_current_frame_button.clicked.connect(self.use_current_video_frame)
        self.prev_frame_button.clicked.connect(lambda: self._step_frame(-1))
        self.next_frame_button.clicked.connect(lambda: self._step_frame(1))
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.frame_index_spin.valueChanged.connect(self._on_frame_spin_changed)
        self.unit_combo.currentTextChanged.connect(self._on_units_changed)
        self.unit_symbol_edit.editingFinished.connect(self._on_units_changed)
        self.custom_unit_edit.editingFinished.connect(self._on_units_changed)
        self.grid_interval_spin.valueChanged.connect(self._on_grid_interval_changed)

        self.point_table.itemChanged.connect(self._on_point_table_item_changed)
        self.point_table.itemSelectionChanged.connect(self._on_point_table_selection_changed)
        self.delete_point_button.clicked.connect(self.delete_selected_point)
        self.clear_points_button.clicked.connect(self.clear_points)

        self.compute_button.clicked.connect(self.compute_calibration)
        self.save_project_button.clicked.connect(self.save_project)
        self.load_project_button.clicked.connect(self.load_project)
        self.export_json_button.clicked.connect(self.export_calibration_json)

        self.rotation_combo.currentIndexChanged.connect(self._on_rotation_changed)

        self.canvas.image_clicked.connect(self._on_canvas_clicked)
        self.canvas.point_selected.connect(self._on_canvas_point_selected)
        self.canvas.point_moved.connect(self._on_canvas_point_moved)
        self.canvas.cursor_moved.connect(self._on_canvas_cursor_moved)
        self.canvas.zoom_changed.connect(self._on_canvas_zoom_changed)

    def open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not file_path:
            return

        try:
            frame = self._read_image(file_path)
            self._apply_loaded_source(
                source_path=file_path,
                source_type="image",
                frame=frame,
                frame_index=0,
                frame_count=1,
                fps=0.0,
            )
            self._create_blank_project()
            self.statusBar().showMessage(f"Loaded image: {file_path}", 5000)
        except CalibrationError as exc:
            self._show_error(str(exc))

    def open_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Videos (*.mp4 *.mov *.m4v *.avi)",
        )
        if not file_path:
            return

        try:
            self._open_video_file(file_path, frame_index=0)
            self._create_blank_project()
            self.statusBar().showMessage(f"Loaded video: {file_path}", 5000)
        except CalibrationError as exc:
            self._show_error(str(exc))

    def save_project(self) -> None:
        if self.project is None:
            self._show_error("There is no project to save yet.")
            return

        self._sync_project_metadata()
        default_name = f"{self.project.project_name or 'calibration_project'}.project.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            default_name,
            "JSON (*.json)",
        )
        if not file_path:
            return

        try:
            save_project_json(self.project, file_path)
            self.statusBar().showMessage(f"Project saved to {file_path}", 5000)
        except CalibrationError as exc:
            self._show_error(str(exc))

    def load_project(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            "",
            "JSON (*.json)",
        )
        if not file_path:
            return

        try:
            loaded_project = load_project_json(file_path)
            self._load_project_into_ui(loaded_project)
            self.statusBar().showMessage(f"Project loaded from {file_path}", 5000)
        except (CalibrationError, FileNotFoundError) as exc:
            self._show_error(str(exc))

    def export_calibration_json(self) -> None:
        if self.project is None or self.project.calibration is None:
            self._show_error("Compute calibration before exporting the calibration JSON.")
            return

        self._sync_project_metadata()
        default_name = f"{self.project.project_name or 'calibration'}.calibration.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Calibration JSON",
            default_name,
            "JSON (*.json)",
        )
        if not file_path:
            return

        try:
            save_calibration_json(self.project, file_path)
            self.statusBar().showMessage(f"Calibration JSON exported to {file_path}", 5000)
        except CalibrationError as exc:
            self._show_error(str(exc))

    def compute_calibration(self) -> None:
        if self.project is None:
            self._show_error("Load an image or video before computing calibration.")
            return

        try:
            self._sync_project_metadata()
            attach_calibration(self.project)
            self.points = self.project.points
            self._refresh_point_table()
            self._refresh_canvas_points()
            self._refresh_metrics_text()
            self._refresh_calibration_overlay()
            self.statusBar().showMessage("Calibration computed successfully.", 5000)
        except CalibrationError as exc:
            self._show_error(str(exc))

    def delete_selected_point(self) -> None:
        row = self.point_table.currentRow()
        if row < 0 or row >= len(self.points):
            return

        deleted_point = self.points.pop(row)
        self._selected_point_id = None
        self._mark_calibration_dirty()
        self._sync_project_metadata()
        self._refresh_point_table()
        self._refresh_canvas_points()
        self.statusBar().showMessage(f"Deleted point {deleted_point.label or deleted_point.id}.", 3000)

    def clear_points(self) -> None:
        if not self.points:
            return

        response = QMessageBox.question(
            self,
            "Clear Points",
            "Clear all current control points and calibration results?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return

        self.points.clear()
        self._selected_point_id = None
        self._mark_calibration_dirty()
        self._sync_project_metadata()
        self._refresh_point_table()
        self._refresh_canvas_points()
        self.statusBar().showMessage("All points cleared.", 3000)

    def use_current_video_frame(self) -> None:
        if self.current_source_type != "video":
            self.statusBar().showMessage("Current frame selection is only relevant for video sources.", 4000)
            return

        self._sync_project_metadata()
        self.statusBar().showMessage(
            f"Using frame {self.current_frame_index} as the calibration frame.",
            4000,
        )

    def _open_video_file(self, file_path: str, *, frame_index: int) -> None:
        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            raise CalibrationError(f"Could not open video file: {file_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if frame_count <= 0:
            capture.release()
            raise CalibrationError(f"Video appears to contain no readable frames: {file_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

        if frame_index < 0 or frame_index >= frame_count:
            capture.release()
            raise CalibrationError(f"Requested frame index {frame_index} is outside the video range 0..{frame_count - 1}.")

        self._release_video_capture()
        self.video_capture = capture
        self._raw_video_width = width
        self._raw_video_height = height
        self.video_rotation_degrees = self._detect_video_rotation_degrees(capture, file_path, width, height)

        # Sync the rotation combo to the auto-detected value without triggering
        # _on_rotation_changed (there are no points to remap yet at this stage).
        _deg_to_idx = {0: 0, 90: 1, 180: 2, 270: 3}
        with QSignalBlocker(self.rotation_combo):
            self.rotation_combo.setCurrentIndex(_deg_to_idx.get(self.video_rotation_degrees, 0))

        frame = self._read_video_frame(frame_index)
        self._apply_loaded_source(
            source_path=file_path,
            source_type="video",
            frame=frame,
            frame_index=frame_index,
            frame_count=frame_count,
            fps=fps,
        )

    def _apply_loaded_source(
        self,
        *,
        source_path: str,
        source_type: str,
        frame: np.ndarray,
        frame_index: int,
        frame_count: int,
        fps: float,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self.current_source_path = source_path
        self.current_source_type = source_type
        self.current_frame = frame.copy()
        self.current_frame_index = int(frame_index)
        self.video_frame_count = int(frame_count)
        self.video_fps = float(fps)

        self.canvas.set_image(frame)

        frame_height, frame_width = frame.shape[:2]
        if width is not None and height is not None:
            frame_width = int(width)
            frame_height = int(height)

        is_video = source_type == "video"
        self._set_video_controls_enabled(is_video)
        self._sync_frame_controls(frame_index, frame_count)
        self._update_source_label(frame_width, frame_height)
        self._update_window_state()

    def _load_project_into_ui(self, loaded_project: CalibrationProject) -> None:
        source_path = loaded_project.source_path
        if not source_path:
            raise CalibrationError("Loaded project does not contain a source path.")

        if loaded_project.input_type == "image":
            frame = self._read_image(source_path)
            validate_reference_resolution(
                loaded_project.reference_width,
                loaded_project.reference_height,
                frame.shape[1],
                frame.shape[0],
            )
            self._release_video_capture()
            self._apply_loaded_source(
                source_path=source_path,
                source_type="image",
                frame=frame,
                frame_index=0,
                frame_count=1,
                fps=0.0,
            )
        elif loaded_project.input_type == "video":
            frame_index = loaded_project.reference_frame_index or 0
            self._open_video_file(source_path, frame_index=frame_index)
            validate_reference_resolution(
                loaded_project.reference_width,
                loaded_project.reference_height,
                self.current_frame.shape[1],
                self.current_frame.shape[0],
            )
        else:
            raise CalibrationError(f"Unsupported project input type: {loaded_project.input_type!r}")

        self.project = loaded_project
        self.points = list(loaded_project.points)
        self._selected_point_id = self.points[0].id if self.points else None
        self._apply_units_to_widgets(loaded_project.units)
        self._refresh_point_table()
        self._refresh_canvas_points()
        self._refresh_metrics_text()
        self._refresh_calibration_overlay()
        self._update_window_state()

    def _read_image(self, file_path: str) -> np.ndarray:
        frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if frame is None:
            raise CalibrationError(f"Could not read image file: {file_path}")
        self._release_video_capture()
        self.video_rotation_degrees = 0
        return frame

    def _read_video_frame(self, frame_index: int) -> np.ndarray:
        if self.video_capture is None:
            raise CalibrationError("No video is currently open.")

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = self.video_capture.read()
        if not ok or frame is None:
            raise CalibrationError(f"Failed to read frame {frame_index} from video.")
        return self._normalize_video_frame_orientation(frame)

    def _detect_video_rotation_degrees(
        self,
        capture: cv2.VideoCapture,
        file_path: str,
        width: int,
        height: int,
    ) -> int:
        rotation_value: Optional[float] = None
        orientation_property = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
        if orientation_property is not None:
            try:
                raw_value = float(capture.get(orientation_property))
                if raw_value != 0.0:
                    rotation_value = raw_value
            except Exception:
                rotation_value = None

        if rotation_value is not None:
            normalized = int(round(rotation_value)) % 360
            if normalized in {90, 180, 270}:
                return normalized

        suffix = Path(file_path).suffix.lower()
        if width > height and suffix in {".mov", ".mp4", ".m4v"}:
            return 90

        return 0

    def _normalize_video_frame_orientation(self, frame: np.ndarray) -> np.ndarray:
        if self.video_rotation_degrees == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.video_rotation_degrees == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.video_rotation_degrees == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _create_blank_project(self) -> None:
        if self.current_frame is None or self.current_source_type is None:
            return

        frame_height, frame_width = self.current_frame.shape[:2]
        self.points = []
        self._selected_point_id = None
        self.project = create_project(
            project_name=self._default_project_name(),
            input_type=self.current_source_type,
            source_path=self.current_source_path,
            reference_width=frame_width,
            reference_height=frame_height,
            units=self._collect_unit_spec(),
            points=[],
            reference_frame_index=self.current_frame_index if self.current_source_type == "video" else None,
            reference_timestamp_ms=self._current_timestamp_ms() if self.current_source_type == "video" else None,
            video_frame_count=self.video_frame_count if self.current_source_type == "video" else None,
            session_metadata={},
        )
        self._refresh_point_table()
        self._refresh_canvas_points()
        self._refresh_metrics_text()
        self._refresh_calibration_overlay()
        self._update_window_state()

    def _default_project_name(self) -> str:
        if not self.current_source_path:
            return "calibration_project"
        return Path(self.current_source_path).stem

    def _collect_unit_spec(self) -> UnitSpec:
        unit_name = self.unit_combo.currentText()
        custom_text = self.custom_unit_edit.text().strip() if unit_name == "custom" else ""
        return UnitSpec(
            unit_name=unit_name,
            unit_symbol=self.unit_symbol_edit.text().strip(),
            custom_text=custom_text,
        )

    def _apply_units_to_widgets(self, unit_spec: UnitSpec) -> None:
        with QSignalBlocker(self.unit_combo):
            index = self.unit_combo.findText(unit_spec.unit_name)
            self.unit_combo.setCurrentIndex(index if index >= 0 else self.unit_combo.findText("custom"))
        with QSignalBlocker(self.unit_symbol_edit):
            self.unit_symbol_edit.setText(unit_spec.unit_symbol)
        with QSignalBlocker(self.custom_unit_edit):
            self.custom_unit_edit.setText(unit_spec.custom_text)
        self.custom_unit_edit.setEnabled(self.unit_combo.currentText() == "custom")

    def _sync_project_metadata(self) -> None:
        if self.project is None or self.current_frame is None or self.current_source_type is None:
            return

        frame_height, frame_width = self.current_frame.shape[:2]
        self.project.project_name = self.project.project_name or self._default_project_name()
        self.project.input_type = self.current_source_type
        self.project.source_path = self.current_source_path
        self.project.reference_width = int(frame_width)
        self.project.reference_height = int(frame_height)
        self.project.units = self._collect_unit_spec()
        self.project.points = self.points
        self.project.resolution_strict = True

        if self.current_source_type == "video":
            self.project.reference_frame_index = int(self.current_frame_index)
            self.project.reference_timestamp_ms = self._current_timestamp_ms()
            self.project.video_frame_count = int(self.video_frame_count)
        else:
            self.project.reference_frame_index = None
            self.project.reference_timestamp_ms = None
            self.project.video_frame_count = None

        self.project.session_metadata["api_ready_preview"] = (
            export_api_ready_config(self.project) if self.project.calibration else None
        )
        self._update_window_state()

    def _mark_calibration_dirty(self) -> None:
        if self.project is not None:
            self.project.calibration = None
        for point in self.points:
            point.residual_image_px = None
            point.residual_world_units = None
        self._refresh_metrics_text()
        self._refresh_calibration_overlay()
        self._update_window_state()

    def _refresh_point_table(self) -> None:
        self._table_sync_in_progress = True
        try:
            self.point_table.setRowCount(len(self.points))
            for row, point in enumerate(self.points):
                enabled_item = QTableWidgetItem()
                enabled_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                enabled_item.setCheckState(Qt.Checked if point.enabled else Qt.Unchecked)
                enabled_item.setData(Qt.UserRole, point.id)
                self.point_table.setItem(row, 0, enabled_item)

                label_item = QTableWidgetItem(point.label)
                label_item.setData(Qt.UserRole, point.id)
                self.point_table.setItem(row, 1, label_item)
                self.point_table.setItem(row, 2, QTableWidgetItem(f"{point.image_x:.3f}"))
                self.point_table.setItem(row, 3, QTableWidgetItem(f"{point.image_y:.3f}"))
                self.point_table.setItem(row, 4, QTableWidgetItem(f"{point.world_x:.2f}"))
                self.point_table.setItem(row, 5, QTableWidgetItem(f"{point.world_y:.2f}"))
                self.point_table.setItem(row, 6, QTableWidgetItem(point.note))

                image_error_text = "" if point.residual_image_px is None else f"{point.residual_image_px:.4f}"
                world_error_text = "" if point.residual_world_units is None else f"{point.residual_world_units:.6f}"
                image_error_item = QTableWidgetItem(image_error_text)
                image_error_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                world_error_item = QTableWidgetItem(world_error_text)
                world_error_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.point_table.setItem(row, 7, image_error_item)
                self.point_table.setItem(row, 8, world_error_item)

            if self._selected_point_id:
                target_row = self._row_for_point_id(self._selected_point_id)
                if target_row is not None:
                    self.point_table.selectRow(target_row)
        finally:
            self._table_sync_in_progress = False

    def _refresh_canvas_points(self) -> None:
        canvas_points = []
        for point in self.points:
            label = point.label or point.id
            if not point.enabled:
                label = f"{label} (off)"
            canvas_points.append(
                CanvasPoint(
                    point_id=point.id,
                    label=label,
                    x=point.image_x,
                    y=point.image_y,
                    note=point.note,
                    selected=point.id == self._selected_point_id,
                    visible=True,
                )
            )
        self.canvas.set_points(canvas_points)
        self.canvas.select_point(self._selected_point_id)

    def _refresh_metrics_text(self) -> None:
        if self.project is None:
            self.metrics_text.setPlainText("Load an image or video to begin.")
            return

        if self.project.calibration is None:
            self.metrics_text.setPlainText("Calibration has not been computed yet.")
            return

        metrics = self.project.calibration.fit_metrics
        warning_lines = self.project.calibration.warnings or ["None"]
        usage_lines = self.project.calibration.usage_notes or []
        summary = [
            f"Model: {self.project.calibration.model_type}",
            f"Points used: {metrics.num_points_used} / {metrics.num_points_total}",
            f"Mean reprojection error (px): {metrics.mean_reprojection_error_px:.4f}",
            f"Median reprojection error (px): {metrics.median_reprojection_error_px:.4f}",
            f"RMS reprojection error (px): {metrics.rms_reprojection_error_px:.4f}",
            f"Max reprojection error (px): {metrics.max_reprojection_error_px:.4f}",
            f"Mean world error: {metrics.mean_world_error_units:.6f}",
            f"Max world error: {metrics.max_world_error_units:.6f}",
            f"Image hull coverage: {metrics.image_hull_fraction * 100.0:.2f}%",
            "",
            "Warnings:",
            *[f"- {warning}" for warning in warning_lines],
            "",
            "Usage notes:",
            *[f"- {line}" for line in usage_lines],
        ]
        self.metrics_text.setPlainText("\n".join(summary))

    def _refresh_calibration_overlay(self) -> None:
        calibration = None if self.project is None else self.project.calibration
        if calibration is None:
            self.canvas.clear_calibration_overlay()
            return
        self.canvas.update_calibration_overlay(calibration, self.grid_interval_spin.value())

    def _update_source_label(self, frame_width: int, frame_height: int) -> None:
        if not self.current_source_path or not self.current_source_type:
            self.current_source_label.setText("No source loaded")
            return

        source_name = Path(self.current_source_path).name
        if self.current_source_type == "video":
            self.current_source_label.setText(
                f"{source_name} | {frame_width}x{frame_height} | frame {self.current_frame_index}/{max(self.video_frame_count - 1, 0)}"
            )
        else:
            self.current_source_label.setText(f"{source_name} | {frame_width}x{frame_height}")

    def _set_video_controls_enabled(self, enabled: bool) -> None:
        self.use_current_frame_button.setEnabled(enabled)
        self.prev_frame_button.setEnabled(enabled)
        self.next_frame_button.setEnabled(enabled)
        self.frame_slider.setEnabled(enabled)
        self.frame_index_spin.setEnabled(enabled)
        self.rotation_combo.setEnabled(enabled)

    def _sync_frame_controls(self, frame_index: int, frame_count: int) -> None:
        self._frame_sync_in_progress = True
        try:
            maximum = max(frame_count - 1, 0)
            self.frame_slider.setRange(0, maximum)
            self.frame_index_spin.setRange(0, maximum)
            self.frame_slider.setValue(int(frame_index))
            self.frame_index_spin.setValue(int(frame_index))
            self.frame_count_label.setText(f"/ {maximum}")
            self.frame_timestamp_label.setText(self._format_timestamp(self._current_timestamp_ms()))
        finally:
            self._frame_sync_in_progress = False

    def _step_frame(self, delta: int) -> None:
        if self.current_source_type != "video":
            return
        target = max(0, min(self.current_frame_index + delta, self.video_frame_count - 1))
        if target != self.current_frame_index:
            self._set_video_frame(target)

    def _set_video_frame(self, frame_index: int) -> None:
        if self.current_source_type != "video" or self.video_capture is None:
            return
        if frame_index == self.current_frame_index:
            return

        if self.points:
            discard = QMessageBox.question(
                self,
                "Change Frame",
                "Changing the calibration frame will clear the current control points and calibration. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if discard != QMessageBox.Yes:
                self._sync_frame_controls(self.current_frame_index, self.video_frame_count)
                return

            self.points.clear()
            self._selected_point_id = None
            self._mark_calibration_dirty()

        try:
            frame = self._read_video_frame(frame_index)
        except CalibrationError as exc:
            self._show_error(str(exc))
            self._sync_frame_controls(self.current_frame_index, self.video_frame_count)
            return

        self.current_frame = frame
        self.current_frame_index = int(frame_index)
        self.canvas.set_image(frame)
        self._sync_frame_controls(self.current_frame_index, self.video_frame_count)
        self._update_source_label(frame.shape[1], frame.shape[0])
        self._sync_project_metadata()
        self._refresh_point_table()
        self._refresh_canvas_points()
        self.statusBar().showMessage(f"Showing frame {self.current_frame_index}.", 2000)

    def _on_frame_slider_changed(self, value: int) -> None:
        if self._frame_sync_in_progress:
            return
        self._set_video_frame(int(value))

    def _on_frame_spin_changed(self, value: int) -> None:
        if self._frame_sync_in_progress:
            return
        self._set_video_frame(int(value))

    def _on_units_changed(self) -> None:
        self.custom_unit_edit.setEnabled(self.unit_combo.currentText() == "custom")
        self._mark_calibration_dirty()
        self._sync_project_metadata()

    def _on_grid_interval_changed(self, _: float) -> None:
        self._refresh_calibration_overlay()

    def _on_canvas_clicked(self, image_x: float, image_y: float) -> None:
        if self.project is None:
            self._create_blank_project()
        if self.project is None:
            return

        point_id = self._next_point_id()
        point_label = f"P{len(self.points) + 1}"
        control_point = ControlPoint(
            id=point_id,
            label=point_label,
            image_x=float(image_x),
            image_y=float(image_y),
            world_x=0.0,
            world_y=0.0,
            note="",
            enabled=True,
            created_order=len(self.points),
        )
        self.points.append(control_point)
        self._selected_point_id = point_id
        self._mark_calibration_dirty()
        self._sync_project_metadata()
        self._refresh_point_table()
        self._refresh_canvas_points()
        new_row = len(self.points) - 1
        self.point_table.selectRow(new_row)
        world_x_item = self.point_table.item(new_row, 4)
        if world_x_item is not None:
            self.point_table.setCurrentItem(world_x_item)
            self.point_table.scrollToItem(world_x_item)
            self.point_table.editItem(world_x_item)
        self.statusBar().showMessage(
            f"Added point {point_label} at ({image_x:.1f}, {image_y:.1f}). Enter its world coordinates in the table.",
            5000,
        )

    def _on_canvas_point_selected(self, point_id: str) -> None:
        self._selected_point_id = point_id
        row = self._row_for_point_id(point_id)
        if row is not None:
            with QSignalBlocker(self.point_table):
                self.point_table.selectRow(row)
        self._update_window_state()

    def _on_canvas_point_moved(self, point_id: str, image_x: float, image_y: float) -> None:
        point = self._point_by_id(point_id)
        if point is None:
            return

        point.image_x = float(image_x)
        point.image_y = float(image_y)
        self._selected_point_id = point_id
        self._mark_calibration_dirty()
        self._sync_project_metadata()
        self._refresh_point_table()
        self._refresh_canvas_points()
        self.statusBar().showMessage(
            f"Moved point {point.label or point.id} to ({image_x:.1f}, {image_y:.1f}).",
            3000,
        )

    def _on_canvas_cursor_moved(self, image_x: float, image_y: float, inside: bool) -> None:
        if not inside:
            self.cursor_label.setText("Pixel: -, -")
            return
        self.cursor_label.setText(f"Pixel: {image_x:.1f}, {image_y:.1f}")

    def _on_canvas_zoom_changed(self, zoom_value: float) -> None:
        self.zoom_label.setText(f"Zoom: {zoom_value * 100.0:.0f}%")

    def _on_point_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._table_sync_in_progress:
            return

        row = item.row()
        if row < 0 or row >= len(self.points):
            return

        point = self.points[row]
        try:
            if item.column() == 0:
                point.enabled = item.checkState() == Qt.Checked
            elif item.column() == 1:
                point.label = item.text().strip() or point.id
            elif item.column() == 2:
                point.image_x = self._clamp_image_x(float(item.text()))
            elif item.column() == 3:
                point.image_y = self._clamp_image_y(float(item.text()))
            elif item.column() == 4:
                point.world_x = float(item.text())
            elif item.column() == 5:
                point.world_y = float(item.text())
            elif item.column() == 6:
                point.note = item.text()
            else:
                return
        except ValueError:
            self.statusBar().showMessage("Invalid numeric value; restoring previous value.", 4000)
            self._refresh_point_table()
            return

        self._selected_point_id = point.id
        self._mark_calibration_dirty()
        self._sync_project_metadata()
        self._refresh_point_table()
        self._refresh_canvas_points()

    def _on_point_table_selection_changed(self) -> None:
        if self._table_sync_in_progress:
            return

        row = self.point_table.currentRow()
        if row < 0 or row >= len(self.points):
            self._selected_point_id = None
            self.canvas.clear_selection()
            self._update_window_state()
            return

        point = self.points[row]
        self._selected_point_id = point.id
        self.canvas.select_point(point.id)
        self.canvas.center_on_point(point.image_x, point.image_y)
        self._update_window_state()

    def _point_by_id(self, point_id: str) -> Optional[ControlPoint]:
        for point in self.points:
            if point.id == point_id:
                return point
        return None

    def _row_for_point_id(self, point_id: str) -> Optional[int]:
        for index, point in enumerate(self.points):
            if point.id == point_id:
                return index
        return None

    def _next_point_id(self) -> str:
        next_index = 1
        existing_ids = {point.id for point in self.points}
        while True:
            candidate = f"pt_{next_index:03d}"
            if candidate not in existing_ids:
                return candidate
            next_index += 1

    def _clamp_image_x(self, value: float) -> float:
        if self.current_frame is None:
            return value
        max_x = max(float(self.current_frame.shape[1] - 1), 0.0)
        return min(max(value, 0.0), max_x)

    def _clamp_image_y(self, value: float) -> float:
        if self.current_frame is None:
            return value
        max_y = max(float(self.current_frame.shape[0] - 1), 0.0)
        return min(max(value, 0.0), max_y)

    def _update_window_state(self) -> None:
        has_project = self.project is not None
        has_points = bool(self.points)
        enabled_count = len([point for point in self.points if point.enabled])
        has_calibration = has_project and self.project.calibration is not None

        self.compute_button.setEnabled(has_project and enabled_count >= 4)
        self.save_project_button.setEnabled(has_project)
        self.export_json_button.setEnabled(bool(has_calibration))
        self.delete_point_button.setEnabled(has_points and self.point_table.currentRow() >= 0)
        self.clear_points_button.setEnabled(has_points)

        project_name = self.project.project_name if self.project else "No project"
        self.project_label.setText(f"Project: {project_name}")

    def _current_timestamp_ms(self) -> Optional[float]:
        if self.current_source_type != "video" or self.video_fps <= 0:
            return None
        return float(self.current_frame_index / self.video_fps * 1000.0)

    def _format_timestamp(self, timestamp_ms: Optional[float]) -> str:
        if timestamp_ms is None:
            return "00:00:00.000"

        total_ms = int(round(timestamp_ms))
        hours = total_ms // 3_600_000
        minutes = (total_ms % 3_600_000) // 60_000
        seconds = (total_ms % 60_000) // 1000
        milliseconds = total_ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def _release_video_capture(self) -> None:
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.video_rotation_degrees = 0
        self._raw_video_width = 0
        self._raw_video_height = 0

    def _on_rotation_changed(self, index: int) -> None:
        """
        Called when the user picks a new rotation from the combo box.

        Workflow
        --------
        1.  Compute the raw→display coordinate transform for the old rotation
            and its inverse, then re-apply for the new rotation to remap every
            existing control point.  World coordinates are unchanged.
        2.  Update video_rotation_degrees so _read_video_frame applies the
            correct cv2.rotate call.
        3.  Re-read the current frame and repaint the canvas.
        4.  Mark calibration dirty — the homography must be recomputed in the
            new coordinate space.
        """
        if self.current_source_type != "video" or self.video_capture is None:
            return

        _idx_to_deg = {0: 0, 1: 90, 2: 180, 3: 270}
        new_rotation = _idx_to_deg.get(index, 0)
        old_rotation = self.video_rotation_degrees

        if new_rotation == old_rotation:
            return

        # Remap existing control-point image coordinates into the new rotated space.
        if self.points and self._raw_video_width > 0 and self._raw_video_height > 0:
            for point in self.points:
                point.image_x, point.image_y = self._remap_point_for_rotation(
                    point.image_x, point.image_y, old_rotation, new_rotation
                )

        # Apply new rotation BEFORE reading the frame (used by _normalize_video_frame_orientation).
        self.video_rotation_degrees = new_rotation

        try:
            frame = self._read_video_frame(self.current_frame_index)
        except CalibrationError as exc:
            self._show_error(str(exc))
            # Roll back so the UI stays consistent.
            self.video_rotation_degrees = old_rotation
            with QSignalBlocker(self.rotation_combo):
                _deg_to_idx = {0: 0, 90: 1, 180: 2, 270: 3}
                self.rotation_combo.setCurrentIndex(_deg_to_idx.get(old_rotation, 0))
            return

        self.current_frame = frame
        self.canvas.set_image(frame)
        self._update_source_label(frame.shape[1], frame.shape[0])
        self._mark_calibration_dirty()
        self._sync_project_metadata()
        self._refresh_point_table()
        self._refresh_canvas_points()
        self.statusBar().showMessage(
            f"Rotation set to {new_rotation}°."
            + (" Point coordinates remapped." if self.points else ""),
            3000,
        )

    # ------------------------------------------------------------------
    # Rotation coordinate-remapping helpers
    # ------------------------------------------------------------------

    def _display_to_raw(self, px: float, py: float, rotation: int) -> tuple[float, float]:
        """
        Inverse of _normalize_video_frame_orientation: map a point from the
        rotated (displayed) frame back to the raw OpenCV frame coordinate space.

        Derivations (raw W×H → displayed):
          0°:   identity
          90° CW:  new = (H-1-y, x)       → inverse: raw = (py, H-1-px)
          180°:    new = (W-1-x, H-1-y)   → inverse: raw = (W-1-px, H-1-py)
          270° CW: new = (y, W-1-x)       → inverse: raw = (W-1-py, px)
        """
        W = float(self._raw_video_width)
        H = float(self._raw_video_height)
        if rotation == 90:
            return py, H - 1.0 - px
        if rotation == 180:
            return W - 1.0 - px, H - 1.0 - py
        if rotation == 270:
            return W - 1.0 - py, px
        return px, py  # 0° — identity

    def _raw_to_display(self, rx: float, ry: float, rotation: int) -> tuple[float, float]:
        """
        Mirror of _normalize_video_frame_orientation: map a raw-frame point to
        the rotated (displayed) coordinate space.
        """
        W = float(self._raw_video_width)
        H = float(self._raw_video_height)
        if rotation == 90:
            return H - 1.0 - ry, rx
        if rotation == 180:
            return W - 1.0 - rx, H - 1.0 - ry
        if rotation == 270:
            return ry, W - 1.0 - rx
        return rx, ry  # 0° — identity

    def _remap_point_for_rotation(
        self, px: float, py: float, old_rot: int, new_rot: int
    ) -> tuple[float, float]:
        """
        Transform a single control-point image coordinate from one rotation's
        display space to another's, by passing through raw frame space.
        """
        raw_x, raw_y = self._display_to_raw(px, py, old_rot)
        return self._raw_to_display(raw_x, raw_y, new_rot)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Calibration Tool", message)
        self.statusBar().showMessage(message, 7000)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
