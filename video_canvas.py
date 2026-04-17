from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QNativeGestureEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)


@dataclass
class CanvasPoint:
    point_id: str
    label: str
    x: float
    y: float
    note: str = ""
    selected: bool = False
    visible: bool = True


class PointMarkerItem(QGraphicsEllipseItem):
    """Interactive point marker with a small text label."""

    def __init__(self, point_id: str, label: str, x: float, y: float, radius: float = 6.0) -> None:
        super().__init__(-radius, -radius, radius * 2.0, radius * 2.0)
        self.point_id = point_id
        self.label_text = label
        self.radius = radius

        self.setPos(QPointF(x, y))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setZValue(10.0)

        self._label_item = QGraphicsSimpleTextItem(label, self)
        self._label_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        label_font = QFont()
        label_font.setPointSize(10)
        self._label_item.setFont(label_font)
        self._label_item.setBrush(QColor("#fff8dc"))
        self._label_item.setPos(QPointF(radius + 4.0, -radius - 2.0))

        self.update_style(selected=False)

    def update_style(self, *, selected: bool) -> None:
        if selected:
            pen = QPen(QColor("#f94144"))
            pen.setWidthF(2.5)
            brush = QColor(255, 241, 118, 220)
        else:
            pen = QPen(QColor("#f8f9fa"))
            pen.setWidthF(1.5)
            brush = QColor(30, 144, 255, 210)

        self.setPen(pen)
        self.setBrush(brush)

    def set_label_text(self, label: str) -> None:
        self.label_text = label
        self._label_item.setText(label)


class VideoCanvas(QGraphicsView):
    image_clicked = Signal(float, float)
    point_selected = Signal(str)
    point_moved = Signal(str, float, float)
    cursor_moved = Signal(float, float, bool)
    zoom_changed = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)

        self._point_items: Dict[str, PointMarkerItem] = {}
        self._overlay_items: list[QGraphicsPathItem] = []
        self._image_qimage: Optional[QImage] = None
        self._image_width = 0
        self._image_height = 0
        self._pan_active = False
        self._pan_start = QPoint()
        self._dragged_marker_id: Optional[str] = None
        self._current_zoom = 1.0

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        self.setBackgroundBrush(QColor("#161a1d"))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

    def has_image(self) -> bool:
        return self._image_qimage is not None

    def image_size(self) -> tuple[int, int]:
        return self._image_width, self._image_height

    def clear_canvas(self) -> None:
        self._pixmap_item.setPixmap(QPixmap())
        self._image_qimage = None
        self._image_width = 0
        self._image_height = 0
        self.clear_calibration_overlay()
        self.clear_points()
        self.resetTransform()
        self._current_zoom = 1.0
        self.zoom_changed.emit(self._current_zoom)
        self._scene.setSceneRect(QRectF())

    def set_image(self, image: np.ndarray) -> None:
        qimage = self.numpy_to_qimage(image)
        self.set_qimage(qimage)

    def set_qimage(self, qimage: QImage) -> None:
        self._image_qimage = qimage.copy()
        self._image_width = self._image_qimage.width()
        self._image_height = self._image_qimage.height()
        self.clear_calibration_overlay()

        pixmap = QPixmap.fromImage(self._image_qimage)
        self._pixmap_item.setPixmap(pixmap)
        self._pixmap_item.setOffset(0.0, 0.0)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

        self.reset_view(fit=True)

    def reset_view(self, *, fit: bool = True) -> None:
        self.resetTransform()
        self._current_zoom = 1.0
        if fit and self.has_image():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._current_zoom = self.transform().m11()
        self.zoom_changed.emit(self._current_zoom)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.has_image():
            event.ignore()
            return

        pixel_delta = event.pixelDelta()
        angle_y = event.angleDelta().y()

        if not pixel_delta.isNull():
            # macOS trackpad two-finger swipe: pan the viewport
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - pixel_delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - pixel_delta.y())
        elif angle_y != 0:
            # Physical mouse wheel: zoom centred on cursor
            self._apply_zoom(math.exp(angle_y / 120.0 * math.log(1.15)))
        else:
            event.ignore()
            return

        event.accept()

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.NativeGesture:
            gesture: QNativeGestureEvent = event  # type: ignore[assignment]
            if gesture.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                if self.has_image():
                    # value() is the fractional delta (+0.05 = 5% zoom in per OS event)
                    self._apply_zoom(1.0 + gesture.value())
                    event.accept()
                    return True
        return super().event(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._should_start_pan(event):
            self._pan_active = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        item = self.itemAt(event.position().toPoint())

        if isinstance(item, QGraphicsSimpleTextItem):
            item = item.parentItem()

        if isinstance(item, PointMarkerItem):
            self._dragged_marker_id = item.point_id
            self.select_point(item.point_id)
            super().mousePressEvent(event)
            return

        if event.button() == Qt.LeftButton and self._is_inside_image(scene_pos):
            self.clear_selection()
            self.image_clicked.emit(float(scene_pos.x()), float(scene_pos.y()))
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._pan_active:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        inside = self._is_inside_image(scene_pos)
        self.cursor_moved.emit(float(scene_pos.x()), float(scene_pos.y()), inside)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._pan_active and event.button() in (Qt.MiddleButton, Qt.RightButton, Qt.LeftButton):
            self._pan_active = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        dragged_id = self._dragged_marker_id
        self._dragged_marker_id = None

        super().mouseReleaseEvent(event)

        if dragged_id:
            item = self._point_items.get(dragged_id)
            if item is not None:
                clamped = self._clamp_scene_point(item.pos())
                item.setPos(clamped)
                self.point_moved.emit(dragged_id, float(clamped.x()), float(clamped.y()))

    def leaveEvent(self, event) -> None:
        self.cursor_moved.emit(-1.0, -1.0, False)
        super().leaveEvent(event)

    def add_or_update_point(self, point: CanvasPoint) -> None:
        item = self._point_items.get(point.point_id)
        if item is None:
            item = PointMarkerItem(point.point_id, point.label, point.x, point.y)
            self._scene.addItem(item)
            self._point_items[point.point_id] = item
        else:
            item.set_label_text(point.label)
            item.setPos(QPointF(point.x, point.y))

        item.setVisible(point.visible)
        item.setSelected(point.selected)
        item.update_style(selected=point.selected)

    def set_points(self, points: list[CanvasPoint]) -> None:
        incoming_ids = {point.point_id for point in points}
        for point_id in list(self._point_items.keys()):
            if point_id not in incoming_ids:
                self.remove_point(point_id)

        for point in points:
            self.add_or_update_point(point)

    def remove_point(self, point_id: str) -> None:
        item = self._point_items.pop(point_id, None)
        if item is not None:
            self._scene.removeItem(item)

    def clear_points(self) -> None:
        for point_id in list(self._point_items.keys()):
            self.remove_point(point_id)

    def clear_calibration_overlay(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def update_calibration_overlay(self, calibration, grid_interval: float) -> None:
        self.clear_calibration_overlay()

        if calibration is None or not self.has_image():
            return

        try:
            step = float(grid_interval)
        except (TypeError, ValueError):
            return

        if step <= 0.0:
            return

        world_hull = np.asarray(calibration.calibration_hull_world, dtype=np.float64)
        homography = np.asarray(calibration.homography_world_to_image, dtype=np.float64)
        if world_hull.size == 0 or world_hull.ndim != 2 or world_hull.shape[1] != 2:
            return

        min_x = float(np.min(world_hull[:, 0]))
        max_x = float(np.max(world_hull[:, 0]))
        min_y = float(np.min(world_hull[:, 1]))
        max_y = float(np.max(world_hull[:, 1]))

        if not np.isfinite([min_x, max_x, min_y, max_y]).all():
            return

        grid_path = QPainterPath()
        start_x = math.floor(min_x / step) * step
        end_x = math.ceil(max_x / step) * step
        start_y = math.floor(min_y / step) * step
        end_y = math.ceil(max_y / step) * step

        for x_value in self._generate_grid_values(start_x, end_x, step):
            self._append_world_line(grid_path, homography, x_value, min_y, x_value, max_y)

        for y_value in self._generate_grid_values(start_y, end_y, step):
            self._append_world_line(grid_path, homography, min_x, y_value, max_x, y_value)

        if not grid_path.isEmpty():
            grid_item = QGraphicsPathItem(grid_path)
            grid_pen = QPen(QColor(255, 215, 0, 150))
            grid_pen.setWidthF(1.0)
            grid_item.setPen(grid_pen)
            grid_item.setZValue(3.0)
            self._scene.addItem(grid_item)
            self._overlay_items.append(grid_item)

        origin = self._project_world_point(homography, 0.0, 0.0)
        if origin is None:
            return

        x_axis_end = max_x if max_x > 0.0 else min_x
        y_axis_end = max_y if max_y > 0.0 else min_y

        x_path = QPainterPath()
        x_point = self._project_world_point(homography, x_axis_end, 0.0)
        if x_point is not None:
            x_path.moveTo(origin)
            x_path.lineTo(x_point)
            x_item = QGraphicsPathItem(x_path)
            x_pen = QPen(QColor("#f94144"))
            x_pen.setWidthF(3.0)
            x_item.setPen(x_pen)
            x_item.setZValue(4.0)
            self._scene.addItem(x_item)
            self._overlay_items.append(x_item)

        y_path = QPainterPath()
        y_point = self._project_world_point(homography, 0.0, y_axis_end)
        if y_point is not None:
            y_path.moveTo(origin)
            y_path.lineTo(y_point)
            y_item = QGraphicsPathItem(y_path)
            y_pen = QPen(QColor("#43aa8b"))
            y_pen.setWidthF(3.0)
            y_item.setPen(y_pen)
            y_item.setZValue(4.0)
            self._scene.addItem(y_item)
            self._overlay_items.append(y_item)

    def select_point(self, point_id: Optional[str]) -> None:
        for existing_id, item in self._point_items.items():
            selected = existing_id == point_id and point_id is not None
            item.setSelected(selected)
            item.update_style(selected=selected)

        if point_id is not None and point_id in self._point_items:
            self.point_selected.emit(point_id)

    def clear_selection(self) -> None:
        self.select_point(None)

    def center_on_point(self, x: float, y: float) -> None:
        self.centerOn(QPointF(x, y))

    def scene_to_image_coordinates(self, scene_pos: QPointF) -> Optional[QPointF]:
        if not self._is_inside_image(scene_pos):
            return None
        clamped = self._clamp_scene_point(scene_pos)
        return QPointF(clamped.x(), clamped.y())

    def image_rect(self) -> QRectF:
        return QRectF(0.0, 0.0, float(self._image_width), float(self._image_height))

    def numpy_to_qimage(self, image: np.ndarray) -> QImage:
        if image.ndim == 2:
            contiguous = np.ascontiguousarray(image)
            height, width = contiguous.shape
            bytes_per_line = contiguous.strides[0]
            qimage = QImage(contiguous.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            return qimage.copy()

        if image.ndim != 3:
            raise ValueError("Expected a 2D grayscale image or 3D color image array.")

        height, width, channels = image.shape
        contiguous = np.ascontiguousarray(image)

        if channels == 3:
            rgb = np.ascontiguousarray(contiguous[:, :, ::-1])
            qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888)
            return qimage.copy()

        if channels == 4:
            rgba = np.ascontiguousarray(contiguous[:, :, [2, 1, 0, 3]])
            qimage = QImage(rgba.data, width, height, rgba.strides[0], QImage.Format_RGBA8888)
            return qimage.copy()

        raise ValueError("Unsupported image channel count. Expected 1, 3, or 4 channels.")

    def _apply_zoom(self, zoom_factor: float) -> None:
        next_zoom = self.transform().m11() * zoom_factor
        next_zoom = max(0.05, min(next_zoom, 100.0))
        actual = next_zoom / self.transform().m11()
        self.scale(actual, actual)
        self._current_zoom = self.transform().m11()
        self.zoom_changed.emit(self._current_zoom)

    def _clamp_scene_point(self, scene_pos: QPointF) -> QPointF:
        if not self.has_image():
            return QPointF(scene_pos)

        x_value = min(max(scene_pos.x(), 0.0), max(float(self._image_width - 1), 0.0))
        y_value = min(max(scene_pos.y(), 0.0), max(float(self._image_height - 1), 0.0))
        return QPointF(x_value, y_value)

    def _is_inside_image(self, scene_pos: QPointF) -> bool:
        if not self.has_image():
            return False
        return self.image_rect().contains(scene_pos)

    def _should_start_pan(self, event: QMouseEvent) -> bool:
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            return True

        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
            return True

        return False

    def _generate_grid_values(self, start: float, end: float, step: float) -> list[float]:
        values = []
        current = start
        max_iterations = 10000
        iteration = 0
        while current <= end + (step * 0.5) and iteration < max_iterations:
            values.append(float(current))
            current += step
            iteration += 1
        return values

    def _append_world_line(
        self,
        path: QPainterPath,
        homography: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> None:
        start_point = self._project_world_point(homography, x0, y0)
        end_point = self._project_world_point(homography, x1, y1)
        if start_point is None or end_point is None:
            return
        path.moveTo(start_point)
        path.lineTo(end_point)

    def _project_world_point(self, homography: np.ndarray, x_value: float, y_value: float) -> Optional[QPointF]:
        source = np.array([x_value, y_value, 1.0], dtype=np.float64)
        projected = homography @ source
        if abs(projected[2]) < 1e-9:
            return None
        image_x = projected[0] / projected[2]
        image_y = projected[1] / projected[2]
        if not np.isfinite(image_x) or not np.isfinite(image_y):
            return None
        return QPointF(float(image_x), float(image_y))
