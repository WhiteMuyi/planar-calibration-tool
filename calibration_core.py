from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_USAGE_NOTES = [
    "This calibration assumes queried points lie on the ground plane (Z=0).",
    "For foot-position estimation, query the bottom-center or lowest visible foot-ground contact pixel, never the bounding-box center.",
    "Apply this calibration only to imagery with exactly the same width and height as the reference calibration.",
    "For matching-resolution videos from the same fixed setup, this transform can be applied frame-by-frame.",
]


class CalibrationError(Exception):
    """Raised when calibration input or solving fails."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"Invalid numeric value for '{field_name}': {value!r}") from exc


def _matrix_to_list(matrix: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    if matrix is None:
        return None
    return np.asarray(matrix, dtype=float).tolist()


def _list_to_matrix(value: Optional[Sequence[Sequence[float]]], field_name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise CalibrationError(f"Field '{field_name}' must be a 3x3 matrix.")
    return matrix


def _points_to_array(points: Sequence["ControlPoint"], *, enabled_only: bool = True) -> Tuple[np.ndarray, np.ndarray, List["ControlPoint"]]:
    active_points = [point for point in points if point.enabled or not enabled_only]
    if not active_points:
        raise CalibrationError("No control points are available.")

    image_points = np.array([[point.image_x, point.image_y] for point in active_points], dtype=np.float64)
    world_points = np.array([[point.world_x, point.world_y] for point in active_points], dtype=np.float64)
    return image_points, world_points, active_points


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return 0.5 * abs(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1)))


def _convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(points) <= 2:
        return points.copy()
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    return hull.astype(np.float64)


def _is_nearly_collinear(points: np.ndarray, *, min_area_threshold: float) -> bool:
    if len(points) < 3:
        return True
    hull = _convex_hull(points)
    area = _polygon_area(hull)
    return area < min_area_threshold


def _point_in_or_on_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    if len(polygon) < 3:
        return False
    test = cv2.pointPolygonTest(polygon.astype(np.float32), point, False)
    return test >= 0


def _transform_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    reshaped = points.reshape(-1, 1, 2).astype(np.float64)
    transformed = cv2.perspectiveTransform(reshaped, homography)
    return transformed.reshape(-1, 2)


def _compute_per_point_errors(
    image_points: np.ndarray,
    world_points: np.ndarray,
    homography_image_to_world: np.ndarray,
    homography_world_to_image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    projected_world = _transform_points(homography_image_to_world, image_points)
    projected_image = _transform_points(homography_world_to_image, world_points)

    world_errors = np.linalg.norm(projected_world - world_points, axis=1)
    image_errors = np.linalg.norm(projected_image - image_points, axis=1)
    return image_errors, world_errors


def _compute_condition_warnings(
    image_points: np.ndarray,
    world_points: np.ndarray,
    image_width: int,
    image_height: int,
) -> List[str]:
    warnings: List[str] = []

    image_area_threshold = max(float(image_width * image_height) * 0.0005, 1.0)
    world_span = np.ptp(world_points, axis=0) if len(world_points) else np.zeros(2, dtype=float)
    world_area_threshold = max(float(world_span[0] * world_span[1]) * 0.01, 1e-9)

    if _is_nearly_collinear(image_points, min_area_threshold=image_area_threshold):
        warnings.append("Image control points are nearly collinear; homography may be unstable.")

    if _is_nearly_collinear(world_points, min_area_threshold=world_area_threshold):
        warnings.append("World control points are nearly collinear; homography may be unstable.")

    image_hull = _convex_hull(image_points)
    image_hull_area = _polygon_area(image_hull)
    image_fraction = image_hull_area / float(max(image_width * image_height, 1))
    if image_fraction < 0.05:
        warnings.append("Control points cover only a small fraction of the image; extrapolation risk is high outside this region.")

    unique_image_points = np.unique(np.round(image_points, decimals=6), axis=0)
    if len(unique_image_points) != len(image_points):
        warnings.append("Duplicate image control points were detected.")

    unique_world_points = np.unique(np.round(world_points, decimals=6), axis=0)
    if len(unique_world_points) != len(world_points):
        warnings.append("Duplicate world control points were detected.")

    return warnings


@dataclass
class UnitSpec:
    unit_name: str
    unit_symbol: str = ""
    custom_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnitSpec":
        return cls(
            unit_name=str(data.get("unit_name", "")),
            unit_symbol=str(data.get("unit_symbol", "")),
            custom_text=str(data.get("custom_text", "")),
        )


@dataclass
class ControlPoint:
    id: str
    label: str
    image_x: float
    image_y: float
    world_x: float
    world_y: float
    note: str = ""
    enabled: bool = True
    created_order: Optional[int] = None
    residual_image_px: Optional[float] = None
    residual_world_units: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlPoint":
        return cls(
            id=str(data.get("id", "")),
            label=str(data.get("label", "")),
            image_x=_as_float(data.get("image_x"), "image_x"),
            image_y=_as_float(data.get("image_y"), "image_y"),
            world_x=_as_float(data.get("world_x"), "world_x"),
            world_y=_as_float(data.get("world_y"), "world_y"),
            note=str(data.get("note", "")),
            enabled=bool(data.get("enabled", True)),
            created_order=data.get("created_order"),
            residual_image_px=data.get("residual_image_px"),
            residual_world_units=data.get("residual_world_units"),
        )


@dataclass
class FitMetrics:
    num_points_total: int
    num_points_used: int
    mean_reprojection_error_px: float
    median_reprojection_error_px: float
    rms_reprojection_error_px: float
    max_reprojection_error_px: float
    mean_world_error_units: float
    max_world_error_units: float
    image_hull_area_px2: float
    image_hull_fraction: float
    per_point_image_errors_px: List[float] = field(default_factory=list)
    per_point_world_errors_units: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FitMetrics":
        return cls(
            num_points_total=int(data.get("num_points_total", 0)),
            num_points_used=int(data.get("num_points_used", 0)),
            mean_reprojection_error_px=_as_float(data.get("mean_reprojection_error_px", 0.0), "mean_reprojection_error_px"),
            median_reprojection_error_px=_as_float(data.get("median_reprojection_error_px", 0.0), "median_reprojection_error_px"),
            rms_reprojection_error_px=_as_float(data.get("rms_reprojection_error_px", 0.0), "rms_reprojection_error_px"),
            max_reprojection_error_px=_as_float(data.get("max_reprojection_error_px", 0.0), "max_reprojection_error_px"),
            mean_world_error_units=_as_float(data.get("mean_world_error_units", 0.0), "mean_world_error_units"),
            max_world_error_units=_as_float(data.get("max_world_error_units", 0.0), "max_world_error_units"),
            image_hull_area_px2=_as_float(data.get("image_hull_area_px2", 0.0), "image_hull_area_px2"),
            image_hull_fraction=_as_float(data.get("image_hull_fraction", 0.0), "image_hull_fraction"),
            per_point_image_errors_px=[float(v) for v in data.get("per_point_image_errors_px", [])],
            per_point_world_errors_units=[float(v) for v in data.get("per_point_world_errors_units", [])],
        )


@dataclass
class CalibrationResult:
    status: str
    model_type: str
    homography_image_to_world: np.ndarray
    homography_world_to_image: np.ndarray
    fit_metrics: FitMetrics
    calibration_hull_image: List[List[float]]
    calibration_hull_world: List[List[float]]
    warnings: List[str] = field(default_factory=list)
    usage_notes: List[str] = field(default_factory=lambda: list(DEFAULT_USAGE_NOTES))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "model_type": self.model_type,
            "homography_image_to_world": _matrix_to_list(self.homography_image_to_world),
            "homography_world_to_image": _matrix_to_list(self.homography_world_to_image),
            "fit_metrics": self.fit_metrics.to_dict(),
            "calibration_hull_image": self.calibration_hull_image,
            "calibration_hull_world": self.calibration_hull_world,
            "warnings": list(self.warnings),
            "usage_notes": list(self.usage_notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationResult":
        return cls(
            status=str(data.get("status", "unknown")),
            model_type=str(data.get("model_type", "planar_homography")),
            homography_image_to_world=_list_to_matrix(data.get("homography_image_to_world"), "homography_image_to_world"),
            homography_world_to_image=_list_to_matrix(data.get("homography_world_to_image"), "homography_world_to_image"),
            fit_metrics=FitMetrics.from_dict(data.get("fit_metrics", {})),
            calibration_hull_image=[[float(v) for v in point] for point in data.get("calibration_hull_image", [])],
            calibration_hull_world=[[float(v) for v in point] for point in data.get("calibration_hull_world", [])],
            warnings=[str(v) for v in data.get("warnings", [])],
            usage_notes=[str(v) for v in data.get("usage_notes", DEFAULT_USAGE_NOTES)],
        )


@dataclass
class CalibrationProject:
    project_version: str
    project_name: str
    input_type: str
    source_path: str
    reference_width: int
    reference_height: int
    units: UnitSpec
    points: List[ControlPoint]
    calibration: Optional[CalibrationResult] = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    reference_frame_index: Optional[int] = None
    reference_timestamp_ms: Optional[float] = None
    video_frame_count: Optional[int] = None
    resolution_strict: bool = True
    session_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_version": self.project_version,
            "project_name": self.project_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "input_type": self.input_type,
            "source_path": self.source_path,
            "reference_width": self.reference_width,
            "reference_height": self.reference_height,
            "reference_frame_index": self.reference_frame_index,
            "reference_timestamp_ms": self.reference_timestamp_ms,
            "video_frame_count": self.video_frame_count,
            "resolution_strict": self.resolution_strict,
            "units": self.units.to_dict(),
            "points": [point.to_dict() for point in self.points],
            "calibration": None if self.calibration is None else self.calibration.to_dict(),
            "session_metadata": dict(self.session_metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationProject":
        calibration_data = data.get("calibration")
        return cls(
            project_version=str(data.get("project_version", "1.0")),
            project_name=str(data.get("project_name", "")),
            input_type=str(data.get("input_type", "image")),
            source_path=str(data.get("source_path", "")),
            reference_width=int(data.get("reference_width", 0)),
            reference_height=int(data.get("reference_height", 0)),
            units=UnitSpec.from_dict(data.get("units", {})),
            points=[ControlPoint.from_dict(item) for item in data.get("points", [])],
            calibration=None if calibration_data is None else CalibrationResult.from_dict(calibration_data),
            created_at=str(data.get("created_at", _utc_now_iso())),
            updated_at=str(data.get("updated_at", _utc_now_iso())),
            reference_frame_index=data.get("reference_frame_index"),
            reference_timestamp_ms=data.get("reference_timestamp_ms"),
            video_frame_count=data.get("video_frame_count"),
            resolution_strict=bool(data.get("resolution_strict", True)),
            session_metadata=dict(data.get("session_metadata", {})),
        )


@dataclass
class QueryResult:
    image_point: Tuple[float, float]
    world_point: Tuple[float, float]
    inside_calibrated_region: bool
    extrapolated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_point": [float(self.image_point[0]), float(self.image_point[1])],
            "world_point": [float(self.world_point[0]), float(self.world_point[1])],
            "inside_calibrated_region": self.inside_calibrated_region,
            "extrapolated": self.extrapolated,
        }


def validate_control_points(points: Sequence[ControlPoint]) -> None:
    enabled_points = [point for point in points if point.enabled]
    if len(enabled_points) < 4:
        raise CalibrationError("At least 4 enabled control points are required to compute a homography.")

    image_points, world_points, _ = _points_to_array(enabled_points, enabled_only=False)

    if _is_nearly_collinear(image_points, min_area_threshold=1.0):
        raise CalibrationError("Image control points are degenerate or nearly collinear.")

    if _is_nearly_collinear(world_points, min_area_threshold=1e-9):
        raise CalibrationError("World control points are degenerate or nearly collinear.")


def validate_reference_resolution(reference_width: int, reference_height: int, candidate_width: int, candidate_height: int) -> None:
    if int(reference_width) != int(candidate_width) or int(reference_height) != int(candidate_height):
        raise CalibrationError(
            "Resolution mismatch: calibration requires "
            f"{reference_width}x{reference_height}, but candidate input is {candidate_width}x{candidate_height}."
        )


def compute_homography(
    points: Sequence[ControlPoint],
    *,
    reference_width: int,
    reference_height: int,
) -> CalibrationResult:
    validate_control_points(points)

    image_points, world_points, active_points = _points_to_array(points, enabled_only=True)

    homography_image_to_world, _ = cv2.findHomography(image_points, world_points, method=0)
    if homography_image_to_world is None:
        raise CalibrationError("OpenCV failed to compute image-to-world homography.")

    homography_world_to_image, _ = cv2.findHomography(world_points, image_points, method=0)
    if homography_world_to_image is None:
        raise CalibrationError("OpenCV failed to compute world-to-image homography.")

    homography_image_to_world = homography_image_to_world.astype(np.float64)
    homography_world_to_image = homography_world_to_image.astype(np.float64)

    image_errors, world_errors = _compute_per_point_errors(
        image_points,
        world_points,
        homography_image_to_world,
        homography_world_to_image,
    )

    for point, image_error, world_error in zip(active_points, image_errors, world_errors):
        point.residual_image_px = float(image_error)
        point.residual_world_units = float(world_error)

    image_hull = _convex_hull(image_points)
    world_hull = _convex_hull(world_points)
    image_hull_area = _polygon_area(image_hull)
    image_hull_fraction = image_hull_area / float(max(reference_width * reference_height, 1))

    fit_metrics = FitMetrics(
        num_points_total=len(points),
        num_points_used=len(active_points),
        mean_reprojection_error_px=float(np.mean(image_errors)),
        median_reprojection_error_px=float(np.median(image_errors)),
        rms_reprojection_error_px=float(math.sqrt(np.mean(np.square(image_errors)))),
        max_reprojection_error_px=float(np.max(image_errors)),
        mean_world_error_units=float(np.mean(world_errors)),
        max_world_error_units=float(np.max(world_errors)),
        image_hull_area_px2=float(image_hull_area),
        image_hull_fraction=float(image_hull_fraction),
        per_point_image_errors_px=[float(value) for value in image_errors],
        per_point_world_errors_units=[float(value) for value in world_errors],
    )

    warnings = _compute_condition_warnings(
        image_points=image_points,
        world_points=world_points,
        image_width=reference_width,
        image_height=reference_height,
    )

    return CalibrationResult(
        status="ok",
        model_type="planar_homography",
        homography_image_to_world=homography_image_to_world,
        homography_world_to_image=homography_world_to_image,
        fit_metrics=fit_metrics,
        calibration_hull_image=[[float(x), float(y)] for x, y in image_hull.tolist()],
        calibration_hull_world=[[float(x), float(y)] for x, y in world_hull.tolist()],
        warnings=warnings,
        usage_notes=list(DEFAULT_USAGE_NOTES),
    )


def map_image_to_world(calibration: CalibrationResult, image_x: float, image_y: float) -> QueryResult:
    point = np.array([[float(image_x), float(image_y)]], dtype=np.float64)
    world_point = _transform_points(calibration.homography_image_to_world, point)[0]
    hull = np.asarray(calibration.calibration_hull_image, dtype=np.float64)
    inside_region = _point_in_or_on_polygon((float(image_x), float(image_y)), hull)
    return QueryResult(
        image_point=(float(image_x), float(image_y)),
        world_point=(float(world_point[0]), float(world_point[1])),
        inside_calibrated_region=inside_region,
        extrapolated=not inside_region,
    )


def map_world_to_image(calibration: CalibrationResult, world_x: float, world_y: float) -> Tuple[float, float]:
    point = np.array([[float(world_x), float(world_y)]], dtype=np.float64)
    image_point = _transform_points(calibration.homography_world_to_image, point)[0]
    return float(image_point[0]), float(image_point[1])


def create_project(
    *,
    project_name: str,
    input_type: str,
    source_path: str,
    reference_width: int,
    reference_height: int,
    units: UnitSpec,
    points: Optional[Sequence[ControlPoint]] = None,
    reference_frame_index: Optional[int] = None,
    reference_timestamp_ms: Optional[float] = None,
    video_frame_count: Optional[int] = None,
    session_metadata: Optional[Dict[str, Any]] = None,
) -> CalibrationProject:
    if input_type not in {"image", "video"}:
        raise CalibrationError(f"Unsupported input_type: {input_type!r}. Expected 'image' or 'video'.")

    return CalibrationProject(
        project_version="1.0",
        project_name=project_name,
        input_type=input_type,
        source_path=str(source_path),
        reference_width=int(reference_width),
        reference_height=int(reference_height),
        units=units,
        points=list(points or []),
        reference_frame_index=reference_frame_index,
        reference_timestamp_ms=reference_timestamp_ms,
        video_frame_count=video_frame_count,
        session_metadata=dict(session_metadata or {}),
    )


def attach_calibration(project: CalibrationProject) -> CalibrationProject:
    calibration = compute_homography(
        project.points,
        reference_width=project.reference_width,
        reference_height=project.reference_height,
    )
    project.calibration = calibration
    project.updated_at = _utc_now_iso()
    return project


def export_api_ready_config(project: CalibrationProject) -> Dict[str, Any]:
    if project.calibration is None:
        raise CalibrationError("Project has no computed calibration to export.")

    return {
        "calibration_version": project.project_version,
        "project_name": project.project_name,
        "model_type": project.calibration.model_type,
        "reference_source_type": project.input_type,
        "reference_source_path": project.source_path,
        "reference_frame_index": project.reference_frame_index,
        "reference_timestamp_ms": project.reference_timestamp_ms,
        "reference_width": project.reference_width,
        "reference_height": project.reference_height,
        "resolution_strict": True,
        "units": project.units.to_dict(),
        "homography_image_to_world": _matrix_to_list(project.calibration.homography_image_to_world),
        "homography_world_to_image": _matrix_to_list(project.calibration.homography_world_to_image),
        "calibration_hull_image": project.calibration.calibration_hull_image,
        "fit_metrics_summary": {
            "num_points_used": project.calibration.fit_metrics.num_points_used,
            "rms_reprojection_error_px": project.calibration.fit_metrics.rms_reprojection_error_px,
            "max_reprojection_error_px": project.calibration.fit_metrics.max_reprojection_error_px,
        },
        "usage_notes": list(project.calibration.usage_notes),
    }


def save_json(data: Dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def load_json(path: str | Path) -> Dict[str, Any]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_project_json(project: CalibrationProject, path: str | Path) -> Path:
    project.updated_at = _utc_now_iso()
    return save_json(project.to_dict(), path)


def load_project_json(path: str | Path) -> CalibrationProject:
    return CalibrationProject.from_dict(load_json(path))


def save_calibration_json(project: CalibrationProject, path: str | Path) -> Path:
    return save_json(export_api_ready_config(project), path)


def load_calibration_json(path: str | Path) -> Dict[str, Any]:
    data = load_json(path)
    required_fields = [
        "reference_width",
        "reference_height",
        "homography_image_to_world",
        "homography_world_to_image",
    ]
    missing = [field_name for field_name in required_fields if field_name not in data]
    if missing:
        raise CalibrationError(f"Calibration JSON is missing required fields: {', '.join(missing)}")
    _list_to_matrix(data["homography_image_to_world"], "homography_image_to_world")
    _list_to_matrix(data["homography_world_to_image"], "homography_world_to_image")
    return data


def calibration_from_project(project: CalibrationProject) -> CalibrationResult:
    if project.calibration is None:
        raise CalibrationError("Project has no computed calibration.")
    return project.calibration


def validate_candidate_dimensions_from_calibration_json(
    calibration_json: Dict[str, Any],
    *,
    candidate_width: int,
    candidate_height: int,
) -> None:
    validate_reference_resolution(
        reference_width=int(calibration_json["reference_width"]),
        reference_height=int(calibration_json["reference_height"]),
        candidate_width=int(candidate_width),
        candidate_height=int(candidate_height),
    )

