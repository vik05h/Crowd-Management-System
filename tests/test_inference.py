import numpy as np
import pytest
from yolo_inference import YOLOInference, resolve_model_path


def test_resolve_model_path():
    path = resolve_model_path("non_existent_file.pt")
    assert path is not None
    assert path.endswith(".pt")


def test_yolo_inference_initialization():
    engine = YOLOInference()
    assert engine.model is not None
    assert engine.enable_heat_map is False


def test_detect_frame():
    engine = YOLOInference()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated_frame, count = engine.detect_frame(dummy_frame)

    assert annotated_frame is not None
    assert annotated_frame.shape == (480, 640, 3)
    assert count >= 0


def test_zoom_functionality():
    engine = YOLOInference()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    engine.detect_frame(dummy_frame)

    engine.set_zoom_cell(1, 2)
    assert engine.zoom_row == 1
    assert engine.zoom_col == 2

    zoomed = engine.get_zoomed_subimage()
    assert zoomed is not None
    assert zoomed.shape == (240, 320, 3)

    engine.set_zoom_cell(-1, -1)
    assert engine.zoom_row is None
    assert engine.zoom_col is None
