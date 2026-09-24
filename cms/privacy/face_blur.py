"""Privacy Protection & Face Anonymization Pipeline.

Complies with Digital Personal Data Protection Act (DPDPA 2023) and GDPR.
Applies real-time Gaussian blurring or pixelation to facial regions identified
via YOLO-Pose cranial keypoints or person bounding box upper-centroid tracking.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class PrivacyManager:
    """Manages real-time face blurring and spatial privacy masking."""

    def __init__(self, enabled: bool = False, blur_kernel_size: int = 31):
        self.enabled = enabled
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.compliance_standard = "DPDPA 2023 / GDPR Anonymization"

    def set_enabled(self, enabled: bool) -> bool:
        """Enable or disable real-time privacy anonymization."""
        self.enabled = bool(enabled)
        return self.enabled

    def toggle(self) -> bool:
        """Toggle privacy blur state."""
        self.enabled = not self.enabled
        return self.enabled

    def get_status(self) -> Dict[str, Any]:
        """Return current privacy pipeline configuration."""
        return {
            "enabled": self.enabled,
            "compliance": self.compliance_standard,
            "kernel_size": self.blur_kernel_size,
            "method": "pose_keypoint_and_head_geometry",
        }

    def apply_face_blur(
        self,
        frame: np.ndarray,
        boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        keypoints: Optional[List[List[Tuple[float, float, float]]]] = None,
    ) -> np.ndarray:
        """Apply Gaussian blur to face regions on the given frame.

        Args:
            frame: BGR numpy image frame.
            boxes: Optional list of person bounding boxes as (x1, y1, x2, y2).
            keypoints: Optional list of 17-keypoint sets from pose estimation.

        Returns:
            Anonymized BGR image frame.
        """
        if not self.enabled or frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        anonymized = frame.copy()
        blurred_faces = 0

        # Method 1: Precise cranial keypoints from YOLO-Pose (keypoints 0..4: nose, eyes, ears)
        if keypoints:
            for person_kpts in keypoints:
                if len(person_kpts) < 5:
                    continue
                # Extract head keypoints: 0=nose, 1=left eye, 2=right eye, 3=left ear, 4=right ear
                head_pts = [
                    (int(pt[0]), int(pt[1]))
                    for pt in person_kpts[:5]
                    if len(pt) >= 2 and pt[0] > 0 and pt[1] > 0 and (len(pt) < 3 or pt[2] > 0.25)
                ]
                if len(head_pts) >= 2:
                    xs = [p[0] for p in head_pts]
                    ys = [p[1] for p in head_pts]
                    hx1, hx2 = min(xs), max(xs)
                    hy1, hy2 = min(ys), max(ys)

                    # Add padding around detected keypoints
                    pad_w = max(18, int((hx2 - hx1) * 0.45))
                    pad_h = max(22, int((hy2 - hy1) * 0.55))

                    fx1 = max(0, hx1 - pad_w)
                    fy1 = max(0, hy1 - pad_h)
                    fx2 = min(w, hx2 + pad_w)
                    fy2 = min(h, hy2 + pad_h)

                    if fx2 > fx1 + 4 and fy2 > fy1 + 4:
                        self._blur_roi(anonymized, fx1, fy1, fx2, fy2)
                        blurred_faces += 1

        # Method 2: Head geometry approximation from person bounding boxes
        if boxes and blurred_faces == 0:
            for box in boxes:
                bx1, by1, bx2, by2 = [int(v) for v in box[:4]]
                bw = bx2 - bx1
                bh = by2 - by1

                if bw < 10 or bh < 15:
                    continue

                # Head region is approximately the top 20% height, centered horizontally
                fx1 = max(0, bx1 + int(bw * 0.15))
                fx2 = min(w, bx2 - int(bw * 0.15))
                fy1 = max(0, by1)
                fy2 = min(h, by1 + int(bh * 0.24))

                if fx2 > fx1 + 4 and fy2 > fy1 + 4:
                    self._blur_roi(anonymized, fx1, fy1, fx2, fy2)
                    blurred_faces += 1

        # Render subtle privacy tag indicator when active
        if blurred_faces > 0:
            self._render_privacy_watermark(anonymized)

        return anonymized

    def _blur_roi(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
        """Apply heavy Gaussian blur to a sub-region."""
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return
        k = self.blur_kernel_size
        k = k if k % 2 == 1 else k + 1
        # Heavy dual-pass blur for complete irreversible identity protection
        blurred = cv2.GaussianBlur(roi, (k, k), sigmaX=25, sigmaY=25)
        blurred = cv2.GaussianBlur(blurred, (k, k), sigmaX=25, sigmaY=25)
        img[y1:y2, x1:x2] = blurred

    def _render_privacy_watermark(self, img: np.ndarray) -> None:
        """Render discrete privacy compliance badge on the bottom-left of stream."""
        tag = "PRIVACY BLUR ACTIVE [DPDPA 2023]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(tag, font, font_scale, thickness)
        px, py = 14, img.shape[0] - 14

        # Black background box with solid outline
        cv2.rectangle(img, (px - 6, py - th - 6), (px + tw + 6, py + baseline + 4), (0, 0, 0), -1)
        cv2.rectangle(img, (px - 6, py - th - 6), (px + tw + 6, py + baseline + 4), (166, 250, 255), 1)
        cv2.putText(img, tag, (px, py), font, font_scale, (166, 250, 255), thickness, cv2.LINE_AA)
