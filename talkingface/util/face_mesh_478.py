import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np
import cv2
import os

# BatchNorm does not appear here because the released weights have it folded
# into the convolutions — verified: Google's own .tflite files contain no
# normalization ops (TFLite always folds BN on export), which is why the convs
# carry bias. The folding cannot be inverted, so the original BN statistics are
# unrecoverable. Structures were recovered from this repo's verified ONNX
# graphs, and the shipped weights were mapped from them and checked for parity
# against ONNX Runtime.

class LandmarkerBlock(nn.Module):
    """Residual inverted bottleneck of Face Landmarker: PW-reduce -> DW 3x3 -> PW-expand, skip, PReLU.

    Narrow in the middle, unlike `MeshBlock`, which keeps full width through a
    DW -> PW pair. The activation sits inside the bottleneck, between the reducing
    1x1 and the depthwise; there is none after the depthwise.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.conv = nn.Sequential(
            # pw-reduce
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.PReLU(hidden),
            # dw
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            # pw-expand
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.act = nn.PReLU(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.conv(x))


class LandmarkerDown(nn.Module):
    """Stride-2 transition of Face Landmarker: 2x2 s2 -> DW 3x3 -> PW 1x1, MaxPool skip, PReLU.

    The 2x2 stride-2 convolution opens the bottleneck instead of a depthwise, and
    because every resolution here is even it tiles exactly — no padding, unlike the
    TF-style 'same' padding `MeshDown` needs. The skip is zero-padded along channels
    when the block widens.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = out_channels // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=2, stride=2),
            nn.PReLU(hidden),
            # dw
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            # pw-expand
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.extra_channels = out_channels - in_channels
        self.act = nn.PReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        skip = self.pool(x)
        if self.extra_channels > 0:
            skip = F.pad(skip, (0, 0, 0, 0, 0, self.extra_channels))
        return self.act(self.conv(x) + skip)


class FaceLandmarkerNet(nn.Module):
    """MediaPipe Face Landmarker: 478 dense 3D landmarks from a 256x256 face crop.

    The successor to `FaceMeshNet`. Where MediaPipe's earlier 478-point model bolted
    attention branches onto the 468-point net — three custom TFLite ops and a five-way
    merge outside the graph — this one is a single network that predicts all 478 points
    directly. Landmarks 0-467 keep the Face Mesh topology and ordering, so the same
    tessellation draws it; 468-477 are the irises.

    Input is (N, 3, 256, 256) RGB in [0, 1]. Returns screen landmarks (N, 478, 3) in
    256-crop pixels and the face-presence logit (N, 1) — apply sigmoid.

    The graph carries a second, already-sigmoided scalar head alongside that logit.
    Google's MediaPipe FaceMesh V2 model card lists three outputs — the landmarks, the
    face flag, and "a limited set of blendshapes that includes cheekPuff and tongueOut",
    meant to be consumed by the separate Blendshapes model — so that head is most likely
    the blendshape one. The landmarks do not depend on it, so `conv_presence` is kept
    here for weight fidelity but left out of `forward`, which preserves the two-output
    contract `FaceMeshNet` established and the ONNX consumers rely on.

    Model card: https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf

    Example:
        >>> model = FaceLandmarkerNet()
        >>> model.load_state_dict(torch.load('weights/face_landmarker_256x256.pt'))
        >>> landmarks, score = model(x)

    Reference:
        https://github.com/google-ai-edge/mediapipe
    """

    NUM_LANDMARKS = 478

    def __init__(self) -> None:
        super().__init__()

        feature_setting = [
            # c, n — channels and number of residual blocks per stage
            [16, 4],
            [32, 4],
            [64, 4],
            [128, 4],
            [128, 4],
            [128, 4],
            [128, 4],
        ]

        features: list[nn.Module] = [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.PReLU(16),
        ]
        in_channels = 16
        for stage, (channels, num_blocks) in enumerate(feature_setting):
            if stage > 0:
                features.append(LandmarkerDown(in_channels, channels))
            features.extend(LandmarkerBlock(channels) for _ in range(num_blocks))
            in_channels = channels
        self.features = nn.Sequential(*features)

        # Declared in the order the source .tflite emits them, which is the order the
        # weights were mapped in.
        self.conv_presence = nn.Conv2d(128, 1, kernel_size=2)
        self.conv_score = nn.Conv2d(128, 1, kernel_size=2)
        self.conv_landmarks = nn.Conv2d(128, 3 * self.NUM_LANDMARKS, kernel_size=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.features(x)

        landmarks = self.conv_landmarks(x)
        score = self.conv_score(x)
        return landmarks.reshape(-1, self.NUM_LANDMARKS, 3), score.reshape(-1, 1)

INPUT_SIZE = (256, 256)

# An ROI is (center_x, center_y, side, angle_degrees) in image coordinates.
Roi = tuple[float, float, float, float]


def roi_from_box(bbox: np.ndarray, keypoints: np.ndarray | None = None, margin: float = 0.25) -> Roi:
    """Build the ROI from a detector box, MediaPipe's detection_to_roi rule.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2).
        keypoints: Optional 5-point landmarks, shape (5, 2) — rows 0 and 1 are
            the left and right eye; they set the ROI rotation (roll).
        margin: Fraction of box size to expand by on each side; the default
            0.25 yields MediaPipe's 1.5x scale.

    Returns:
        A square, optionally rotated ROI as (center_x, center_y, side, angle).
    """
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    side = (1.0 + 2.0 * margin) * max(x2 - x1, y2 - y1)

    angle = 0.0
    if keypoints is not None:
        dx, dy = (float(v) for v in np.asarray(keypoints)[1] - np.asarray(keypoints)[0])
        angle = float(np.degrees(np.arctan2(dy, dx)))

    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, side, angle


def warp_roi(image: np.ndarray, roi: Roi, size: int = INPUT_SIZE[0]) -> tuple[np.ndarray, np.ndarray]:
    """Sample a rotated square ROI directly at the model's input resolution.

    Single bilinear resampling straight to the model size — mirroring
    MediaPipe's ImageToTensorCalculator, which never materializes an
    intermediate crop (a warp-then-resize implementation interpolates twice
    and quantizes the crop side to whole pixels). Out-of-image regions are
    zero-padded, like MediaPipe's own cropper.

    Args:
        image: Full BGR image, shape (H, W, 3).
        roi: The (center_x, center_y, side, angle) ROI to cut.
        size: Output edge length in pixels; the model's input resolution.

    Returns:
        A tuple of the (size, size, 3) crop and the inverse 2x3 affine mapping
        crop-pixel coordinates back to full-image coordinates.
    """
    cx, cy, side, angle = roi
    out = size
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, out / side)
    matrix[0, 2] += out / 2.0 - cx
    matrix[1, 2] += out / 2.0 - cy

    crop = cv2.warpAffine(image, matrix, (out, out))
    return crop, cv2.invertAffineTransform(matrix)

net_face_mesh = None

@torch.no_grad()
def predict_mesh(
    image: np.ndarray,
    bbox: np.ndarray,
    keypoints: np.ndarray,
    margin: float = 0.25,
    input_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense landmarks with the PyTorch net. Returns (landmarks (N, K, 3), scores (N,)).

    K is 478 for `FaceLandmarkerNet`
    """
    global net_face_mesh
    if net_face_mesh is None:
        net_face_mesh = FaceLandmarkerNet().eval()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_weights = os.path.join(current_dir, "../../checkpoint", "face_landmarker_256x256.pt")
        net_face_mesh.load_state_dict(torch.load(default_weights))

    roi = roi_from_box(bbox, keypoints, margin)
    blobs, inverses = [], []

    crop, inverse = warp_roi(image, roi, input_size)
    blobs.append(np.transpose(crop[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1)))
    inverses.append(inverse)

    landmarks, logits = (t.numpy() for t in net_face_mesh(torch.from_numpy(np.stack(blobs))))
    scores = 1.0 / (1.0 + np.exp(-logits.ravel().astype(np.float64)))
    landmarks = landmarks.astype(np.float64)
    idx = 0
    landmarks[idx, :, :2] = landmarks[idx, :, :2] @ inverse[:, :2].T + inverse[:, 2]
    landmarks[idx, :, 2] *= roi[2] / input_size
    return landmarks.astype(np.float32), scores.astype(np.float32)