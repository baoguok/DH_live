import os
import sys
os.environ["kmp_duplicate_lib_ok"] = "true"

# 确保 MatesX/MatAnyone2 目录在 sys.path 中，以便找到 matanyone2 子包
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

import cv2
import numpy as np

import torch
from matanyone2.utils.inference_utils import gen_dilate, gen_erosion
from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.get_default_model import get_matanyone2_model
from matanyone2.utils.device import get_default_device, safe_autocast_decorator

import warnings
warnings.filterwarnings("ignore")

device = get_default_device()

# Global model and recurrent state (singleton pattern)
_sess = None
rec = None
_ti = 0

# Default parameters (using default from original code)
_n_warmup = 10
_r_erode = 10
_r_dilate = 10


def get_onnx_session():
    """
    Get the MatAnyone2 model (singleton pattern)
    """
    global _sess
    if _sess is None:
        ckpt_path = os.path.join(_CURRENT_DIR, '../checkpoint/matanyone2.pth')
        _sess = get_matanyone2_model(ckpt_path, device)
    return _sess


@torch.inference_mode()
@safe_autocast_decorator()
def process_img_matting(frame_rgba, is_new_video=True):
    """
    Process a single RGBA frame to extract matting.

    Args:
        frame_rgba (np.ndarray): Input frame in RGBA format (HxWx4), uint8.
        is_new_video (bool): If True, re-initialize the inference state for a new video.

    Returns:
        np.ndarray: Output RGBA frame with predicted alpha (HxWx4), uint8.
    """
    global rec, _ti
    # Get model (will initialize only once)
    sess = get_onnx_session()

    # Prepare RGB image as tensor (0-1, CxHxW)
    frame_rgb = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)
    image = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).to(device) / 255.0

    if is_new_video:
        # Initialize the inference processor
        rec = InferenceCore(sess, cfg=sess.cfg)
        _ti = 0

        # Extract mask from the alpha channel of the first frame
        mask = frame_rgba[:, :, 3].astype(np.uint8)
        
        # [optional] erode & dilate
        if _r_dilate > 0:
            mask = gen_dilate(mask, _r_dilate, _r_dilate)
        if _r_erode > 0:
            mask = gen_erosion(mask, _r_erode, _r_erode)
            
        mask_t = torch.from_numpy(mask).float().to(device)
        objects = [1]

        # Step 0: encode given mask
        output_prob = rec.step(image, mask_t, objects=objects)      
        # Step 0: first frame prediction
        output_prob = rec.step(image, first_frame_pred=True)      
        _ti = 1

        # Warmup: extra first_frame_pred steps to match original n_warmup behavior
        while _ti <= _n_warmup:
            output_prob = rec.step(image, first_frame_pred=True) 
            _ti += 1

        # Output prob corresponds to the first frame's final prediction
        mask_out = rec.output_prob_to_mask(output_prob)
        pha = mask_out.unsqueeze(2).cpu().numpy()
        pha = np.round(np.clip(pha * 255.0, 0, 255)).astype(np.uint8)

        final_rgba = np.concatenate([frame_rgb, pha], axis=2)
        return final_rgba

    # Subsequent frame logic
    elif rec is not None:
        output_prob = rec.step(image)
        _ti += 1

        mask_out = rec.output_prob_to_mask(output_prob)
        pha = mask_out.unsqueeze(2).cpu().numpy()
        pha = np.round(np.clip(pha * 255.0, 0, 255)).astype(np.uint8)

        final_rgba = np.concatenate([frame_rgb, pha], axis=2)
        return final_rgba
    
    else:
        # Fallback error if called without initialization
        raise RuntimeError("Inference core not initialized. Please call with is_new_video=True for the first frame.")


# Example usage
# if __name__ == "__main__":
#     # First frame of a new video
#     out1 = process_img_matting(first_frame_rgba, is_new_video=True)
#     # Subsequent frames
#     for frame in subsequent_frames:
#         out = process_img_matting(frame, is_new_video=False)