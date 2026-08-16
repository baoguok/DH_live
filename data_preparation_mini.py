import subprocess
import tqdm
import numpy as np
import cv2
import argparse
import os
import math
import pickle
import shutil
import glob

from talkingface.util.face_detect_scrfd import SCRFD, FaceDetectionError as SCRFDFaceDetectionError
from talkingface.util.face_mesh_478 import predict_mesh
from talkingface.utils import smooth_array

MODULO_N = 16
# 自定义异常类
class VideoProcessingError(Exception):
    """视频处理基类异常"""
    pass

class FFmpegError(VideoProcessingError):
    """FFmpeg处理异常"""
    pass

class FaceDetectionError(VideoProcessingError):
    """人脸检测异常"""
    pass

class FirstFrameFaceDetectionError(FaceDetectionError):
    """首帧人脸检测异常"""
    pass

class FaceMeshDetectionError(VideoProcessingError):
    """面部网格检测异常"""
    pass

class EnvironmentError(VideoProcessingError):
    """环境配置错误"""
    pass

# 全局 SCRFD 检测器（懒初始化，整个流程复用）
_detector = None

def _get_detector(confThreshold: float = 0.5, nmsThreshold: float = 0.5) -> SCRFD:
    global _detector
    if _detector is None:
        _detector = SCRFD(None, confThreshold=confThreshold, nmsThreshold=nmsThreshold)
    return _detector


def detect_face(frame: np.ndarray, min_detection_confidence: float = 0.5) -> list:
    """人脸检测并验证有效性

    使用 SCRFD 检测单张人脸，返回像素坐标 [xmin, xmax, ymin, ymax]。

    Args:
        frame: BGR 图像
        min_detection_confidence: 置信度阈值

    Returns:
        [xmin, xmax, ymin, ymax] 像素坐标列表
    """
    detector = _get_detector(confThreshold=min_detection_confidence)
    try:
        xmin, ymin, xmax, ymax, _ = detector.detect_single_face(frame)
    except SCRFDFaceDetectionError as e:
        raise FaceDetectionError(str(e)) from e
    return [xmin, xmax, ymin, ymax]


def calc_face_interact(face0, face1):
    x_min = min(face0[0], face1[0])
    x_max = max(face0[1], face1[1])
    y_min = min(face0[2], face1[2])
    y_max = max(face0[3], face1[3])
    tmp0 = ((face0[1] - face0[0]) * (face0[3] - face0[2])) / ((x_max - x_min) * (y_max - y_min))
    tmp1 = ((face1[1] - face1[0]) * (face1[3] - face1[2])) / ((x_max - x_min) * (y_max - y_min))
    return min(tmp0, tmp1)


def detect_face_mesh(frame: np.ndarray) -> np.ndarray:
    """面部网格检测，返回 (478, 3) 关键点

    使用 SCRFD 人脸框 + FaceLandmarkerNet 478 点预测。
    需要先检测到人脸框（通过 detect_face 或直接提供 bbox）。

    Args:
        frame: BGR 图像

    Returns:
        pts_3d: (478, 3) 关键点数组，x/y 为像素坐标，z 为深度
    """
    detector = _get_detector()
    bboxes, scores, kpss = detector._detect_raw(frame)

    if len(bboxes) == 0:
        raise FaceMeshDetectionError("未检测到人脸，无法进行面部网格检测")

    bbox = bboxes[0]  # [x, y, w, h]
    kps = kpss[0]     # (5, 2)

    # 转为 (x1, y1, x2, y2) 格式
    bbox_xyxy = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)
    landmarks, face_score = predict_mesh(frame, bbox_xyxy, kps)

    if face_score[0] < 0.5:
        raise FaceMeshDetectionError("面部网格检测置信度过低")

    # landmarks: (1, 478, 3) → (478, 3)
    pts_3d = landmarks[0].astype(np.float32)
    return pts_3d

def save_thumbnail(frame, vid_width, vid_height, output_thumbnail):
    if vid_width > vid_height:
        new_width = 480
        new_height = int((vid_height / vid_width) * 480)
    else:
        new_height = 480
        new_width = int((vid_width / vid_height) * 480)

    # Resize the frame
    resized_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
    resized_frame = cv2.resize(resized_frame, (new_width, new_height))
    cv2.imwrite(output_thumbnail, resized_frame)
    
def encode_binary_pixels(frame, width, modulo_value):
    """
    在右上角2x2区域编码二进制序号

    2x2编码区域（右上角）:
    [y, x-1]   [y, x]     <- 位1, 位0
    [y+1, x-1] [y+1, x]   <- 位3, 位2

    缓冲区域与相邻像素一致
    """
    for bit in range(4):
        is_white = (modulo_value >> bit) & 1
        color = 255 if is_white else 0

        if bit == 0:
            dy, dx = 0, 0
        elif bit == 1:
            dy, dx = 0, 1
        elif bit == 2:
            dy, dx = 1, 0
        elif bit == 3:
            dy, dx = 1, 1
        else:
            dy, dx = 0, 0

        frame[dy, width - 1 - dx] = [color, color, color]

def extract_from_video(
        data_dir: str,
        output_pkl_path: str,
        output_video_path: str,
        matting: bool,
        reverse_option: bool
) -> None:
    """从视频提取关键点"""
    img_list = glob.glob(os.path.join(data_dir, "*.png"))
    img_list.sort()   # 按序号排序
    vid_width = 0
    vid_height = 0
    if 1:
        total_frames = len(img_list)
        pts_3d = np.zeros((total_frames, 478, 3))
        face_rect = None
        for frame_index in tqdm.tqdm(range(total_frames)):
            frame = cv2.imread(img_list[frame_index], cv2.IMREAD_UNCHANGED)  # 按帧读取视频
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            vid_width = frame.shape[1]
            vid_height = frame.shape[0]
            if frame_index == 0:
                try:
                    rect = detect_face(frame[:, :, :3], 0.25)
                    # detect_face 现在返回像素坐标 [xmin, xmax, ymin, ymax]
                    x_min, x_max, y_min, y_max = rect
                except FaceDetectionError:
                    # 尝试裁剪后检测
                    crop_y0 = int(0.1 * vid_height)
                    crop_y1 = int(0.9 * vid_height)
                    crop_x0 = int(0.1 * vid_width)
                    crop_x1 = int(0.9 * vid_width)
                    cropped = frame[crop_y0:crop_y1, crop_x0:crop_x1, :3]
                    try:
                        rect = detect_face(cropped, 0.25)
                    except FaceDetectionError as e:
                        raise FirstFrameFaceDetectionError("首帧人脸检测失败") from e

                    # 裁剪图上的像素坐标 + 偏移 = 原图坐标
                    x_min = rect[0] + crop_x0
                    x_max = rect[1] + crop_x0
                    y_min = rect[2] + crop_y0
                    y_max = rect[3] + crop_y0

                # save_thumbnail(frame, vid_width, vid_height, output_thumbnail)
                y_mid = (y_min + y_max) / 2.
                x_mid = (x_min + x_max) / 2.
                crop_size = max(x_max - x_min, y_max - y_min) * 0.8
                x_min = int(max(0, x_mid - crop_size))
                y_min = int(max(0, y_mid - crop_size))
                x_max = int(min(vid_width, x_mid + crop_size))
                y_max = int(min(vid_height, y_mid + crop_size))
                face_rect = (x_min, y_min, x_max, y_max)

            # 裁剪人脸区域
            x0, y0, x1, y1 = face_rect
            face_region = frame[y0:y1, x0:x1, :3]
            # print(y_min, y_max, x_min, x_max)
            # cv2.imshow("s", frame_face)
            # cv2.waitKey(10)
            try:
                frame_kps = detect_face_mesh(face_region)
            except FaceMeshDetectionError as e:
                raise VideoProcessingError(f"第{frame_index}帧面部网格检测失败") from e
            pts_3d[frame_index] = frame_kps + [x0, y0, 0]

            # 根据frame_kps更新face_rect
            x_min, y_min, x_max, y_max = frame_kps[:, 0].min(), frame_kps[:, 1].min(), frame_kps[:, 0].max(), frame_kps[:, 1].max()
            x_min, y_min, x_max, y_max = x0+x_min, y0+y_min, x0+x_max, y0+y_max
            x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
            crop_size = max(x_max - x_min, y_max - y_min) * 0.8
            x_min = int(max(0, x_mid - crop_size))
            y_min = int(max(0, y_mid - crop_size))
            x_max = int(min(vid_width, x_mid + crop_size))
            y_max = int(min(vid_height, y_mid + crop_size))
            face_rect = (x_min, y_min, x_max, y_max)
            if frame_index > 0:
                # 2. 计算相邻帧之间 XY 坐标的移动距离，超出一定范围就认定不合理
                frame_diff = pts_3d[frame_index] - pts_3d[frame_index - 1]
                xy_displacement = np.sqrt(frame_diff[:, 0] ** 2 + frame_diff[:, 1] ** 2)
                xy_displacement = xy_displacement.mean()
                if xy_displacement > crop_size/6:
                    # cv2.imshow("frame", frame_bgr)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    raise VideoProcessingError(f"第{frame_index}帧面部范围大幅度改变，请检查")


            if matting:
                from MatAnyone2.run import process_img_matting
                final_rgba = process_img_matting(frame, frame_index == 0)
                green_bgr = np.zeros((final_rgba.shape[0], final_rgba.shape[1], 3), dtype=np.uint8)
                green_bgr[:, :, 1] = 255

                alpha = final_rgba[:, :, 3:4] / 255.0  # Normalize alpha to [0, 1]
                final_bgr = (green_bgr * (1 - alpha) + final_rgba[:, :, :3][:, :, ::-1] * alpha).astype(np.uint8)
            else:
                final_bgr = frame[:, :, :3][:, :, ::-1]

            modulo_value = frame_index % MODULO_N
            encode_binary_pixels(final_bgr, vid_width, modulo_value)

            cv2.imwrite(os.path.join(data_dir, f"{frame_index:06d}.png"), final_bgr)

            if reverse_option:
                frame_count_inverse = total_frames * 2 - frame_index - 1
                modulo_value = frame_count_inverse % MODULO_N
                encode_binary_pixels(final_bgr, vid_width, modulo_value)
                cv2.imwrite(os.path.join(data_dir, f"{frame_count_inverse:06d}.png"), final_bgr)

        # 保存关键点
        with open(output_pkl_path, "wb") as f:
            pickle.dump(pts_3d, f)
            
        pts_3d = pts_3d.reshape(len(pts_3d), -1)
        smooth_array_ = smooth_array(pts_3d, weight=[0.01, 0.08, 0.82, 0.08, 0.01])
        pts_3d = smooth_array_.reshape(len(pts_3d), 478, 3)
        
        fps = 25
        crf = 18
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(data_dir, '%06d.png'),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            output_video_path
        ]
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
    return total_frames


def prepare_video(
        input_path: str,
        output_path: str,
        resize_option: bool = False
) -> int:
    if resize_option:
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rotate_code = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        print(f"video info: width-{width} height-{height} rotate_code-{rotate_code}")
        if rotate_code == 90 or rotate_code == 270:
            width, height = height, width
        scale = min(720 / width, 1280 / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        # 确保新的宽高为偶数
        new_width = new_width //2*2
        new_height = new_height //2*2
        cap.release()
        vf_arg = f"scale={new_width}:{new_height}"
        cmd = [
            "ffmpeg", "-i", input_path,
            "-vf", vf_arg,
            "-r", "25", '-f', 'image2', "-y", os.path.join(output_path, '%06d.png')
        ]
    else:
        cmd = [
            "ffmpeg", "-i", input_path,
            "-r", "25", '-f', 'image2', "-y", os.path.join(output_path, '%06d.png')
        ]

    print("ffmpeg cmd: ", cmd)
    # Run the command
    subprocess.run(cmd, check=True)

    # Count the number of frames generated
    frame_count = len([f for f in os.listdir(output_path) if f.endswith('.png')])
    return frame_count



def data_preparation_mini(input_video, video_dir_path, matting = False, resize_option = False, reverse_option = True):
    # 检测系统环境是否有ffmpeg
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("FFmpeg未安装或不在PATH中，请安装ffmpeg并设置为环境变量")

    # 创建输出目录
    data_dir = os.path.join(video_dir_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    frames_png_dir = os.path.join(video_dir_path, "frames")
    os.makedirs(frames_png_dir, exist_ok=True)

    frame_count = prepare_video(input_video, frames_png_dir, resize_option = resize_option)

    # 提取关键点
    output_pkl_path = os.path.join(data_dir, "processed.pkl")
    output_video_path = os.path.join(data_dir, "processed.mp4")
    extract_from_video(frames_png_dir, output_pkl_path, output_video_path, matting, reverse_option)
    shutil.rmtree(frames_png_dir)
    result = {
        "status": "success",
        "frame_count": frame_count,
        "output_video": output_video_path
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='视频人脸关键点提取工具')
    parser.add_argument('input_video', type=str, help='输入视频文件路径')
    parser.add_argument('output_dir', type=str, help='输出文件夹位置')
    parser.add_argument('--matting', action='store_true',
                        help='启用抠图功能（默认：禁用）')
    parser.add_argument('--resize', action='store_true',
                        help='启用视频缩放功能（默认：禁用）')

    # 解析参数
    args = parser.parse_args()

    print(f"输入视频: {args.input_video}")
    print(f"输出目录: {args.output_dir}")
    print(f"抠图功能: {'启用' if args.matting else '禁用'}")
    print(f"缩放功能: {'启用' if args.resize else '禁用'}")

    # 调用处理函数
    data_preparation_mini(
        args.input_video,
        args.output_dir,
        matting=args.matting,
        resize_option=args.resize,
        reverse_option=True  # 反向帧生成默认启用
    )
    print("处理完成!")

if __name__ == "__main__":
    main()
