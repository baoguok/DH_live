import cv2
import argparse
import numpy as np
import os
import sys

class FaceDetectionError(Exception):
    """人脸检测异常"""
    pass

class SCRFD():
    def __init__(self, onnxmodel, confThreshold=0.5, nmsThreshold=0.5):
        self.inpWidth = 640
        self.inpHeight = 640
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        
        if onnxmodel is None:
            current_dir_path = os.path.dirname(os.path.abspath(__file__))
            onnxmodel = os.path.join(current_dir_path, "../../checkpoint", "scrfd_2.5g_kps.onnx")

        self.net = cv2.dnn.readNet(onnxmodel)
        self.keep_ratio = True
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
    def resize_image(self, srcimg):
        padh, padw, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padw, self.inpWidth - neww - padw, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale) + 1, self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, self.inpHeight - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, padh, padw
    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    def distance2kps(self, points, distance, max_shape=None):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)
    def detect(self, srcimg):
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, (self.inpWidth, self.inpHeight), (127.5, 127.5, 127.5), swapRB=True)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # inference output
        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outs[idx][0]
            bbox_preds = outs[idx + self.fmc * 1][0] * stride
            kps_preds = outs[idx + self.fmc * 2][0] * stride

            height = blob.shape[2] // stride
            width = blob.shape[3] // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

            pos_inds = np.where(scores >= self.confThreshold)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = self.distance2kps(anchor_centers, kps_preds)
            # kpss = kps_preds
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list).ravel()
        # bboxes = np.vstack(bboxes_list) / det_scale
        # kpss = np.vstack(kpss_list) / det_scale
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)
        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        bboxes[:, 0] = (bboxes[:, 0] - padw) * ratiow
        bboxes[:, 1] = (bboxes[:, 1] - padh) * ratioh
        bboxes[:, 2] = bboxes[:, 2] * ratiow
        bboxes[:, 3] = bboxes[:, 3] * ratioh
        kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
        kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh
        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), self.confThreshold, self.nmsThreshold)
        for i in indices:
            # i = i[0]
            xmin, ymin, xamx, ymax = int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 0] + bboxes[i, 2]), int(bboxes[i, 1] + bboxes[i, 3])
            cv2.rectangle(srcimg, (xmin, ymin), (xamx, ymax), (0, 0, 255), thickness=2)
            for j in range(5):
                cv2.circle(srcimg, (int(kpss[i, j, 0]), int(kpss[i, j, 1])), 1, (0,255,0), thickness=-1)
            cv2.putText(srcimg, str(round(scores[i], 3)), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        return srcimg

    def _detect_raw(self, srcimg):
        """核心检测，返回 (bboxes_nms, scores_nms, kpss_nms) 经 NMS 后的结果
        bboxes: (N, 4) 每行为 [x, y, w, h]
        kpss: (N, 5, 2) 每行为 5 个关键点 (left_eye, right_eye, nose, left_mouth, right_mouth)
        """
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, (self.inpWidth, self.inpHeight), (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outs[idx][0]
            bbox_preds = outs[idx + self.fmc * 1][0] * stride
            kps_preds = outs[idx + self.fmc * 2][0] * stride

            height = blob.shape[2] // stride
            width = blob.shape[3] // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

            pos_inds = np.where(scores >= self.confThreshold)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            kpss = self.distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)
        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        bboxes[:, 0] = (bboxes[:, 0] - padw) * ratiow
        bboxes[:, 1] = (bboxes[:, 1] - padh) * ratioh
        bboxes[:, 2] = bboxes[:, 2] * ratiow
        bboxes[:, 3] = bboxes[:, 3] * ratioh
        kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
        kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh

        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), self.confThreshold, self.nmsThreshold)
        if len(indices) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty((0, 5, 2))
        indices = np.array(indices).flatten()
        return bboxes[indices], scores[indices], kpss[indices]

    def detect_single_face(self, srcimg):
        """检测单张人脸，返回 (xmin, ymin, xmax, ymax, keypoints)

        Args:
            srcimg: cv2 读取的 BGR 图像

        Returns:
            (xmin, ymin, xmax, ymax, keypoints)
            keypoints: (5, 2) 5个关键点 [left_eye, right_eye, nose, left_mouth, right_mouth]

        Raises:
            FaceDetectionError: 人脸检测各种异常
        """
        bboxes, scores, kpss = self._detect_raw(srcimg)
        h, w = srcimg.shape[:2]

        # 人脸数量检查
        if len(bboxes) == 0:
            raise FaceDetectionError("未检测到人脸")
        if len(bboxes) > 1:
            raise FaceDetectionError("检测到多个人脸")

        bbox = bboxes[0]  # [x, y, w, h]
        kps = kpss[0]     # (5, 2) - kps[0]图片左侧眼, kps[1]图片右侧眼, kps[2]鼻, kps[3]图片左侧嘴角, kps[4]图片右侧嘴角

        # SCRFD 关键点是图片视角：kps[0]在图片左侧（人的右眼），kps[1]在图片右侧（人的左眼）
        # 正面人脸在图片坐标中: kps[0].x < nose.x < kps[1].x
        image_left_eye = kps[0]   # 图片左侧（人的右眼）
        image_right_eye = kps[1]  # 图片右侧（人的左眼）
        nose = kps[2]

        # 人脸角度检查：鼻子的 x 应在两眼之间
        if nose[0] < image_left_eye[0] or nose[0] > image_right_eye[0]:
            raise FaceDetectionError("人脸角度不符合要求，请提供正脸图片")

        xmin, ymin, bw, bh = bbox
        xmax = xmin + bw
        ymax = ymin + bh

        # 边界检查
        if xmin < 0 or xmax < 0 or ymin > h or ymax > h or xmin > w or xmax > w:
            raise FaceDetectionError("人脸区域超出画面边界")

        # 尺寸检查
        if bw < 80 or bh < 80:
            raise FaceDetectionError("人脸尺寸不能低于80*80像素")

        return int(xmin), int(ymin), int(xmax), int(ymax), kps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='s_l.jpg', help='image path')
    args = parser.parse_args()
    # imgpath = r"c:\Users\kleinlee\Downloads\qq.png"
    detector = SCRFD(None, confThreshold=0.5, nmsThreshold=0.5)
    srcimg = cv2.imread(args.imgpath)
    height, width, _ = srcimg.shape
    if max(height, width) > 720:
        scale = 720 / max(height, width)
        srcimg = cv2.resize(srcimg, None, fx=scale, fy=scale)

       # 人脸框检测 + 校验
    try:
        xmin, ymin, xmax, ymax, kps5 = detector.detect_single_face(srcimg)
    except FaceDetectionError as e:
        print(f"人脸检测失败: {e}")
        sys.exit(1)
    print(f"人脸框: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

    # 可视化人脸框
    vis = srcimg.copy()
    cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)

    # 联合 face_mesh_478 做 478 点关键点预测
    from face_mesh_478 import predict_mesh
    bbox_xyxy = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
    landmarks, score = predict_mesh(srcimg, bbox_xyxy, kps5)
    print(f"478 点关键点形状: {landmarks.shape}, 置信度: {score[0]:.4f}")

    # 可视化关键点（每隔 5 个点画一个，避免太密）
    pts = landmarks[0, :, :2]  # (478, 2) x,y
    for k in range(0, len(pts), 1):
        px, py = int(pts[k, 0]), int(pts[k, 1])
        if 0 <= px < vis.shape[1] and 0 <= py < vis.shape[0]:
            cv2.circle(vis, (px, py), 1, (0, 255, 0), thickness=-1)

    cv2.imshow('Face Detect + Mesh 478', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()