import cv2
import os
import numpy as np
import tensorflow as tf  # 替换 RKNNLite
import math

class PalmDetector:
    def __init__(self):
        # 初始化TFLite解释器
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_initialized = False
        
        # 模型配置参数 - 与Config.cpp中的palm_detect_v0_10_model_config_保持一致
        self.num_classes = 1
        self.num_anchors = 2016
        self.num_coords = 18
        self.score_clipping_thresh = 100.0
        self.x_scale = 192.0
        self.y_scale = 192.0
        self.h_scale = 192.0
        self.w_scale = 192.0
        self.min_score_thresh = 0.5
        self.min_suppression_threshold = 0.3
        self.num_keypoints = 7
        
        # 生成锚框
        self.anchors = self.generate_anchors()
    
    def load_model(self, model_path, target=None):
        """加载TFLite模型"""
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False

        print('--> Loading TFLite model:', model_path)
        try:
            # 加载TFLite模型
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # 获取输入输出详情
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            if self.input_details:
                print(f'Input tensor shape: {self.input_details[0]["shape"]}')
            if self.output_details:
                print(f'Number of output tensors: {len(self.output_details)}')
                for i, output_detail in enumerate(self.output_details):
                    print(f'Output {i} shape: {output_detail["shape"]}')
            
            self.is_initialized = True
            return True
        except Exception as e:
            print(f'Load TFLite model failed: {e}')
            return False
    
    def generate_anchors(self):
        """根据Config.cpp中的generate_anchors方法实现锚框生成"""
        options = {
            'num_layers': 4,
            'min_scale': 0.1484375,
            'max_scale': 0.75,
            'input_size_height': 192,
            'input_size_width': 192,
            'anchor_offset_x': 0.5,
            'anchor_offset_y': 0.5,
            'strides': [8, 16, 16, 16],
            'aspect_ratios': [1.0],
            'reduce_boxes_in_lowest_layer': False,
            'interpolated_scale_aspect_ratio': 1.0,
            'fixed_anchor_size': True
        }
        
        anchors = []
        layer_id = 0
        strides_size = len(options['strides'])
        
        while layer_id < strides_size:
            anchor_height = []
            anchor_width = []
            aspect_ratios = []
            scales = []
            
            # 处理相同步长的连续层
            last_same_stride_layer = layer_id
            while (last_same_stride_layer < strides_size and 
                   options['strides'][last_same_stride_layer] == options['strides'][layer_id]):
                
                scale = self.calculate_scale(options['min_scale'], 
                                           options['max_scale'], 
                                           last_same_stride_layer, 
                                           strides_size)
                
                if last_same_stride_layer == 0 and options['reduce_boxes_in_lowest_layer']:
                    aspect_ratios.extend([1.0, 2.0, 0.5])
                    scales.extend([0.1, scale, scale])
                else:
                    for ar in options['aspect_ratios']:
                        aspect_ratios.append(ar)
                        scales.append(scale)
                    
                    if options['interpolated_scale_aspect_ratio'] > 0.0:
                        if last_same_stride_layer == strides_size - 1:
                            scale_next = 1.0
                        else:
                            scale_next = self.calculate_scale(options['min_scale'],
                                                           options['max_scale'],
                                                           last_same_stride_layer + 1,
                                                           strides_size)
                        scales.append(np.sqrt(scale * scale_next))
                        aspect_ratios.append(options['interpolated_scale_aspect_ratio'])
                
                last_same_stride_layer += 1
            
            for i in range(len(aspect_ratios)):
                ratio_sqrts = np.sqrt(aspect_ratios[i])
                anchor_height.append(scales[i] / ratio_sqrts)
                anchor_width.append(scales[i] * ratio_sqrts)
            
            stride = options['strides'][layer_id]
            feature_map_height = int(np.ceil(options['input_size_height'] / stride))
            feature_map_width = int(np.ceil(options['input_size_width'] / stride))
            
            for y in range(feature_map_height):
                for x in range(feature_map_width):
                    for anchor_id in range(len(anchor_height)):
                        x_center = (x + options['anchor_offset_x']) / feature_map_width
                        y_center = (y + options['anchor_offset_y']) / feature_map_height
                        
                        new_anchor = {
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': 1.0 if options['fixed_anchor_size'] else anchor_width[anchor_id],
                            'height': 1.0 if options['fixed_anchor_size'] else anchor_height[anchor_id]
                        }
                        anchors.append(new_anchor)
            
            layer_id = last_same_stride_layer
        
        return anchors
    
    def calculate_scale(self, min_scale, max_scale, stride_index, num_strides):
        if num_strides == 1:
            return (max_scale + min_scale) * 0.5
        else:
            return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)
    
    def resize_pad(self, img):
        h, w = img.shape[:2]
        target_size = (192, 192)
        scale = min(target_size[1]/w, target_size[0]/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        pad_top = (target_size[0] - new_h) // 2
        pad_left = (target_size[1] - new_w) // 2
        padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
        return padded, scale, (pad_left, pad_top)
    
    def preprocess(self, img):
        img_float = img.astype(np.float32) / 255.0
        input_data = np.expand_dims(img_float, axis=0)  # [1,192,192,3]
        return input_data
    
    def decode_boxes(self, raw_boxes):
        decoded_boxes = []
        for i in range(len(raw_boxes)):
            raw_box = raw_boxes[i]
            anchor = self.anchors[i]
            x_center = raw_box[0] / self.x_scale * anchor['width'] + anchor['x_center']
            y_center = raw_box[1] / self.y_scale * anchor['height'] + anchor['y_center']
            w = raw_box[2] / self.w_scale * anchor['width']
            h = raw_box[3] / self.h_scale * anchor['height']
            ymin = y_center - h / 2.0
            xmin = x_center - w / 2.0
            ymax = y_center + h / 2.0
            xmax = x_center + w / 2.0
            keypoints = []
            for k in range(self.num_keypoints):
                offset = 4 + k * 2
                if offset + 1 < len(raw_box):
                    keypoint_x = raw_box[offset] / self.x_scale * anchor['width'] + anchor['x_center']
                    keypoint_y = raw_box[offset + 1] / self.y_scale * anchor['height'] + anchor['y_center']
                    keypoints.append((keypoint_x, keypoint_y))
            decoded_boxes.append({
                'ymin': ymin, 'xmin': xmin,
                'ymax': ymax, 'xmax': xmax,
                'keypoints': keypoints
            })
        return decoded_boxes
    
    def tensors_to_detections(self, raw_box_tensor, raw_score_tensor):
        detection_boxes = self.decode_boxes(raw_box_tensor)
        detection_scores = []
        for i in range(len(raw_score_tensor)):
            clipped_score = max(-self.score_clipping_thresh, min(self.score_clipping_thresh, float(raw_score_tensor[i][0])))
            sigmoid_score = 1.0 / (1.0 + np.exp(-clipped_score))
            detection_scores.append(sigmoid_score)
        output_detections = []
        for i in range(len(detection_boxes)):
            score = detection_scores[i]
            if score >= self.min_score_thresh:
                detection = detection_boxes[i].copy()
                detection['score'] = score
                output_detections.append(detection)
                
                
        return output_detections
    
    def weighted_non_max_suppression(self, detections):
        if not detections:
            return []
        sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        processed = [False] * len(sorted_detections)
        result = []
        for i in range(len(sorted_detections)):
            if processed[i]:
                continue
            current = sorted_detections[i]
            overlapping_indices = [i]
            current_box = {
                'x': current['xmin'],
                'y': current['ymin'],
                'width': current['xmax'] - current['xmin'],
                'height': current['ymax'] - current['ymin']
            }
            for j in range(i + 1, len(sorted_detections)):
                if processed[j]:
                    continue
                other = sorted_detections[j]
                other_box = {
                    'x': other['xmin'],
                    'y': other['ymin'],
                    'width': other['xmax'] - other['xmin'],
                    'height': other['ymax'] - other['ymin']
                }
                iou = self.jaccard_overlap(current_box, other_box)
                if iou > self.min_suppression_threshold:
                    overlapping_indices.append(j)
            for idx in overlapping_indices:
                processed[idx] = True
            if len(overlapping_indices) > 1:
                total_score = 0.0
                weighted_ymin = 0.0
                weighted_xmin = 0.0
                weighted_ymax = 0.0
                weighted_xmax = 0.0
                num_kps = len(current['keypoints'])
                weighted_kps = np.zeros((num_kps, 2), dtype=np.float32)
                for idx in overlapping_indices:
                    det = sorted_detections[idx]
                    score = det['score']
                    total_score += score
                    weighted_ymin += det['ymin'] * score
                    weighted_xmin += det['xmin'] * score
                    weighted_ymax += det['ymax'] * score
                    weighted_xmax += det['xmax'] * score
                    for k in range(num_kps):
                        weighted_kps[k][0] += det['keypoints'][k][0] * score
                        weighted_kps[k][1] += det['keypoints'][k][1] * score
                if total_score > 0.0:
                    weighted_detection = {
                        'ymin': weighted_ymin / total_score,
                        'xmin': weighted_xmin / total_score,
                        'ymax': weighted_ymax / total_score,
                        'xmax': weighted_xmax / total_score,
                        'keypoints': [(kp[0]/total_score, kp[1]/total_score) for kp in weighted_kps],
                        'score': total_score / len(overlapping_indices)
                    }
                    result.append(weighted_detection)
            else:
                result.append(current)
        return result
    
    def jaccard_overlap(self, box_a, box_b):
        max_x = min(box_a['x'] + box_a['width'], box_b['x'] + box_b['width'])
        min_x = max(box_a['x'], box_b['x'])
        max_y = min(box_a['y'] + box_a['height'], box_b['y'] + box_b['height'])
        min_y = max(box_a['y'], box_b['y'])
        width = max(0.0, max_x - min_x)
        height = max(0.0, max_y - min_y)
        inter = width * height
        area_a = box_a['width'] * box_a['height']
        area_b = box_b['width'] * box_b['height']
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return inter / union
    
    def detect(self, img):
        if not self.is_initialized:
            print('Model not initialized!')
            return []
        
        padded_img, scale, (pad_left, pad_top) = self.resize_pad(img)
        input_data = self.preprocess(padded_img)  # [1,192,192,3], float32
        
        # TFLite inference 替代 RKNN inference
        try:
            # 设置输入张量
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # 执行推理
            self.interpreter.invoke()
            
            # 获取输出张量
            outputs = []
            for output_detail in self.output_details:
                output_data = self.interpreter.get_tensor(output_detail['index'])
                outputs.append(output_data)
        except Exception as e:
            print(f'Inference failed: {e}')
            return []
        
        # defensive checks and prints
        print('Inference outputs type:', type(outputs))
        if not isinstance(outputs, (list, tuple)):
            print('Unexpected inference outputs type:', type(outputs))
            return []
        print('Number of outputs:', len(outputs))
        for i, out in enumerate(outputs):
            try:
                arr = np.asarray(out)
                print(f' output[{i}] shape: {arr.shape}, dtype: {arr.dtype}')
            except Exception as e:
                print(f' output[{i}] cannot be converted to ndarray:', e)
        
        if len(outputs) < 2:
            print('Unexpected number of outputs:', len(outputs))
            return []
        #输出
        out_reg = np.squeeze(np.asarray(outputs[0]))
        out_clf = np.squeeze(np.asarray(outputs[1]))
        
        print('After squeeze: out_reg.shape =', out_reg.shape, 'out_clf.shape =', out_clf.shape)
        print('Anchors count =', len(self.anchors))
        
        # Try to ensure reg shape is [num_anchors, num_coords]
        if out_reg.ndim == 1:
            # maybe flattened
            if out_reg.size == len(self.anchors) * self.num_coords:
                out_reg = out_reg.reshape((len(self.anchors), self.num_coords))
                print('Reshaped out_reg to', out_reg.shape)
            else:
                print('Unexpected flattened reg size:', out_reg.size)
        elif out_reg.ndim == 2:
            # maybe transpose
            if out_reg.shape[0] != len(self.anchors) and out_reg.shape[1] == len(self.anchors):
                out_reg = out_reg.T
                print('Transposed out_reg to', out_reg.shape)
        else:
            print('Unexpected out_reg ndim:', out_reg.ndim)
        
        # Normalize classification shape to [num_anchors]
        if out_clf.ndim == 2 and out_clf.shape[1] == 1:
            out_clf = out_clf[:, 0]
        elif out_clf.ndim == 2 and out_clf.shape[0] == 1:
            out_clf = out_clf[0, :]
        elif out_clf.ndim > 2:
            out_clf = out_clf.reshape(-1)
        print('Final out_reg.shape =', out_reg.shape, 'Final out_clf.shape =', out_clf.shape)
        
        # Print snippets for debugging
        print('--- Regression first 5 boxes (first 5 coords each) ---')
        for i in range(min(5, out_reg.shape[0])):
            print(f' box[{i}][:18]:', out_reg[i][:18])
        
        print('--- Classification raw scores (first 50) ---')
        flat_clf = np.asarray(out_clf).flatten()
        print(flat_clf[:100])
        
        # Apply sigmoid to scores for readability (with clipping to avoid overflow)
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))
        probs = sigmoid(flat_clf)
        print('--- Classification probs (first 50) ---')
        print(probs[:50])
        
        # Top-k scores
        k = 10
        if probs.size > 0:
            topk_idx = np.argsort(probs)[-k:][::-1]
            print(f'Top {k} probs and indices:')
            for idx in topk_idx:
                print(f' idx={idx}, prob={probs[idx]:.6f}, raw={flat_clf[idx]:.6f}')
        else:
            print('No classification scores to show.')
        
        # Now shape them into expected tensors
        raw_box_tensor = out_reg
        raw_score_tensor = flat_clf.reshape(-1, 1)
        
        
                # 添加打印语句，显示前5个锚框的七个点原始值
        print("\n=== 显示前5个锚框的七个点原始值 (Before Decoding) ===")
        # 只打印前5个锚框以避免输出过多
        for i in range(min(5, out_reg.shape[0])):
            # 应用sigmoid计算置信度
            prob = sigmoid(flat_clf[i]) if i < len(flat_clf) else 0.0
            
            if prob > 0:  # 只显示置信度大于0.1的锚框
                print(f"\nAnchor {i} (confidence: {prob:.4f}):")
                # 打印边界框原始值
                print(f"  Raw box values: [{', '.join(f'{v:.4f}' for v in out_reg[i][:4])}]")
                # 打印七个关键点的原始值
                print(f"  Raw keypoints values:")
                for k in range(7):  # 7个关键点
                    offset = 4 + k * 2
                    if offset + 1 < len(out_reg[i]):
                        kp_x_raw = out_reg[i][offset]
                        kp_y_raw = out_reg[i][offset + 1]
                        print(f"    Keypoint {k} raw: ({kp_x_raw:.4f}, {kp_y_raw:.4f})")
        
        
        detections = self.tensors_to_detections(raw_box_tensor, raw_score_tensor)
        filtered_detections = self.weighted_non_max_suppression(detections)
        
        
        # 添加打印语句，显示反归一化前的关键点坐标
        print("\n=== Detection Before Denormalization ===")
        for i, det in enumerate(filtered_detections):
            print(f"\nDetection {i+1} (score: {det['score']:.4f}):")
            print(f"Normalized Box: [{det['xmin']:.4f}, {det['ymin']:.4f}, {det['xmax']:.4f}, {det['ymax']:.4f}]")
            print(f"Number of normalized keypoints: {len(det['keypoints'])}")
            for kp_idx, kp in enumerate(det['keypoints']):
                print(f"Normalized Keypoint {kp_idx}: ({kp[0]:.4f}, {kp[1]:.4f})")
        
        # 反归一化坐标（从[0,1]映射回原始图像）
        h, w = img.shape[:2]
        # 修改反归一化坐标的逻辑      
        for det in filtered_detections:
            det['xmin'] = det['xmin'] * 192
            det['ymin'] = det['ymin'] * 192
            det['xmax'] = det['xmax'] * 192
            det['ymax'] = det['ymax'] * 192
            det['xmin'] = (det['xmin'] - pad_left) / scale
            det['ymin'] = (det['ymin'] - pad_top) / scale
            det['xmax'] = (det['xmax'] - pad_left) / scale
            det['ymax'] = (det['ymax'] - pad_top) / scale
            det['xmin'] = max(0, min(det['xmin'], w))
            det['ymin'] = max(0, min(det['ymin'], h))
            det['xmax'] = max(0, min(det['xmax'], w))
            det['ymax'] = max(0, min(det['ymax'], h))
            for i in range(len(det['keypoints'])):
                kp_x, kp_y = det['keypoints'][i]
                kp_x = kp_x * 192
                kp_y = kp_y * 192
                kp_x = (kp_x - pad_left) / scale
                kp_y = (kp_y - pad_top) / scale
                det['keypoints'][i] = (kp_x, kp_y)
                
                # 添加打印语句，显示反归一化后的原始图像关键点坐标
            print(f"\n=== Detection on Original Image (score: {det['score']:.4f}) ===")
            print(f"Box: [{det['xmin']:.1f}, {det['ymin']:.1f}, {det['xmax']:.1f}, {det['ymax']:.1f}]")
            print(f"Number of keypoints: {len(det['keypoints'])}")
            for kp_idx, kp in enumerate(det['keypoints']):
                print(f"Keypoint {kp_idx}: ({kp[0]:.1f}, {kp[1]:.1f})")

        
        return filtered_detections
    
    def release(self):
        if self.is_initialized:
            try:
                # TFLite解释器不需要显式释放资源
                self.interpreter = None
                self.input_details = None
                self.output_details = None
            except Exception:
                pass
            self.is_initialized = False
    
    def detection2roi(self, detections):
        """
        将检测框转换为ROI区域，与C++版本的DetectorBase::detection2roi函数保持一致
        """
        rois = []
        for detection in detections:
            roi = {}
            
            # 从Config.cpp中palm_detect_v0_10_model_config_获取配置参数
            detection2roi_method = "box"  # 重要：与C++版本一致，使用"box"方法
            kp1 = 0  # 第一个关键点索引
            kp2 = 2  # 重要：第二个关键点索引，与C++版本一致
            theta0 = math.pi / 2  # 重要：旋转偏移，与C++版本一致
            dscale = 2.6  # 缩放因子
            dy = -0.5  # 重要：Y轴偏移，与C++版本一致
            
            if detection2roi_method == "box":
                # Compute box center and scale
                # 使用box方法计算中心和大小
                roi['xc'] = (detection['xmin'] + detection['xmax']) / 2.0
                roi['yc'] = (detection['ymin'] + detection['ymax']) / 2.0
                roi['scale'] = detection['xmax'] - detection['xmin']  # assumes square boxes

                print(f"  Calculated roi['scale']: {roi['scale']:.2f}")
                roi['theta'] = 0.0  # box方法不使用旋转
            elif detection2roi_method == "alignment" and len(detection['keypoints']) >= kp2 + 1:
                # 如果使用alignment方法且关键点可用
                roi['xc'] = detection['keypoints'][kp1].x
                roi['yc'] = detection['keypoints'][kp1].y
                
                x1 = detection['keypoints'][kp2][0]
                y1 = detection['keypoints'][kp2][1]
                
                roi['scale'] = math.sqrt((roi['xc'] - x1) ** 2 + (roi['yc'] - y1) ** 2) * 2.0
                
                # 计算旋转角度
                roi['theta'] = math.atan2(roi['yc'] - y1, roi['xc'] - x1) - theta0
            else:
                # 回退到box方法
                roi['xc'] = (detection['xmin'] + detection['xmax']) / 2.0
                roi['yc'] = (detection['ymin'] + detection['ymax']) / 2.0
                roi['scale'] = detection['xmax'] - detection['xmin']
                #roi['theta'] = 0.0
            
            # 应用Y轴偏移和缩放因子
            #roi['yc'] += dy * roi['scale']
            #roi['scale'] *= dscale
            
            #rois.append(roi)
            
            # 应用Y轴偏移和缩放因子
            roi['yc'] += dy * roi['scale']
            roi['scale'] *= dscale
            
            # 无论使用哪种模式，都根据关键点计算旋转角度（与C++版本保持一致）
            if len(detection['keypoints']) > kp2:
                x0 = detection['keypoints'][kp1][0]
                y0 = detection['keypoints'][kp1][1]
                x1 = detection['keypoints'][kp2][0]
                y1 = detection['keypoints'][kp2][1]
                
                roi['theta'] = math.atan2(y0 - y1, x0 - x1) - theta0
                # 添加打印语句，输出各个参数值
                print(f"关键点数量: {len(detection['keypoints'])}")
                print(f"kp1索引: {kp1}, kp2索引: {kp2}")
                print(f"关键点{kp1}坐标: (x0={x0:.2f}, y0={y0:.2f})")
                print(f"关键点{kp2}坐标: (x1={x1:.2f}, y1={y1:.2f})")
                print(f"基准角度theta0: {theta0:.4f}弧度 ({math.degrees(theta0):.2f}度)")
                print(f"向量(y0-y1, x0-x1): ({y0-y1:.2f}, {x0-x1:.2f})")
                print(f"atan2结果: {math.atan2(y0 - y1, x0 - x1):.4f}弧度 ({math.degrees(math.atan2(y0 - y1, x0 - x1)):.2f}度)")
                print(f"计算出的旋转角度theta: {roi['theta']:.4f}弧度 ({math.degrees(roi['theta']):.2f}度)")
                print("---")
                
            else:
                roi['theta'] = 0.0  # 如果关键点不可用，则使用默认旋转角度
                
            
            rois.append(roi)
            
            
            
        
        return rois
    
    def extract_roi_points(self, roi):
        """
        从ROI参数生成四边形的四个点，与C++版本的LandmarkBase::extract_roi函数保持一致
        """
        # 定义模板点（单位正方形的四个角）
        template_points = [
            (-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)
        ]
        
        # 缩放模板点
        scaled_points = []
        for pt in template_points:
            scaled_points.append((pt[0] * roi['scale'] / 2, pt[1] * roi['scale'] / 2))
        
        # 应用旋转
        cos_theta = math.cos(roi['theta'])
        sin_theta = math.sin(roi['theta'])
        
        rotated_points = []
        for pt in scaled_points:
            x_rot = cos_theta * pt[0] - sin_theta * pt[1]
            y_rot = sin_theta * pt[0] + cos_theta * pt[1]
            rotated_points.append((x_rot + roi['xc'], y_rot + roi['yc']))
        
        return rotated_points
    
    def draw_roi(self, image, roi_boxes):
        """
        绘制ROI区域框，与C++版本的draw_roi函数保持一致
        """
        # C++中的颜色定义
        tria_blue = (255, 0, 0)  # BGR格式
        tria_pink = (255, 0, 255)  # BGR格式
        
        for box in roi_boxes:
            if len(box) != 4:  # 确保每个ROI是四边形
                continue
            
            p1 = (int(box[0][0]), int(box[0][1]))
            p2 = (int(box[1][0]), int(box[1][1]))
            p3 = (int(box[2][0]), int(box[2][1]))
            p4 = (int(box[3][0]), int(box[3][1]))
            
            # 绘制四条边，与C++版本保持一致的颜色和线宽
            cv2.line(image, p1, p2, tria_blue, 2)
            cv2.line(image, p1, p3, tria_pink, 2)
            cv2.line(image, p2, p4, tria_blue, 2)
            cv2.line(image, p3, p4, tria_blue, 2)


class HandLandmark:
    def __init__(self):
        # 初始化TFLite解释器
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_initialized = False
        
        # 模型配置参数
        self.image_size = 224  # 根据实际模型输入大小调整
    
    def load_model(self, model_path, target=None):
        """加载TFLite模型"""
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False

        print('--> Loading TFLite model:', model_path)
        try:
            # 加载TFLite模型
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # 获取输入输出详情
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            if self.input_details:
                print(f'Input tensor shape: {self.input_details[0]["shape"]}')
            if self.output_details:
                print(f'Number of output tensors: {len(self.output_details)}')
                for i, output_detail in enumerate(self.output_details):
                    print(f'Output {i} shape: {output_detail["shape"]}')
            
            self.is_initialized = True
            return True
        except Exception as e:
            print(f'Load TFLite model failed: {e}')
            return False
    
    def extract_roi(self, image, rois):
        """从原始图像中提取ROI区域"""
        roi_images = []
        affine_matrices = []
        for roi in rois:
            # 计算ROI的四个点
            roi_points = self.calculate_roi_points(roi)
            
            # 计算仿射变换矩阵
            target_points = np.array([[0, 0], [self.image_size-1, 0], [0, self.image_size-1]], dtype=np.float32)
            roi_3points = np.array([roi_points[0], roi_points[3], roi_points[1]], dtype=np.float32)
            M = cv2.getAffineTransform(roi_3points, target_points)
            
            # 应用仿射变换获取ROI图像
            roi_image = cv2.warpAffine(image, M, (self.image_size, self.image_size))
            
            roi_images.append(roi_image)
            affine_matrices.append(M)
        
        return roi_images, affine_matrices, rois
    
    def calculate_roi_points(self, roi):
        """计算ROI的四个顶点"""
        # 与PalmDetector中的extract_roi_points方法相同
        template_points = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
        
        scaled_points = []
        for pt in template_points:
            scaled_points.append((pt[0] * roi['scale'] / 2, pt[1] * roi['scale'] / 2))
        
        cos_theta = math.cos(roi['theta'])
        sin_theta = math.sin(roi['theta'])
        
        rotated_points = []
        for pt in scaled_points:
            x_rot = cos_theta * pt[0] - sin_theta * pt[1]
            y_rot = sin_theta * pt[0] + cos_theta * pt[1]
            rotated_points.append((x_rot + roi['xc'], y_rot + roi['yc']))
        
        return rotated_points
    
    def predict(self, roi_images):
        """在ROI图像上预测关键点"""
        if not self.is_initialized:
            print('Hand landmark model not initialized!')
            return [], [], []
        
        scores_list = []
        landmarks_list = []
        handedness_list = []
        
        for roi_image in roi_images:
            # 预处理
            input_data = self.preprocess(roi_image)
            
            try:
                # TFLite推理
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                
                # 获取输出
                # 假设输出包括关键点坐标、置信度和左右手信息
                # 具体输出索引和格式需要根据实际模型调整
                outputs = []
                for output_detail in self.output_details:
                    outputs.append(self.interpreter.get_tensor(output_detail['index']))
                
                # 解析输出
                # 这里的解析逻辑需要根据实际模型的输出格式进行调整
                # 以下为示例代码
                if len(outputs) >= 2:
                    # 假设第一个输出是关键点坐标，形状为[1, 21, 2]或类似
                    landmarks = np.squeeze(outputs[0])
                    
                    # 假设第二个输出是置信度
                    score = float(np.squeeze(outputs[1])) if len(outputs) >= 2 else 0.5
                    
                    # 假设第三个输出是左右手信息
                    handedness = float(np.squeeze(outputs[2])) if len(outputs) >= 3 else 0.0
                    
                    scores_list.append(score)
                    landmarks_list.append(landmarks.tolist())
                    handedness_list.append(handedness)
            except Exception as e:
                print(f'Landmark prediction failed: {e}')
                scores_list.append(0.0)
                landmarks_list.append([])
                handedness_list.append(0.0)
        
        return scores_list, landmarks_list, handedness_list
    
    def preprocess(self, roi_image):
        """预处理ROI图像"""
        # 调整大小
        resized = cv2.resize(roi_image, (self.image_size, self.image_size))
        # 转换为float并归一化
        img_float = resized.astype(np.float32) / 255.0
        # 添加批次维度
        input_data = np.expand_dims(img_float, axis=0)
        return input_data
    
    def denormalize_landmarks(self, landmarks_list, affine_matrices):
        """将关键点坐标映射回原始图像"""
        denormalized_landmarks = []
        
        for i, landmarks in enumerate(landmarks_list):
            if not landmarks:
                denormalized_landmarks.append([])
                continue
            
            M = affine_matrices[i]
            # 计算逆变换矩阵
            Minv = cv2.invertAffineTransform(M)
            
            denormalized = []
            for lm in landmarks:
                try:
                    # 确保关键点坐标是有效的
                    if isinstance(lm, (list, tuple)) and len(lm) >= 2:
                        # 将归一化坐标映射到ROI图像上的坐标
                        x_roi = lm[0] * self.image_size
                        y_roi = lm[1] * self.image_size
                        
                        # 应用逆仿射变换映射回原始图像
                        x_orig = Minv[0, 0] * x_roi + Minv[0, 1] * y_roi + Minv[0, 2]
                        y_orig = Minv[1, 0] * x_roi + Minv[1, 1] * y_roi + Minv[1, 2]
                        
                        denormalized.append((x_orig, y_orig))
                except Exception as e:
                    print(f'Error denormalizing landmark: {e}')
            
            denormalized_landmarks.append(denormalized)
        
        return denormalized_landmarks
    
    def draw_landmarks(self, image, landmarks, handedness=None):
        """在图像上绘制关键点和连线"""
        if not landmarks:
            return
        
        # 定义关键点连接方式
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20)   # 小指
        ]
        
        # 绘制关键点
        for i, landmark in enumerate(landmarks):
            try:
                if isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # 绿色点
                    cv2.putText(image, str(i), (x+5, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                print(f'Error drawing landmark {i}: {e}')
        
        # 绘制连线
        for conn in connections:
            try:
                if conn[0] < len(landmarks) and conn[1] < len(landmarks):
                    if len(landmarks[conn[0]]) >= 2 and len(landmarks[conn[1]]) >= 2:
                        x1, y1 = int(landmarks[conn[0]][0]), int(landmarks[conn[0]][1])
                        x2, y2 = int(landmarks[conn[1]][0]), int(landmarks[conn[1]][1])
                        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            except Exception as e:
                print(f'Error drawing connection {conn}: {e}')
        
        # 绘制左右手信息
        if handedness is not None and len(landmarks) > 0 and len(landmarks[0]) >= 2:
            try:
                hand_label = 'Left' if handedness < 0.5 else 'Right'
                cv2.putText(image, hand_label, (int(landmarks[0][0])-30, int(landmarks[0][1])-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f'Error drawing handedness: {e}')
    
    def release(self):
        if self.is_initialized:
            try:
                # TFLite解释器不需要显式释放资源
                self.interpreter = None
                self.input_details = None
                self.output_details = None
            except Exception:
                pass
            self.is_initialized = False


# 使用示例
if __name__ == '__main__':
    # 初始化手掌检测器
    detector = PalmDetector()
    # 修改为TFLite模型路径
    palm_model_path = 'models/hand_detector.tflite'
    if not detector.load_model(palm_model_path):
        exit(-1)
    
    # 初始化手部关键点检测器
    hand_landmark = HandLandmark()
    # 修改为TFLite模型路径
    landmark_model_path = 'models/hand_landmarks_detector.tflite'
    if not hand_landmark.load_model(landmark_model_path):
        detector.release()
        exit(-1)
    
    try:
        img_path = '16.jpg'
        img = cv2.imread(img_path)
        if img is None:
            print('Failed to read image:', img_path)
            exit(-1)
        
        # 检测手掌
        detections = detector.detect(img)
        
        # 绘制检测框和关键点
        for det in detections:
            x1, y1 = int(det['xmin']), int(det['ymin'])
            x2, y2 = int(det['xmax']), int(det['ymax'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            score = det.get('score', 0.0)
            cv2.putText(img, f'{score:.2f}', (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for i, (kp_x, kp_y) in enumerate(det['keypoints']):
                cv2.circle(img, (int(kp_x), int(kp_y)), 3, (0, 0, 255), -1)
                cv2.putText(img, str(i), (int(kp_x)+5, int(kp_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # 计算并绘制ROI区域框
        if len(detections) > 0:
            # 计算ROI（仅计算一次）
            rois = detector.detection2roi(detections)


            # 添加代码：打印ROI区域的坐标信息
            print("ROI区域坐标信息:")
            for i, roi in enumerate(rois):
                print(f"ROI {i+1}:")
                print(f"  中心坐标: (xc={roi['xc']}, yc={roi['yc']})")
                print(f"  缩放因子: {roi['scale']}")
                print(f"  旋转角度: {roi['theta']} 弧度")
                
                # 如果需要，可以同时打印ROI四边形的四个顶点坐标
                roi_points = detector.extract_roi_points(roi)
                print(f"  四边形顶点: {roi_points}")
            
            # 生成ROI四边形的四个点并绘制（可选，用于可视化）
            roi_boxes = []
            for roi in rois:
                roi_points = detector.extract_roi_points(roi)
                roi_boxes.append(roi_points)
            detector.draw_roi(img, roi_boxes)
            
            # 直接使用已经计算好的ROI进行关键点检测
            # 提取ROI区域图像
            roi_images, affine_matrices, _ = hand_landmark.extract_roi(img, rois)
            
            if roi_images:
                # 进行关键点预测
                scores_list,landmarks_list, handedness_list = hand_landmark.predict(roi_images)
                
                # 将关键点坐标映射回原始图像
                denormalized_landmarks = hand_landmark.denormalize_landmarks(landmarks_list, affine_matrices)
            # 添加代码：打印关键点坐标和置信度
            print("手部关键点坐标（映射回原始图像后）:")
            for hand_idx, landmarks in enumerate(denormalized_landmarks):
                print(f"手 {hand_idx+1} (置信度: {scores_list[hand_idx]:.4f}):")
                if not landmarks:
                    print("  未检测到关键点")
                else:
                    for landmark_idx, lm in enumerate(landmarks):
                        try:
                            # 使用索引访问避免解包错误
                            x, y = lm[0], lm[1]
                            print(f"  关键点 {landmark_idx}: (x={x:.2f}, y={y:.2f})")
                        except (IndexError, TypeError) as e:
                            print(f"  关键点 {landmark_idx}: 无效坐标 ({e})")
                
                # 绘制关键点和连线
                # 找到以下代码段并进行修改
                for i, landmarks in enumerate(denormalized_landmarks):
                    # 修复：handedness变量未定义的问题
                    handedness = handedness_list[i] if i < len(handedness_list) else None
                    hand_landmark.draw_landmarks(img, landmarks, handedness)
        
        # 如果在 headless 环境下，注释掉下面两行以避免 X11/Qt 错误
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        out_file = 'result.jpg'
        cv2.imwrite(out_file, img)
        print(f'Detected {len(detections)} palms, result saved to {out_file}')
    finally:
        # 释放资源
        hand_landmark.release()
        detector.release()