# # 이미지 세그멘테이션 =====================================

# import torch
# from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision.transforms import functional as F
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# class MaskRCNNPredictor:
#     def __init__(self):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model = maskrcnn_resnet50_fpn(pretrained=True)
#         self.model.to(self.device)
#         self.model.eval()
#         self.class_names = [
#             ''
#         ]

#     def predict(self, image_path, confidence_threshold=0.5):
#         image = Image.open(image_path).convert('RGB')
#         image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             predictions = self.model(image_tensor)
#         pred = predictions[0]

#         keep_idx = pred['scores'] > confidence_threshold

#         boxes = pred['boxes'][keep_idx].cpu().numpy()
#         labels = pred['labels'][keep_idx].cpu().numpy()
#         scores = pred['scores'][keep_idx].cpu().numpy()
#         masks = pred['masks'][keep_idx].cpu().numpy()

#         return image, boxes, labels, scores, masks


    # def visualize_results(self, image, boxes, labels, scores, masks):
    #     fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    #     axes[0].imshow(image)
    #     axes[0].set_title('Object Detection')
    #     axes[0].axis('off')

    #     colors = plt.cm.Set1(np.linspace(0, 1, len(masks)))

    #     for i, mask in enumerate(masks):
    #         mask_colored = np.zeros((*mask.shape[1:], 4))
    #         mask_colored[:, :, :3] = colors[i][:3]
    #         mask_colored[:, :, 3] = mask[0] * 0.7
            
    #         axes[1].imshow(mask_colored)

    #     plt.tight_layout()
    #     plt.show()


# def demo_maskrcnn():
#     predictor = MaskRCNNPredictor()

#     image, boxes, labels, scores, masks = predictor.predict('living_room.jpeg')
#     predictor.visualize_results(image, boxes, labels, scores, masks)

#     print("Mask R-CNN 모델이 준비되었습니다.")
#     print(f"지원하는 클래스 수: {len(predictor.class_names)}")
#     print(f"사용 디바이스: {predictor.device}")

#     print("\n지원하는 주요 클래스들:")
#     for i, class_name in enumerate(predictor.class_names[1:21]):
#         print(f" {i+1}: {class_name}")

# demo_maskrcnn()





# # MaskRCNN 구축 (pretrained)
# predictor = MaskRCNNPredictor()
# image_path = 'living_room.jpeg'
# image, boxes, labels, scores, masks = predictor.predict()









# # YOLO ======================================
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# class YOLOSegmentation:
#     def __init__(self, model_name='yolov8n-seg.pt'):
#         self.model = YOLO(model_name)


#     def predict_and_visualize(self, image_path, confidence=0.5):
#         results = self.model(image_path, conf=confidence)
        
#         for r in results:
#             img = cv2.imread(image_path)
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
#             fig, axes = plt.subplots(1, 3, figsize=(18, 6))

#             axes[0].imshow(img_rgb)
#             axes[0].set_title('Original Image')
#             axes[0].axis('off')

#             img_with_boxes = r.plot()

#             axes[1].imshow(img_with_boxes)
#             axes[1].set_title('Detection Results')
#             axes[1].axis('off')

#             if r.masks is not None:
#                 masks = r.masks.data.cpu().numpy()
#                 combined_mask = np.zeros_like(img_rgb)

#                 for i, mask in enumerate(masks):
#                     mask_resized = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]))
#                     color = np.random.randint(0, 255, 3)
#                     colored_mask = np.zeros_like(img_rgb)
#                     colored_mask[mask_resized > 0.5] = color
#                     combined_mask = cv2.addWeighted(combined_mask, 1, colored_mask, 0.7, 0)
                
#                 result_img = cv2.addWeighted(img_rgb, 0.6, combined_mask, 0.4, 0)

#                 axes[2].imshow(result_img)
#                 axes[2].set_title('Segmentation Masks')
            
#             else:
#                 axes[2].text(0.5, 0.5, 'No masks detected',
#                              transform=axes[2].transAxes, ha='center', va='center')
#                 axes[2].set_title('No Segmentation Results')
#                 axes[2].axis('off')
#                 plt.tight_layout()
#                 plt.show()

#                 if r.boxes is not None:
#                     print(f"검출된 객체 수: {len(r.boxes)}")

#                     for i, box in enumerate(r.boxes):
#                         class_id = int(box.cls[0])
#                         confidence = float(box.conf[0])
#                         class_name = self.model.names[class_id]
#                         print(f"객체 {i+1}: {class_name} (신뢰도: {confidence:.2f})")
    

#     def demo_yolo_segmentation():
#         yolo_seg = YOLOSegmentation()

#         print("YOLO 세그멘테이션 모델이 준비되었습니다.")
#         print(f"지원하는 클래스: {list(yolo_seg.model.names.values())}")

#     demo_yolo_segmentation()




# # yolo 인스턴스 생성
# yolo_seg = YOLOSegmentation()


# # predict


# # visualize






# # IOU ========================================

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np


# def calculate_iou(box1, box2):
#         # box 형식: [x1, y1, x2, y2]
#         x1_inter = min(box1[0], box2[0])
#         y1_inter = min(box1[1], box2[1])
#         x2_inter = min(box1[2], box2[2])
#         y2_inter = min(box1[3], box2[3])

#         if x2_inter > x1_inter and y2_inter > y1_inter:
#             intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
#         else:
#             intersection = 0
        

#         area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#         area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

#         union = area1 + area2 - intersection
#         iou = intersection / union if union > 0 else 0

#         return iou



# def visualize_iou():
#     plt.rc('font', family='')
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     cases = [
#         {'box1': [50, 50, 150, 150], 'box2': [100, 100, 200, 200], 'title': 'IoU = 0.14'},
#         {'box1': [50, 50, 150, 150], 'box2': [75, 75, 175, 175], 'title': 'IoU = 0.36'},
#         {'box1': [50, 50, 150, 150], 'box2': [60, 60, 140, 140], 'title': 'IoU = 0.64'},
#     ]

#     plt.rc('font', family='')
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     for i, case in enumerate(cases):
#         box1, box2 = case['box1'], case['box2']
#         img = np.ones((250, 250, 3)) * 0.9
#         axes[i].imshow(img)

#         rect1 = patches.Rectangle((box1[0], box1[1]), box1[2]-box1[0], box1[3]-box1[1],
#                                 linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
#         axes[i].add_patch(rect1)
        
#         rect2 = patches.Rectangle((box2[0], box2[1]), box2[2]-box2[0], box2[3]-box2[1],
#                                 linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3)
#         axes[i].add_patch(rect2)

#     plt.tight_layout()
#     plt.show()

# visualize_iou()





# # NMS (수정하기) ==========================================
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


# def calculate_iou(box1, box2):
#         # box 형식: [x1, y1, x2, y2]
#         x1_inter = max(box1[0], box2[0])
#         y1_inter = max(box1[1], box2[1])
#         x2_inter = min(box1[2], box2[2])
#         y2_inter = min(box1[3], box2[3])

#         if x2_inter <= x1_inter and y2_inter <= y1_inter:
#             return 0.0
        

#         intersection_area = (x2_inter = x1_inter) * (y2_inter - y1_inter)

#         box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#         box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

#         union_area = box1_area + box2_area - intersection_area
        

#         return intersection_area / union_area


# def nms(boxes, scores, iou_threshold=0.5):
#     indices = np.argsort(scores)[::-1]
#     keep = []
      
#     while len(indices) > 0:
#         current = indices[0]
#         keep.append(current)

#         if len(indices) == 1:
#             break
            
#         current_box = boxes[current]
#         other_boxes = boxes[indices[1:]]

#         ious = []
#         for other_box in other_boxes:
#             iou = calculate_iou(current_box, other_box)
#             ious.append(iou)
        
#         ious = np.array(ious)

#         indices = indices[1:][ious < iou_threshold]
    
#     return keep



# def visualize_nms():
#     boxes = [
#         [50, 50, 150, 150],
#         [60, 60, 160, 160],
#         [200, 100, 300, 200],
#         [210, 110, 310, 210],
#         [70, 70, 170, 170]
#     ]



# 