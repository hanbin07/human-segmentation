import cv2
import torch
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import os

# Load the Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Create folders for saving images and masks
if not os.path.exists('images'):
    os.makedirs('images')
if not os.path.exists('masks'):
    os.makedirs('masks')

# Video input setup
video_path = 'test2.mp4'  # Path to video file
cap = cv2.VideoCapture(video_path)
frame_idx = 0

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the image for model input
        image = F.to_tensor(frame).unsqueeze(0)

        # Move tensor to GPU if available
        if torch.cuda.is_available():
            image = image.to('cuda')
            model.to('cuda')

        # Perform prediction
        prediction = model(image)

        # The class ID for a person in the COCO dataset is 1
        person_class_id = 1

        # Extract and save masks corresponding to the person
        for i in range(len(prediction[0]['labels'])):
            if prediction[0]['labels'][i] == person_class_id:
                mask = prediction[0]['masks'][i, 0]
                mask = mask.mul(255).byte().cpu().numpy()
                mask = np.where(mask > 127, 1, 0).astype(np.uint8)
                
                # Resize mask to original image size
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                
                # Create file name
                frame_name = f'{frame_idx:03d}'
                
                # Save mask
                cv2.imwrite(f'masks/{frame_name}.png', mask * 255)
                
                # Apply mask for image segmentation
                result = cv2.bitwise_and(frame, frame, mask=mask)
                
                # Save the resulting image
                cv2.imwrite(f'images/{frame_name}.png', result)
                
                break  # Process only one person

        frame_idx += 1  # Increment frame index

cap.release()
print("Segmentation completed and saved.")
