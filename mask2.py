import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import os


model = deeplabv3_resnet50(pretrained=True).eval()


if not os.path.exists('images'):
    os.makedirs('images')
if not os.path.exists('masks'):
    os.makedirs('masks')


video_path = 'test.mp4'  
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(0)  


    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')


    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)


    person_class_id = 15


    mask = output_predictions == person_class_id


    mask = mask.cpu().numpy()
    mask = cv2.resize(mask.astype('uint8'), (frame.shape[1], frame.shape[0]))


    result = cv2.bitwise_and(frame, frame, mask=mask)


    frame_name = f'{frame_idx:03d}'
    

    cv2.imwrite(f'images/{frame_name}.png', result)

   
    cv2.imwrite(f'masks/{frame_name}.png', mask * 255)

    frame_idx += 1  

cap.release()
print("Segmentation completed and saved.")
