import cv2
import mediapipe as mp
import macmouse
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

#load pretrained model -- create a GestureRecognizer object
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

while True:
    result, image = cap.read()
    image = cv2.flip(image,1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    hand_det = hands.process(image_rgb)

    
    if hand_det.multi_hand_landmarks:
        #print(hand_det.multi_hand_landmarks[0].landmark[9])
        #change image format to get hand gestures 
        mp_img = mp.Image(image_format = mp.ImageFormat.SRGB, data = image_rgb)
        try:
            recognition_result = recognizer.recognize(mp_img).gestures[0][0].category_name
        except:
            recognition_result = None
        #print(recognizer.recognize(mp_img).gestures)
        print(recognition_result)

        x_cood = hand_det.multi_hand_landmarks[0].landmark[9].x * 1440
        y_cood = hand_det.multi_hand_landmarks[0].landmark[9].y * 900

        macmouse.move(x_cood ,y_cood)
        if recognition_result == "Closed_Fist":
            macmouse.press()
        else:
            macmouse.release()
        
        #print(x_cood, y_cood)
        for hand_landmarks in hand_det.multi_hand_landmarks:
            drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    
    cv2.imshow("slika", image)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
print("I <3 U")
cv2.destroyAllWindows()












