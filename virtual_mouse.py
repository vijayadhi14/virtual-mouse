import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Smoothening params
smoothening = 7
plocx, plocy = 0, 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            index_x, index_y = 0, 0
            thumb_x, thumb_y = 0, 0

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index finger tip
                    cv2.circle(frame, (x, y), 12, (0, 255, 255), cv2.FILLED)
                    index_x = (screen_width / frame_width) * x
                    index_y = (screen_height / frame_height) * y

                    # Smoothened mouse movement
                    clocx = plocx + (index_x - plocx) / smoothening
                    clocy = plocy + (index_y - plocy) / smoothening
                    pyautogui.moveTo(clocx, clocy)
                    plocx, plocy = clocx, clocy

                if id == 4:  # Thumb tip
                    cv2.circle(frame, (x, y), 12, (0, 0, 255), cv2.FILLED)
                    thumb_x = (screen_width / frame_width) * x
                    thumb_y = (screen_height / frame_height) * y

            # Click if index & thumb are close
            if abs(index_y - thumb_y) < 40:
                pyautogui.click()
                time.sleep(0.2)  # debounce click

    cv2.imshow("Virtual Mouse", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
