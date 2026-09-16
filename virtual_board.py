# .\venv\Scripts\Activate.ps1
# .\venv\Scripts\python.exe virtual_board.py

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector


# ============================================================
# SETTINGS
# ============================================================

WIDTH = 1280
HEIGHT = 720

BRUSH_SIZE = 8
ERASER_SIZE = 40

TOP_BAR_HEIGHT = 90

# Colors in BGR format
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

# Use CAP_DSHOW on Windows for a more reliable open
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    raise SystemExit


# ============================================================
# HAND DETECTOR
# ============================================================

detector = HandDetector(
    detectionCon=0.7,
    minTrackCon=0.5,
    maxHands=1
)


# ============================================================
# CANVAS
# ============================================================

canvas = np.zeros(
    (HEIGHT, WIDTH, 3),
    dtype=np.uint8
)


# ============================================================
# DRAWING VARIABLES
# ============================================================

xp = 0
yp = 0

current_color = RED

brush_size = BRUSH_SIZE

drawing = False


# ============================================================
# UNDO SYSTEM
# ============================================================

history = []

MAX_HISTORY = 20


def save_history():
    global history
    history.append(canvas.copy())
    if len(history) > MAX_HISTORY:
        history.pop(0)


# ============================================================
# TOOLBAR BUTTONS
# ============================================================

buttons = [
    ("RED", RED),
    ("BLUE", BLUE),
    ("GREEN", GREEN),
    ("ERASER", WHITE),
    ("UNDO", None),
    ("CLEAR", None),
    ("SAVE", None)
]

button_width = 170


# ============================================================
# DRAW TOOLBAR
# ============================================================

def draw_toolbar(frame):
    for i, (name, color) in enumerate(buttons):
        x1 = i * button_width
        x2 = x1 + button_width

        cv2.rectangle(frame, (x1, 0), (x2, TOP_BAR_HEIGHT), (40, 40, 40), -1)
        cv2.rectangle(frame, (x1, 0), (x2, TOP_BAR_HEIGHT), WHITE, 2)

        if color is not None:
            cv2.rectangle(frame, (x1 + 10, 15), (x1 + 50, 55), color, -1)

        cv2.putText(
            frame, name, (x1 + 60, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2
        )


# ============================================================
# GESTURE HELPERS
# ============================================================
# NOTE: cvzone's thumb detection (fingers[0]) is unreliable and
# depends heavily on hand orientation. We deliberately IGNORE it
# and only look at index/middle/ring/pinky for gesture matching.
# This is the main fix for "drawing doesn't trigger."

def is_draw_gesture(fingers):
    # Index up, everything else (except thumb) down
    return fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0


def is_select_gesture(fingers):
    # Index + middle up, ring/pinky down
    return fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0


def is_clear_gesture(fingers):
    # All four (ignoring thumb) up
    return fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Camera not working")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Resize
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # ========================================================
    # HAND DETECTION
    # ========================================================

    hands, frame = detector.findHands(frame, draw=True)

    fingers = None

    if hands:

        hand = hands[0]
        fingers = detector.fingersUp(hand)

        # Index finger tip coordinates
        x, y = hand["lmList"][8][0:2]

        # ====================================================
        # OPEN HAND = CLEAR
        # ====================================================

        if is_clear_gesture(fingers):

            cv2.putText(
                frame, "CLEAR", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 3
            )

            canvas[:] = 0
            xp = 0
            yp = 0
            drawing = False

        # ====================================================
        # INDEX + MIDDLE = SELECT TOOL
        # ====================================================

        elif is_select_gesture(fingers):

            xp = 0
            yp = 0
            drawing = False

            cv2.circle(frame, (x, y), 15, GREEN, 3)

            if y < TOP_BAR_HEIGHT:

                button_index = x // button_width

                if button_index == 0:
                    current_color = RED
                    brush_size = BRUSH_SIZE

                elif button_index == 1:
                    current_color = BLUE
                    brush_size = BRUSH_SIZE

                elif button_index == 2:
                    current_color = GREEN
                    brush_size = BRUSH_SIZE

                elif button_index == 3:
                    current_color = BLACK
                    brush_size = ERASER_SIZE

                elif button_index == 4:
                    if len(history) > 0:
                        canvas[:] = history.pop()

                elif button_index == 5:
                    save_history()
                    canvas[:] = 0

                elif button_index == 6:
                    cv2.imwrite("virtual_board.png", canvas)
                    print("Saved as virtual_board.png")

        # ====================================================
        # INDEX ONLY = DRAW
        # ====================================================

        elif is_draw_gesture(fingers):

            cv2.circle(frame, (x, y), 10, current_color, -1)

            # Start drawing
            if xp == 0 and yp == 0:
                xp = x
                yp = y
                save_history()

            # Only draw within the canvas area (below toolbar)
            if y > TOP_BAR_HEIGHT:
                cv2.line(canvas, (xp, yp), (x, y), current_color, brush_size)

            xp = x
            yp = y
            drawing = True

        # ====================================================
        # OTHER GESTURES
        # ====================================================

        else:
            xp = 0
            yp = 0
            drawing = False

    else:
        xp = 0
        yp = 0
        drawing = False

    # ========================================================
    # CREATE WHITEBOARD DISPLAY
    # ========================================================

    white_board = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, drawing_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Cut a hole in the white background wherever something is drawn,
    # then drop the actual drawn colors into that hole.
    # (Using cv2.add here was the bug: 255 + anything saturates back
    # to 255/white, so drawn lines were invisible.)
    mask_inv = cv2.bitwise_not(drawing_mask)
    white_bg_with_hole = cv2.bitwise_and(white_board, white_board, mask=mask_inv)
    drawing_part = cv2.bitwise_and(canvas, canvas, mask=drawing_mask)

    white_board = cv2.add(white_bg_with_hole, drawing_part)

    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        white_board, "VIRTUAL WHITEBOARD",
        (WIDTH // 2 - 220, 170),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 3
    )

    cv2.putText(
        white_board, "Use your hand to write",
        (WIDTH // 2 - 170, 205),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2
    )

    # ========================================================
    # ADD CAMERA HAND OVERLAY
    # ========================================================

    overlay = cv2.addWeighted(white_board, 0.85, frame, 0.15, 0)

    # ========================================================
    # TOOLBAR
    # ========================================================

    draw_toolbar(overlay)

    # ========================================================
    # DEBUG INFO (shows exactly what gesture is being detected)
    # ========================================================

    if fingers is not None:
        cv2.putText(
            overlay, f"Fingers: {fingers}",
            (WIDTH - 320, HEIGHT - 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2
        )

    status = "DRAWING" if drawing else "IDLE"
    cv2.putText(
        overlay, f"Status: {status}",
        (WIDTH - 320, HEIGHT - 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2
    )

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    cv2.putText(
        overlay, "INDEX = DRAW", (20, HEIGHT - 75),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2
    )

    cv2.putText(
        overlay, "TWO FINGERS = TOOLS", (20, HEIGHT - 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2
    )

    cv2.putText(
        overlay, "OPEN HAND = CLEAR", (20, HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2
    )

    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow("AI Virtual Whiteboard", overlay)

    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        cv2.imwrite("virtual_board.png", canvas)
        print("Drawing saved as virtual_board.png")

    elif key == ord("c"):
        save_history()
        canvas[:] = 0
        xp = 0
        yp = 0

    elif key == ord("u"):
        if len(history) > 0:
            canvas[:] = history.pop()

    elif key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()