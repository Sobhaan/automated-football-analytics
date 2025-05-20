import cv2
import numpy as np
import math

# Window & arrow parameters
WINDOW_NAME = "Adjust Forward Direction (Enter=confirm, Esc=exit)"
W, H = 640, 480
CENTER = (W // 2, H // 2)
RADIUS = 150  # arrow length in pixels

# State
angle_rad = 0.0
locked = False

def draw_arrow(img, angle):
    """Draw an arrow from CENTER at the given angle."""
    end_x = int(CENTER[0] + RADIUS * math.cos(angle))
    end_y = int(CENTER[1] + RADIUS * math.sin(angle))
    cv2.arrowedLine(img, CENTER, (end_x, end_y),
                    color=(0, 255, 0), thickness=2, tipLength=0.2)
    return (end_x, end_y)

def on_mouse(event, x, y, flags, param):
    global angle_rad, locked
    if event == cv2.EVENT_MOUSEMOVE and not locked:
        # Update angle to point at mouse
        angle_rad = math.atan2(y - CENTER[1], x - CENTER[0])
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Toggle lock on click
        locked = not locked

def forward_vector():
    global angle_rad, locked

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("Move your mouse to rotate the arrow.")
    print("Left-click to lock/unlock the arrow.")
    print("Press Enter to confirm and return forward vector.")

    while True:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        tip = draw_arrow(canvas, angle_rad)

        # Compute current forward vector
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        # Draw lock status
        status = "LOCKED" if locked else "UNLOCKED"
        cv2.putText(canvas, f"[{status}]",
                    (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

        # Draw live forward vector
        vec_text = f"Vec: ({dx:+.3f}, {dy:+.3f})"
        cv2.putText(canvas, vec_text,
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 10):  # Enter
            print(f"Forward vector set to: ({dx:.4f}, {dy:.4f})")
            cv2.destroyAllWindows()
            return (dx, dy)
        elif key == 27:  # Esc
            print("Exited without confirmation.")
            cv2.destroyAllWindows()
            return None

if __name__ == "__main__":
    forward_vector()
