import numpy as np
from pynput import keyboard


class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0
    vx_increment = 0.2
    vy_increment = 0.1
    dyaw_increment = 0.2

    min_vx = -1.0
    max_vx = 1.0
    min_vy = -0.1
    max_vy = 0.1
    min_dyaw = -1.0
    max_dyaw = 1.0
    camera_follow = True
    reset_requested = False

    @classmethod
    def update_vx(cls, delta):
        """update forward velocity"""
        cls.vx = np.clip(cls.vx + delta, cls.min_vx, cls.max_vx)
        print(f"vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")

    @classmethod
    def update_vy(cls, delta):
        """update lateral velocity"""
        cls.vy = np.clip(cls.vy + delta, cls.min_vy, cls.max_vy)
        print(f"vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")

    @classmethod
    def update_dyaw(cls, delta):
        """update angular velocity"""
        cls.dyaw = np.clip(cls.dyaw + delta, cls.min_dyaw, cls.max_dyaw)
        print(f"vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")

    @classmethod
    def stop_robot(cls):
        """stop robot"""
        cls.vx = 0.0
        cls.vy = 0.0
        cls.dyaw = 0.0
        print("stopping robot")
        print(f"vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}")

    @classmethod
    def toggle_camera_follow(cls):
        cls.camera_follow = not cls.camera_follow
        print(f"Camera follow: {cls.camera_follow}")

    @classmethod
    def reset(cls):
        """reset all velocities to zero"""
        cls.vx = 0.0
        cls.vy = 0.0
        cls.dyaw = 0.0
        print(
            f"Velocities reset: vx: {cls.vx:.2f}, vy: {cls.vy:.2f}, dyaw: {cls.dyaw:.2f}"
        )


def on_press(key):
    """Key press event handler"""
    try:
        # Number key controls: 8/2 control forward/backward (vx), 4/6 control left/right (vy), 7/9 control left/right turn (dyaw), 5 control stop robot, 0 control reset robot
        if hasattr(key, "char") and key.char is not None:
            c = key.char.lower()

            if c == "3":
                # 3 -> stop robot
                cmd.stop_robot()
            elif c == "8":
                # 8 -> forward (increase vx)
                cmd.update_vx(cmd.vx_increment)
            elif c == "2":
                # 2 -> backward (decrease vx)
                cmd.update_vx(-cmd.vx_increment)
            elif c == "4":
                # 4 -> left (decrease vy)
                cmd.update_vy(cmd.vy_increment)
            elif c == "6":
                # 6 -> right (increase vy)
                cmd.update_vy(-cmd.vy_increment)
            elif c == "7":
                # 7 -> turn left (increase dyaw)
                cmd.update_dyaw(cmd.dyaw_increment)
            elif c == "9":
                # 9 -> turn right (decrease dyaw)
                cmd.update_dyaw(-cmd.dyaw_increment)
            elif c == "f":
                # toggle camera follow
                cmd.toggle_camera_follow()
            elif c == "0":
                # request reset robot state in main loop (thread-safe flag)
                cmd.reset_requested = True
                print("Reset requested (0 key pressed)")
                cmd.reset()
    except AttributeError:
        pass


def on_release(key):
    """Key release event handler"""
    # If movement should only occur while keys are held down, handle it here
    pass


def start_keyboard_listener():
    """Start keyboard listener"""
    print("=" * 60)
    print("Keyboard control instructions:")
    print("  ↑ Up arrow: Increase forward speed (vx)")
    print("  ↓ Down arrow: Decrease forward speed (vx)")
    print("  ← Left arrow: Increase left turn rate (dyaw)")
    print("  → Right arrow: Increase right turn rate (dyaw)")
    print("  3 key: Stop robot")
    print("  0 key: Reset all speeds to 0")
    print("  F key: Toggle camera follow mode")
    print("=" * 60)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener
