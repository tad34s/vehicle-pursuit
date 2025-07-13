from pynput import keyboard


class KeyboardListener:
    def __init__(self):
        self.pressed = False
        self.keyboard = keyboard.Listener(self.on_press)

    def reset(self):
        self.pressed = False

    def was_pressed(self) -> bool:
        return self.pressed

    def start(self):
        self.keyboard.start()

    def end(self):
        self.keyboard.join()

    def on_press(self, key):
        if key == keyboard.Key.esc:
            self.pressed = False
            print("---- Will not save next model")
        try:
            k = key.char
        except AttributeError:
            return

        if k == "s":
            self.pressed = True
            print("---- Will save next model")

        return
