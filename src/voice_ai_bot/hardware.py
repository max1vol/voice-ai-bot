from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Iterator

LOGGER = logging.getLogger(__name__)


class HatHardware:
    def __init__(self, button_gpio: int, led_gpio: int, pull_up: bool):
        from gpiozero import Button, LED

        self.button = Button(button_gpio, pull_up=pull_up, bounce_time=0.05)
        self.led = LED(led_gpio)
        self._blink_stop = threading.Event()
        self._blink_thread: threading.Thread | None = None
        LOGGER.info("AIY HAT GPIO configured: button=%s pull_up=%s led=%s", button_gpio, pull_up, led_gpio)

    def wait_for_press(self, timeout: float | None = None) -> bool:
        return bool(self.button.wait_for_press(timeout=timeout))

    def wait_for_release(self, timeout: float | None = None) -> bool:
        return bool(self.button.wait_for_release(timeout=timeout))

    def on(self) -> None:
        self.stop_blinking()
        self.led.on()

    def off(self) -> None:
        self.stop_blinking()
        self.led.off()

    def blink(self, on_seconds: float = 0.18, off_seconds: float = 0.18) -> None:
        self.stop_blinking()
        self._blink_stop.clear()

        def run() -> None:
            while not self._blink_stop.is_set():
                self.led.on()
                time.sleep(on_seconds)
                self.led.off()
                time.sleep(off_seconds)

        self._blink_thread = threading.Thread(target=run, name="led-blink", daemon=True)
        self._blink_thread.start()

    def stop_blinking(self) -> None:
        if self._blink_thread is None:
            return
        self._blink_stop.set()
        self._blink_thread.join(timeout=1)
        self._blink_thread = None

    @contextlib.contextmanager
    def blinking(self) -> Iterator[None]:
        self.blink()
        try:
            yield
        finally:
            self.off()

    def confirm_clear(self) -> None:
        self.stop_blinking()
        for _ in range(3):
            self.led.on()
            time.sleep(0.08)
            self.led.off()
            time.sleep(0.08)

    def signal_error(self) -> None:
        self.stop_blinking()
        for _ in range(8):
            self.led.on()
            time.sleep(0.05)
            self.led.off()
            time.sleep(0.05)
