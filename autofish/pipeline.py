from __future__ import annotations

import time
from collections import deque

from . import audit, config
from .mobilenet import AsyncMobileNetClassifier, MobileNetClassifier
from .state import FishingStateMachine, InvalidTransition
from .window_capture import WindowCapture


class RateLimiter:
    def __init__(self, fps: float) -> None:
        self.interval = 1.0 / fps
        self.next_time = time.perf_counter()

    def set_fps(self, fps: float) -> None:
        self.interval = 1.0 / fps

    def wait(self) -> None:
        self.next_time += self.interval
        now = time.perf_counter()
        sleep_sec = self.next_time - now

        if sleep_sec > 0:
            time.sleep(sleep_sec)
        else:
            self.next_time = now


class FpsMeter:
    def __init__(self) -> None:
        self.times = deque(maxlen=300)
        self.last_log = time.perf_counter()

    def tick(self, state_name: str, mode: str) -> None:
        now = time.perf_counter()
        self.times.append(now)

        if now - self.last_log < config.FPS_LOG_INTERVAL_SEC:
            return

        self.last_log = now

        if len(self.times) < 2:
            return

        duration = self.times[-1] - self.times[0]
        if duration <= 0:
            return

        fps_value = (len(self.times) - 1) / duration
        audit.fps(mode=mode, fps_value=fps_value, state=state_name)


def run() -> None:
    audit.setup()
    audit.normal("autofish starting")

    capture = WindowCapture()
    capture.activate()

    classifier = MobileNetClassifier()
    predictor = AsyncMobileNetClassifier(classifier)

    machine = FishingStateMachine(capture=capture, predictor=predictor)

    rate = RateLimiter(config.NORMAL_TARGET_FPS)
    fps_meter = FpsMeter()

    last_time = time.perf_counter()

    try:
        while True:
            target_fps = machine.target_fps
            mode = "fight" if target_fps == config.FIGHT_TARGET_FPS else "normal"
            rate.set_fps(target_fps)

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            frame = capture.grab()

            pred = None
            if machine.state.value != "fighting_with_fish":
                predictor.submit(frame)
                pred = predictor.latest()

                if pred is None:
                    pred = predictor.predict_now(frame)

            try:
                machine.step(frame_bgr=frame, pred=pred, dt=dt)
            except Exception as exc:
                audit.exception("state step failed", exc)

                if config.DEBUG:
                    raise

                if isinstance(exc, InvalidTransition):
                    machine.recover_from_transition_error(str(exc))
                else:
                    machine.recover_from_exception(str(exc))

            fps_meter.tick(state_name=machine.state.value, mode=mode)
            rate.wait()

    except KeyboardInterrupt:
        audit.normal("stopped by user")
    finally:
        machine.close()
        predictor.close()


if __name__ == "__main__":
    run()