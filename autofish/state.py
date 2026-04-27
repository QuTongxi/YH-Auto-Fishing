from __future__ import annotations

import time
from enum import Enum

import pydirectinput

from . import audit, config
from .fight import FightController
from .mobilenet import AsyncMobileNetClassifier, Prediction
from .window_capture import WindowCapture


class RunState(str, Enum):
    WAITING_FOR_START = "waiting_for_start"
    WAITING_FOR_FISH = "waiting_for_fish"
    FIGHTING_WITH_FISH = "fighting_with_fish"
    AFTER_FIGHTING = "after_fighting"
    POST_FIGHT_VERIFY = "post_fight_verify"


class Vision(str, Enum):
    WAITING_FOR_START = "waiting_for_start"
    WAITING_FOR_FISH = "waiting_for_fish"
    CATCHING_FISH = "catching_fish"
    FIGHTING_WITH_FISH = "fighting_with_fish"
    AFTER_FIGHTING = "after_fighting"
    NONSENSE = "nonsense"


class StateMachineError(RuntimeError):
    pass


class InvalidTransition(StateMachineError):
    pass


class FightDetectionFailure(StateMachineError):
    pass


class PostFightTimeout(StateMachineError):
    pass

def normalize_label(label: str) -> str:
    return label.strip().replace(" ", "")

def normalize_label(label: str) -> str:
    return label.strip().replace(" ", "")


def vision_from_prediction(pred: Prediction) -> Vision:
    label = normalize_label(pred.label)

    if label == normalize_label(config.LABEL_WAITING_FOR_START):
        return Vision.WAITING_FOR_START
    if label == normalize_label(config.LABEL_WAITING_FOR_FISH):
        return Vision.WAITING_FOR_FISH
    if label == normalize_label(config.LABEL_CATCHING_FISH):
        return Vision.CATCHING_FISH
    if label == normalize_label(config.LABEL_FIGHTING_WITH_FISH):
        return Vision.FIGHTING_WITH_FISH
    if label == normalize_label(config.LABEL_AFTER_FIGHTING):
        return Vision.AFTER_FIGHTING
    if label == normalize_label(config.LABEL_NONSENSE):
        return Vision.NONSENSE

    raise ValueError(f"Unknown model label: raw={pred.label!r}, normalized={label!r}")


def press_key(key: str) -> None:
    pydirectinput.keyDown(key)
    time.sleep(config.KEY_PRESS_SEC)
    pydirectinput.keyUp(key)


def click_after_fight(capture: WindowCapture) -> None:
    x, y = capture.client_center_screen()
    time.sleep(config.AFTER_FIGHT_CLICK_DELAY_SEC)

    for _ in range(config.AFTER_FIGHT_CLICK_COUNT):
        pydirectinput.click(x=x, y=y)


class FishingStateMachine:
    def __init__(
        self,
        capture: WindowCapture,
        predictor: AsyncMobileNetClassifier,
    ) -> None:
        self.capture = capture
        self.predictor = predictor
        self.fight = FightController()
        self.state = RunState.WAITING_FOR_START
        self.post_fight_started_at = 0.0
        self.last_start_press_at = 0.0
        self.last_action_at = 0.0
        self.last_after_fight_click_at = 0.0


    def close(self) -> None:
        self.fight.close()

    @property
    def target_fps(self) -> float:
        if self.state == RunState.FIGHTING_WITH_FISH:
            return config.FIGHT_TARGET_FPS
        return config.NORMAL_TARGET_FPS

    def _set_state(self, new_state: RunState, reason: str) -> None:
        if new_state != self.state:
            audit.transition(self.state.value, new_state.value, reason)
            self.state = new_state

    def step(self, frame_bgr, pred: Prediction | None, dt: float) -> None:
        if self.state == RunState.FIGHTING_WITH_FISH:
            self._step_fighting(frame_bgr, dt)
            return

        if pred is None:
            return

        vision = vision_from_prediction(pred)

        if pred.confidence < config.MIN_MODEL_CONFIDENCE:
            audit.debug(f"low confidence ignored: {pred.label} {pred.confidence:.3f}")
            return

        if self.state == RunState.WAITING_FOR_START:
            self._step_waiting_for_start(vision)
        elif self.state == RunState.WAITING_FOR_FISH:
            self._step_waiting_for_fish(vision)
        elif self.state == RunState.AFTER_FIGHTING:
            self._step_after_fighting(vision)
        elif self.state == RunState.POST_FIGHT_VERIFY:
            self._step_post_fight_verify(vision)
        else:
            raise StateMachineError(f"Unknown state: {self.state}")

    def _click_after_fighting_once(self, reason: str) -> None:
        click_after_fight(self.capture)
        self._mark_action()
        self.last_after_fight_click_at = time.perf_counter()
        audit.debug(f"after_fighting clicked: {reason}")

    def _step_waiting_for_start(self, vision: Vision) -> None:
        if vision == Vision.NONSENSE:
            return

        if vision == Vision.WAITING_FOR_START:
            now = time.perf_counter()

            press_key(config.KEY_START_OR_CATCH)

            self.last_start_press_at = now
            self._mark_action()

            self._set_state(RunState.WAITING_FOR_FISH, "press F at waiting_for_start")
            return

        self._ignore_or_raise(
            f"state={self.state.value}, vision={vision.value}, expected=waiting_for_start"
        )


    def _step_waiting_for_fish(self, vision: Vision) -> None:
        if vision in {Vision.NONSENSE, Vision.WAITING_FOR_FISH}:
            return

        if vision == Vision.WAITING_FOR_START:
            now = time.perf_counter()
            elapsed_from_press = now - self.last_start_press_at

            if self._in_transition_grace():
                audit.debug(
                    f"waiting_for_fish sees waiting_for_start during grace: "
                    f"elapsed_from_action={now - self.last_action_at:.3f}s"
                )
                return

            if elapsed_from_press >= config.START_RETRY_SEC:
                audit.debug(
                    f"waiting_for_fish still sees waiting_for_start, retry F: "
                    f"elapsed_from_press={elapsed_from_press:.3f}s"
                )

                press_key(config.KEY_START_OR_CATCH)

                self.last_start_press_at = now
                self._mark_action()
                return

            return

        if vision == Vision.CATCHING_FISH:
            press_key(config.KEY_START_OR_CATCH)

            self._mark_action()
            self.fight.reset()

            self._set_state(RunState.FIGHTING_WITH_FISH, "press F at catching_fish")
            return

        self._ignore_or_raise(
            f"state={self.state.value}, vision={vision.value}, expected=waiting_for_fish/catching_fish"
        )

    def _step_fighting(self, frame_bgr, dt: float) -> None:
        result = self.fight.step(frame_bgr, dt)

        if result.status == "active":
            return

        self.fight.reset()

        pred = self.predictor.predict_now(frame_bgr)
        vision = vision_from_prediction(pred)

        if vision == Vision.FIGHTING_WITH_FISH:
            raise FightDetectionFailure(
                "Traditional fight detector failed: model still predicts fighting_with_fish"
            )

        if vision == Vision.AFTER_FIGHTING:
            self._click_after_fighting_once(
                "after_fighting after fight detector release"
            )
            self._set_state(
                RunState.AFTER_FIGHTING,
                "after_fighting after fight detector release"
            )
            return

        if vision in {
            Vision.WAITING_FOR_START,
            Vision.WAITING_FOR_FISH,
            Vision.NONSENSE,
        }:
            self.post_fight_started_at = time.perf_counter()
            self._set_state(RunState.POST_FIGHT_VERIFY, f"fight released, vision={vision.value}")
            return

        raise InvalidTransition(
            f"fight released but got invalid vision={vision.value}"
        )

    def _step_after_fighting(self, vision: Vision) -> None:
        if vision == Vision.AFTER_FIGHTING:
            if self._in_transition_grace():
                audit.debug(
                    f"after_fighting remains during grace: "
                    f"elapsed={time.perf_counter() - self.last_action_at:.3f}s"
                )
                return

            self._click_after_fighting_once("after_fighting still visible")
            return

        if vision == Vision.WAITING_FOR_START:
            self._set_state(
                RunState.WAITING_FOR_START,
                "after_fighting closed, now waiting_for_start"
            )
            return

        if vision == Vision.NONSENSE:
            return

        self._ignore_or_raise(
            f"state={self.state.value}, vision={vision.value}, expected=after_fighting/waiting_for_start"
        )

    def _step_post_fight_verify(self, vision: Vision) -> None:
        elapsed = time.perf_counter() - self.post_fight_started_at

        if vision == Vision.AFTER_FIGHTING:
            self._click_after_fighting_once("after_fighting verified")
            self._set_state(RunState.AFTER_FIGHTING, "after_fighting verified")
            return

        if vision in {
            Vision.WAITING_FOR_START,
            Vision.WAITING_FOR_FISH,
            Vision.NONSENSE,
        }:
            if elapsed > config.POST_FIGHT_VERIFY_TIMEOUT_SEC:
                raise PostFightTimeout("post-fight verification timed out")
            return

        self._ignore_or_raise(
            f"post_fight_verify got invalid vision={vision.value}"
        )

    def recover_from_exception(self, reason: str) -> None:
        audit.warn(f"enter recovery: {reason}")

        start = time.perf_counter()

        while time.perf_counter() - start < config.ERROR_RECOVERY_TIMEOUT_SEC:
            frame = self.capture.grab()
            pred = self.predictor.predict_now(frame)
            vision = vision_from_prediction(pred)

            if vision == Vision.WAITING_FOR_START:
                self.fight.reset()
                self._set_state(RunState.WAITING_FOR_START, "recovered: waiting_for_start")
                return

            if vision == Vision.AFTER_FIGHTING:
                self.fight.reset()
                click_after_fight(self.capture)
                self._mark_action()
                self._set_state(RunState.WAITING_FOR_START, "recovered: after_fighting clicked")
                return

            time.sleep(1.0 / config.NORMAL_TARGET_FPS)

        raise StateMachineError(f"recovery timeout after exception: {reason}")

    def recover_from_transition_error(self, reason: str) -> None:
        audit.warn(
            f"transition error recovery: {reason}, wait 1.8s -> press F -> wait 0.2s -> resync by model"
        )
        time.sleep(1.8)
        press_key(config.KEY_START_OR_CATCH)
        self._mark_action()
        time.sleep(0.2)

        frame = self.capture.grab()
        pred = self.predictor.predict_now(frame)
        vision = vision_from_prediction(pred)
        self._sync_state_from_vision(vision, reason=f"transition error resync: {reason}")

    def _mark_action(self) -> None:
        self.last_action_at = time.perf_counter()


    def _in_transition_grace(self) -> bool:
        if self.last_action_at <= 0:
            return False

        elapsed = time.perf_counter() - self.last_action_at
        return elapsed <= config.TRANSITION_GRACE_SEC


    def _ignore_or_raise(self, message: str) -> None:
        if self._in_transition_grace():
            audit.debug(f"transition grace ignored: {message}")
            return

        raise InvalidTransition(message)

    def _sync_state_from_vision(self, vision: Vision, reason: str) -> None:
        self.fight.reset()

        if vision == Vision.WAITING_FOR_START:
            self._set_state(RunState.WAITING_FOR_START, reason)
            return

        if vision == Vision.WAITING_FOR_FISH:
            self._set_state(RunState.WAITING_FOR_FISH, reason)
            return

        if vision in {Vision.CATCHING_FISH, Vision.FIGHTING_WITH_FISH}:
            self._set_state(RunState.FIGHTING_WITH_FISH, reason)
            return

        if vision == Vision.AFTER_FIGHTING:
            self._set_state(RunState.AFTER_FIGHTING, reason)
            return

        audit.warn(f"resync ignored because vision is nonsense: {reason}")


