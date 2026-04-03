from .motorscontroller import MotorsController
from .camerascontroller import CamerasController
from .aiagent import AIAgent


class MasterController:
    def __init__(self):
        self._started = False
        self.motors_controller = MotorsController()
        self.ai_agent = AIAgent(master_controller=self)
        self.cameras_controller = CamerasController(master_controller=self)

    def start(self):
        if self._started:
            return
        self.motors_controller.start()
        self.ai_agent.start()
        self._started = True

    def stop(self):
        shutdown_errors: list[tuple[str, Exception]] = []
        try:
            for component_name, stop_component in (
                ("camera workers", self.cameras_controller.stop),
                ("AI agent", self.ai_agent.stop),
                ("motors controller", self.motors_controller.stop),
            ):
                try:
                    stop_component()
                except Exception as exc:
                    print(f"Error stopping {component_name}: {exc}")
                    shutdown_errors.append((component_name, exc))
        finally:
            self._started = False

        if shutdown_errors:
            failed_components = ", ".join(name for name, _ in shutdown_errors)
            raise RuntimeError(
                f"Master controller shutdown failed for: {failed_components}"
            ) from shutdown_errors[0][1]

    def reset(self):
        self.motors_controller.reset()
        self.cameras_controller.reset()

    def __del__(self):
        del self.motors_controller
        del self.cameras_controller