import datetime
from .motorscontroller import MotorsController
from .camerascontroller import CamerasController
from .aiagent import AIAgent


class MasterController:
    def __init__(self):
        self.motors_controller = MotorsController()
        self.ai_agent = AIAgent(master_controller=self)
        self.cameras_controller = CamerasController(master_controller=self)

    def start(self):
        self.motors_controller.start()

    def stop(self):
        self.ai_agent.stop()
        self.motors_controller.stop()

    def reset(self):
        self.motors_controller.reset()
        self.cameras_controller.reset()

    def __del__(self):
        del self.motors_controller
        del self.cameras_controller