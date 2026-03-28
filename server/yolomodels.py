import datetime
from ultralytics import YOLO
from pathlib import Path

class YOLOModels:
    """
    Singleton class to manage YOLO models.
    """
    # Model names available for selection (no instance required).
    MODEL_NAMES = [
        "Simple and Fast",
        "Mid. Quality Mid. Speed",
        "Good Quality Moderate Speed",
        "Accurate and Slow",
        "Very Accurate and Very Slow"
    ]
    DEFAULT_MODEL_NAME = "Simple and Fast"

    _instance = None
    # cached model name, model type and model instance
    _cached_model_name = None
    _cached_model_type = None
    _cached_model_device = None
    _cached_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YOLOModels, cls).__new__(cls)
        return cls._instance

    def _set_cached_model(self, *, model_name: str, model_type: str, device: str = "cpu",  model_filename: str = None):
        #path of a directory of this file
        this_file_path = Path(__file__).resolve()
        model_dir = this_file_path.parent / "ai_models" / "yolo"
        #check if the model filename is valid
        if model_filename is None:
            print(f"Model filename is not valid: {model_filename}")
            return None
        #check if the model filename exists
        if not (model_dir / model_filename).exists():
            print(f"Model filename does not exist: {model_filename}")
            return None

        self._cached_model_name = model_name
        self._cached_model_type = model_type
        self._cached_model_device = device
        self._cached_model = YOLO(model_dir / model_filename)
        self._cached_model.to(device=device)
        return self._cached_model

    def convert_model(self):
        for model_name in YOLOModels.MODEL_NAMES:
            model = self.get_model(model_name=model_name, device="cpu")
            model.export(format="ncnn")
            model.export(format="engine")
            model.export(format="openvino")

    def get_model(self, *,model_name: str, device: str = "cpu") -> YOLO:
        # default model type is pt
        model_type = "pt"
        device = (device + " ").split(" ")[0]
        # if the device is arm_cpu, set the model type to ncnn
        if device == "arm_cpu":
            device = "cpu"
            model_type = "ncnn"
        # elif device == "cuda:0":
        #     device = "cuda:0"
        #     model_type = "ncnn"
        # if the model name and type are the same as the cached model, return the cached model
        if self._cached_model_name == model_name and self._cached_model_type == model_type and self._cached_model is not None:
            return self._cached_model
        # get the model from the model name and type
        if model_type == "pt":
            if model_name == YOLOModels.MODEL_NAMES[0]:
                return self._set_cached_model(model_name=model_name, model_type=model_type, device=device, model_filename="yolo26n-pose.pt")
            elif model_name == YOLOModels.MODEL_NAMES[1]:
                return self._set_cached_model(model_name=model_name, model_type=model_type, device=device, model_filename="yolo26s-pose.pt")
            elif model_name == YOLOModels.MODEL_NAMES[2]:
                return self._set_cached_model(model_name=model_name, model_type=model_type, device=device, model_filename="yolo26m-pose.pt")
            elif model_name == YOLOModels.MODEL_NAMES[3]:
                return self._set_cached_model(model_name=model_name, model_type=model_type, device=device, model_filename="yolo26l-pose.pt")
            elif model_name == YOLOModels.MODEL_NAMES[4]:
                return self._set_cached_model(model_name=model_name, model_type=model_type, device=device, model_filename="yolo26x-pose.pt")
            else:
                print(f"Invalid model name: {model_name}")
                return self._set_cached_model(model_name=model_name, model_type=model_type, device=device, model_filename="yolo26n-pose.pt")
        elif model_type == "ncnn":
            if model_name == YOLOModels.MODEL_NAMES[0]:
                return self._set_cached_model(model_name=model_name, model_type=model_type, device=device, model_filename="yolo26n-pose_ncnn_model")
            elif model_name == YOLOModels.MODEL_NAMES[1]:
                return self._set_cached_model(model_name=model_name, model_type=model_type, device=device, model_filename="yolo26s-pose_ncnn_model")
            elif model_name == YOLOModels.MODEL_NAMES[2]:
                return self._set_cached_model(model_name=model_name, model_type=model_type, device=device, model_filename="yolo26m-pose_ncnn_model")
            elif model_name == YOLOModels.MODEL_NAMES[3]:
                return self._set_cached_model(model_name=model_name, model_type=model_type, device=device, model_filename="yolo26l-pose_ncnn_model")
            elif model_name == YOLOModels.MODEL_NAMES[4]:
                return self._set_cached_model(model_name=model_name, model_type=model_type, device=device, model_filename="yolo26x-pose_ncnn_model")
            else:
                print(f"Invalid model name: {model_name}")
                return self._set_cached_model(model_name=model_name, model_type=model_type, device=device, model_filename="yolo26n-pose_ncnn_model")
        else:
            print(f"Invalid model type: {model_type}. The only valid types are 'pt' and 'ncnn'.")
            return self._set_cached_model(model_name=model_name, model_type=model_type, device=device, model_filename="yolo26n-pose.pt")
                
    @property
    def default_model_name(self) -> str:
        return YOLOModels.MODEL_NAMES[0]
