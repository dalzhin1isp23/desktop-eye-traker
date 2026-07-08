class BaseCamera:
    def connect(self) -> bool:
        raise NotImplementedError

    def read_frame(self):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError