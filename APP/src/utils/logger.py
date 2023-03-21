from datetime import datetime

class CLOG():
    def __init__(self, processName, timed = False, silence = False) -> None:
        self.processName = processName
        self.silence = silence
        self.timed = timed

    def log(self, msg):
        if not self.silence:
            if self.timed:
                _time = datetime.now().strftime("%H:%M:%S")
                print(f"[{_time}] __{self.processName}__ {msg}")
            else:
                print(f"__{self.processName}__ {msg}")