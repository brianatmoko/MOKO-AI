import queue
import threading

class CognitiveScheduler:
    """
    Mengatur antrean panggilan ke LLM Engine agar tidak bertabrakan.
    Mencegah 2 proses yang meminta AI berpikir bersamaan, yang bisa
    membuat RAM/CPU meledak.
    """
    def __init__(self):
        self.task_queue = queue.Queue()
        self.is_processing = False
        self._lock = threading.Lock()

    def submit_task(self, task_func, *args, **kwargs):
        """Memasukkan tugas (misal: analisis LLM) ke antrean."""
        self.task_queue.put((task_func, args, kwargs))
        self._try_process_next()

    def _try_process_next(self):
        with self._lock:
            if self.is_processing or self.task_queue.empty():
                return
            self.is_processing = True
            
        task_func, args, kwargs = self.task_queue.get()
        
        # Jalankan di thread terpisah agar scheduler tidak block
        def wrapper():
            try:
                task_func(*args, **kwargs)
            finally:
                with self._lock:
                    self.is_processing = False
                self._try_process_next()
                
        threading.Thread(target=wrapper, daemon=True).start()

# Global access point
scheduler = CognitiveScheduler()
