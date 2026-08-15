"""
Member 3 — Audio Playback & Debounce Logic.
This module sits at the END of the pipeline: it receives a predicted string label (e.g.
'A') from the GestureClassifier and plays the matching pre-recorded audio.

Architecture:
- Per-Gesture Debounce: Each gesture has its own 2-second cooldown timer.
- Single Audio Worker Thread: A dedicated background thread processes audio sequentially 
  from a bounded queue (sie 1), ensuring no backlog and no stale gestures.
- Pre-Recorded Audio: Loads .wav files from audio-alphabet directory using scipy.
- Non-Blocking Pipeline: speak() returns instantly; ML pipeline never blocks.
"""
import soundfile as sf
import sounddevice as sd
import time
import threading
import queue
from pathlib import Path

class AudioVocalizer:
    def __init__(self, audio_directory_path="assets/audio"):
        """Initialize the audio vocalizer with per-gesture debounce and worker thread."""
        # self.gesture_last_played = {}  
        # unused: self.cooldown_seconds = 2.0  # 2-second cooldown per gesture
        self.debounce_lock = threading.Lock()  # Protects gesture_last_played
        
        # State flag: cleared during shutdown to prevent new gestures from queuing
        self.stable_frames_required = 10
        self._pending_gesture = None
        self._pending_count = 0
        self._last_spoken = None

        self.is_running = threading.Event()
        self.is_running.set()
        
        self.audio_queue = queue.Queue(maxsize=1)
        
        # Load pre-recorded audio files from configurable path
        self.audio_files = self._load_audio_files(audio_directory_path)
        
        self.worker_thread = threading.Thread(target=self._audio_worker, daemon=False)
        self.worker_thread.start()
    
    def speak(self, predicted_gesture):
      
        # 1. Normalize case immediately to prevent case-sensitivity leaks
        gesture_upper = predicted_gesture.upper()
        
        # 2. Abort immediately if no audio file exists (prevents "ghost debouncing")
        if gesture_upper not in self.audio_files:
            return

        with self.debounce_lock:
            # if the gesture running is not in the set, return
            if not self.is_running.is_set():
                return
        if gesture_upper != self._pending_gesture:
            # if current gesture is different than pending, reset pending to current and reset count
            self._pending_gesture = gesture_upper
            self._pending_count = 1
        else:
            self._pending_count += 1
        
        if self._pending_count >= self.stable_frames_required and gesture_upper != self._last_spoken:
            print(f"Vocalizing: {gesture_upper}")
            self._last_spoken = gesture_upper
            self._pending_count = 0

            sd.stop()

            try:
                self.audio_queue.put_nowait(gesture_upper)
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
                self.audio_queue.put_nowait(gesture_upper)
            

    def _load_audio_files(self, directory_path):
     
        audio_dir = Path(directory_path)
        audio_files = {}
        
        if not audio_dir.exists():
            print(f"[WARNING] Audio directory not found: {audio_dir}")
            return audio_files
        
        # Load all .wav files
        for wav_file in sorted(audio_dir.glob("*.wav")):
            gesture_label = wav_file.stem.upper() 
            try:
                audio_data, sample_rate = sf.read(str(wav_file))
                audio_files[gesture_label] = (sample_rate, audio_data)
                #print(f"[DEBUG] Loaded audio: {gesture_label} -> {wav_file.name}")
            except Exception as e:
                print(f"[ERROR] Failed to load {wav_file.name}: {e}")
        
        #print(f"[DEBUG] Loaded {len(audio_files)} audio files\n")
        return audio_files

    def _audio_worker(self):
    
        #print("[DEBUG] Audio worker started\n")
        
        while True:
            try:
                gesture = self.audio_queue.get(timeout=1.0)
                
                if gesture is None:
                    break
                
                try:
                    #print(f"[DEBUG] Playing: '{gesture}'")
                    sample_rate, audio_data = self.audio_files[gesture]
                    sd.play(audio_data, samplerate=sample_rate)
                    sd.wait()  # Block until playback completes
                    #print(f"[DEBUG] Done: '{gesture}'")
                except Exception as e:
                    print(f"[ERROR] Failed to play audio for '{gesture}': {e}")
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
        
        print("[DEBUG] Audio worker stopped\n")


    def shutdown(self):
        """Gracefully shutdown the audio worker thread."""
        self.is_running.clear()
        
        with self.debounce_lock:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            
            self.audio_queue.put_nowait(None)
        
        self.worker_thread.join(timeout=5.0)