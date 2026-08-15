import os
import io
import threading
import cv2
import pandas as pd
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from src.tracker import HandTracker
from src.neural_net import ASLNeuralNetwork
from src.vocalizer import AudioVocalizer
from google import genai
from elevenlabs.client import ElevenLabs

load_dotenv()

STABLE_FRAMES = 20  # frames a gesture must be held to register. makes it so that the sound is not spammed repeatedly

def main():
    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    eleven_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    vocalizer = AudioVocalizer()

    csv_path = os.path.join(os.path.dirname(__file__), 'assets', 'data', 'asl_landmarks_kaggle.csv')
    df = pd.read_csv(csv_path)
    num_classes = len(df[df['label'].str.len() == 1]['label'].unique())
    neural_net = ASLNeuralNetwork(num_classes=num_classes)

    neural_net.train_model()
    print("Model trained. Press 'W' to start/stop word recording. Press 'q' to quit.")

    def process_word(letters):
        # This runs on a background thread so the camera loop never pauses.
        # All three blocking operations (network call, TTS, audio playback)
        # happen here while the main thread keeps reading frames.
        print(f"Sending to Gemini: {letters}")
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                f"You are a highly advanced autocorrect engine. Your task is to process a single input string and return one valid, real English word."
                    "Rules:"
                    "1. If the input is already a valid English word, return it exactly as is."
                    "2. If the input is misspelled, jumbled, or a non-word (e.g., 'Vello'), map it to the closest real English word based on character similarity."
                    "3. THE FREQUENCY RULE: If there are multiple valid English words that match the misspelling (e.g., 'Hello', 'Cello', 'Jello'), you MUST select the word that has the highest usage frequency in everyday conversational English."
                    "4. Output STRICTLY the final corrected word and absolutely nothing else. No punctuation, no explanation, no conversational filler."
                    f"Input: '{letters}'."
            )
        )
        word = response.text.strip()
        print(f"Gemini corrected word: {word}")

        audio = eleven_client.text_to_speech.convert(
            text=word,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_monolingual_v1",
        )
        audio_bytes = b"".join(audio)
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        sd.play(data, samplerate=samplerate)
        sd.wait()

    recording = False
    letter_buffer = []
    pending_letter = None
    pending_count = 0
    last_added = None

    # Non-recording mode repeat tracking.
    # Tracks the last gesture spoken and which frame it was spoken on so we
    # can allow the same letter to repeat after a 20-frame gap.
    last_spoken_gesture = None
    last_spoken_frame   = -20   # start at -20 so frame 0 is always eligible
    frame_count         = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        landmark_array, display_frame = tracker.process_frame(frame)

        if landmark_array:
            gesture = neural_net.predict(landmark_array)

            if gesture:
                if not recording:
                    cv2.putText(display_frame, f"Gesture: {gesture}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                    # If the same gesture has been held for 20+ frames since it
                    # was last spoken, clear the vocalizer's memory of it so the
                    # repeat is allowed through on the next speak() call.
                    frames_since_last = frame_count - last_spoken_frame
                    if gesture == last_spoken_gesture and frames_since_last >= 20:
                        vocalizer._last_spoken = None

                    # Always call speak() every frame so the vocalizer's internal
                    # _pending_count can accumulate to its threshold (10 frames).
                    # Gating this call was the bug — the counter never reached 10.
                    vocalizer.speak(gesture)

                    # Track when this gesture was spoken so we can time the repeat.
                    # We update this only when the vocalizer actually fires, which
                    # happens inside speak() once _pending_count hits the threshold.
                    if gesture != last_spoken_gesture:
                        last_spoken_gesture = gesture
                        last_spoken_frame   = frame_count
                    elif vocalizer._last_spoken == gesture:
                        last_spoken_frame = frame_count
                else:
                    # Recording mode: accumulate letters, no vocalization
                    if gesture == pending_letter:
                        pending_count += 1
                    else:
                        pending_letter = gesture
                        pending_count = 1
                        last_added = None

                    if pending_count >= STABLE_FRAMES and gesture != last_added:
                        letter_buffer.append(gesture)
                        last_added = gesture
                        print(f"Letter: {gesture} -> {''.join(letter_buffer)}")
            else:
                if recording:
                    pending_letter = None
                    pending_count = 0
                    last_added = None

        # HUD
        if recording:
            cv2.putText(display_frame, "[ RECORDING ]", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(display_frame, ''.join(letter_buffer), (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            cv2.putText(display_frame, "Press W to record a word, press again to stop recording", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        cv2.imshow("ASL Translator", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('w'):
            if not recording:
                recording = True
                letter_buffer = []
                pending_letter = None
                pending_count = 0
                last_added = None
                print("Recording started...")
            else:
                recording = False
                if letter_buffer:
                    letters = ''.join(letter_buffer)
                    # daemon=True means this thread won't block the program
                    # from exiting if the user presses 'q' mid-playback.
                    threading.Thread(
                        target=process_word, args=(letters,), daemon=True
                    ).start()
                else:
                    print("No letters recorded.")

    vocalizer.shutdown()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
