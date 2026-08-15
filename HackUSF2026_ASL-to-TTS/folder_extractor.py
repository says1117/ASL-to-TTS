import cv2
import os
import csv
from src.tracker import HandTracker

# ── Why initialize once outside the loop? ────────────────────────────────────
# HandTracker loads a MediaPipe .task model file from disk on every __init__
# call. Loading a model is expensive (reads file, allocates memory, compiles
# inference graph). Doing it once here and reusing it for every image keeps
# the script fast.
tracker = HandTracker()

# ── Paths ─────────────────────────────────────────────────────────────────────
# os.path.dirname(__file__) gives us the folder this script lives in,
# so the paths work no matter where you launch the script from.
# For Kaggle dataset: os.path.join(os.path.dirname(__file__), "asl_alphabet_train", "asl_alphabet_train")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "asl_alphabet_train", "asl_alphabet_train")

# Points at the same file train_model() reads, so running this script
# once is all you need before launching main.py.
# For Kaggle dataset, use: os.path.join(os.path.dirname(__file__), "assets", "data", "asl_landmarks_kaggle.csv")
OUTPUT_CSV  = os.path.join(os.path.dirname(__file__), "assets", "data", "asl_landmarks_kaggle.csv")

# ── Open the CSV once and keep it open for the entire run ────────────────────
# "w" mode creates the file if it doesn't exist and OVERWRITES it if it does
# (clean slate every run). newline="" is required on Windows to prevent csv
# from inserting blank lines between rows.
total_extracted = 0

with open(OUTPUT_CSV, "w", newline="") as csv_file:

    writer = csv.writer(csv_file)

    # Write the header row: x0, x1, … x62, label
    # Must match the column names save_data() and train_model() use — both
    # use "x0…x62". Using a different prefix (e.g. "v0") would cause
    # train_model()'s df.drop(columns=["label"]) to still work, but mixing
    # two naming conventions in the same CSV would corrupt any future merge.
    writer.writerow([f"x{i}" for i in range(63)] + ["label"])

    # ── Outer loop: one iteration per gesture label (subfolder) ──────────────
    # os.listdir() returns all names inside DATASET_DIR.
    # sorted() makes the terminal output appear in alphabetical order.
    for label in sorted(os.listdir(DATASET_DIR)):

        label_dir = os.path.join(DATASET_DIR, label)

        # Guard: os.listdir() includes files too (e.g. a stray .DS_Store).
        # We only want real subfolders — skip anything that isn't a directory.
        if not os.path.isdir(label_dir):
            continue

        # Collect only image files so we don't accidentally feed MediaPipe a
        # .txt or .DS_Store. A tuple is slightly faster than a list for `in`.
        images = [
            f for f in os.listdir(label_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        total = len(images)

        # ── Inner loop: one iteration per image in this label's folder ───────
        for idx, filename in enumerate(images, start=1):

            # Build the full path to this specific image file
            img_path = os.path.join(label_dir, filename)

            # cv2.imread() reads the image into a NumPy array (BGR format).
            # Returns None if the file is corrupt or not a valid image — we
            # guard against that on the next line.
            frame = cv2.imread(img_path)
            if frame is None:
                continue  # Skip unreadable files silently

            # process_frame() returns (landmark_array, display_frame).
            # We only need landmark_array here; "_" discards the annotated
            # frame because we have no window to display it in batch mode.
            landmark_array, _ = tracker.process_frame(frame)

            # ── Crucial skip ─────────────────────────────────────────────────
            # MediaPipe couldn't find a hand in this image (bad angle, blur,
            # cropping, lighting). landmark_array will be an empty list [].
            # Writing an empty row would corrupt the CSV, so we skip it.
            if not landmark_array:
                continue

            # Append one row: 63 floats followed by the label string.
            # landmark_array is already a flat list of 63 floats from tracker.
            writer.writerow(landmark_array + [label])
            total_extracted += 1

            # ── Progress message ──────────────────────────────────────────────
            # \r moves the cursor back to the start of the current line so
            # each update OVERWRITES the previous one instead of scrolling.
            # end="" suppresses the automatic newline print() adds.
            # flush=True forces the output buffer to display immediately
            # (Python buffers stdout by default — without this you'd see
            # nothing until the buffer fills or the script finishes).
            print(f"\rProcessing {label}: {idx}/{total} images...", end="", flush=True)

        # Print a newline after finishing each label so the next label's
        # progress message starts on a fresh line.
        print()

print(f"✓ Done. Extracted {total_extracted} hand landmarks from dataset.")
print(f"  Saved to: {OUTPUT_CSV}")
