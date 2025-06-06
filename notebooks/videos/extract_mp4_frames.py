import cv2
import os

def extract_frames(video_path, output_folder, frame_prefix="frame", skip_frames=0):
    """
    Extracts frames from a video file and saves them as images.

    Args:
        video_path (str): Path to the input .mp4 file.
        output_folder (str): Path to the folder where extracted frames will be saved.
        frame_prefix (str): Prefix for the saved frame filenames (will eventually be prefix_00000.jpg).
                            If you want just 00000.jpg, this can be set to an empty string ""
                            and handled during formatting. We'll adjust internally to meet your requirement.
        skip_frames (int): Save one frame every N frames. 0 means save all frames.
                           For example, skip_frames=1 means save one frame, skip one, save one (i.e., frames 1, 3, 5...).
                           More commonly, it might be "save every Nth frame," which is the logic used below.
    """

    # 1. Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    # 2. Create the output folder (if it doesn't exist)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Frames will be saved in: '{output_folder}'")

    # 3. Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    # 4. Get video information (optional, but sometimes useful)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: FPS={fps:.2f}, Total Frames={total_frames}")

    # 5. Read and save frames one by one
    frame_count = 0  # Total frames read
    saved_frame_count = 0 # Actual number of frames saved (used for naming)

    while True:
        ret, frame = cap.read()

        if not ret: # If ret is False, it means the video ended or there was a read error
            break

        frame_count += 1

        if (frame_count - 1) % (skip_frames + 1) == 0:
            # Construct filename, e.g., 00000.jpg, 00001.jpg ...
            # Use 5 digits, zero-padded at the beginning.
            output_filename = f"{saved_frame_count:05d}.jpg"
            output_path = os.path.join(output_folder, output_filename)

            # Save the frame
            cv2.imwrite(output_path, frame)
            # print(f"Saved: {output_path}") # You can uncomment this to see each frame being saved
            saved_frame_count += 1

    # 6. Release resources
    cap.release()
    # cv2.destroyAllWindows() # If any display windows were opened

    print(f"\nProcessing complete. Total frames read: {frame_count}.")
    print(f"Successfully saved {saved_frame_count} frames to '{output_folder}'.")


if __name__ == "__main__":
    video_file = "02_juggle.mp4"  # <--- Modify this to your video file path
    output_dir = "02_juggle" # <--- Modify this to your desired output folder for frames

    extract_frames(video_file, output_dir, skip_frames=0)
