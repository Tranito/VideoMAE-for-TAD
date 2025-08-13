import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
import json
from natsort import natsorted
import pandas as pd
from tqdm import tqdm
import zipfile
from collections import defaultdict
import concurrent.futures
import torch
from collections import defaultdict

import os

def visualize_predictions_with_clip_and_anomaly(
    predictions,
    ground_truth,
    clip_frames,
    clip_keys,
    dataset='dada',
    clip_names="",
    phase='val',
    fps=30,
    graph_height=400,
    width=800,
    output_path=""
):
    """
    Create a video visualizing predictions and ground truth over time,
    with:
        - the clip frame at each timestep above,
        - grey-highlighted regions between collision and end of anomaly,
        - a moving red line indicating the current frame.
    
    Args:
        predictions (np.ndarray): Prediction values (argmax applied).
        ground_truth (np.ndarray): Ground truth labels (0-5, -1).
        clip_frames (np.ndarray): Array of RGB frames (T, H, W, 3).
        clip_keys (tuple): (clip_id, toa) for DADA2K.
    """
    print(f"len clip_frames: {len(clip_frames)}, len predictions: {len(predictions)}, len ground_truth: {len(ground_truth)}")
    assert len(predictions) == len(ground_truth) == len(clip_frames), \
        "Predictions, ground truth, and clip_frames must all be the same length."
    
    print

    output_dir = os.path.join(output_path, dataset, phase)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_frames = len(predictions)
    
    # Resize clip frames to desired width
    resized_clip_frames = []
    for frame in clip_frames:
        resized = cv2.resize(frame, (width, width * frame.shape[0] // frame.shape[1]))
        resized_clip_frames.append(resized)
    frame_height = resized_clip_frames[0].shape[0]
    total_height = frame_height + graph_height

    # Set up matplotlib figure and canvas
    fig, ax = plt.subplots(figsize=(width / 100, graph_height / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Setup video writer
    filename = f"{dataset}_{phase}_{clip_names.split('/')[0]}_{clip_names.split('/')[1]}_sorted.mp4"
    full_path = os.path.join(output_dir, filename)
    print(f"full output path: {full_path}")
    out = cv2.VideoWriter(full_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, total_height))

    def draw_collision_to_anomaly_background(ax, ground_truth, toa):
        """Mark background grey between collision frame and end of anomaly window"""
        # Find end of anomaly window (last frame with label -1)
        anomaly_end = -1
        for i, val in enumerate(ground_truth):
            if val == -1:
                anomaly_end = max(anomaly_end, i)
        
        if toa >= 0 and anomaly_end >= toa:
            ax.axvspan(toa, anomaly_end, facecolor='grey', alpha=0.3, label='Collision to Anomaly End')

    # Generate each video frame
    for i in range(num_frames):
        ax.clear()

        # Plot with grey background for collision-to-anomaly region
        draw_collision_to_anomaly_background(ax, ground_truth, clip_keys[1])
        
        ax.plot(predictions, label='Prediction', color='blue')
        ax.plot(ground_truth, label='Ground Truth', color='green')
        
        # Mark collision frame
        if clip_keys[1] >= 0:    
            ax.axvline(x=clip_keys[1], label='Time of Accident Frame', color='red')

        ax.axvline(x=i, color='black', linewidth=2, label='Current Frame')
        ax.set_xlim(0, num_frames)
        ax.set_ylim(0, 5)  # Updated y-limits for new label range
        ax.set_title("Accident Prediction - Time Bins Analysis")
        ax.set_xlabel('Frame number')
        ax.set_ylabel('Label Bin Number')
        ax.legend(loc='upper right', fontsize='small')

        # Render plot to image
        canvas.draw()
        buf = canvas.buffer_rgba()
        graph_img = np.asarray(buf)[..., :3]
        graph_img = cv2.resize(graph_img, (width, graph_height))

        # Get and resize current video frame
        frame = resized_clip_frames[i].copy()
        
        # Add prediction text overlay to the frame
        current_pred = predictions[i]
        label = ground_truth[i]
        
        # Create label text (green)
        if label == -1:
            label_text = "Label: -1 (masked)"
        elif label == 0:
            label_text = "Label: 0 (normal)"
        else:
            label_text = f"Label: {int(label)}s before collision"
        
        # Create prediction text (blue)
        if np.isnan(current_pred):
            pred_text = "Pred: n/a"
        elif current_pred == 0:
            pred_text = "Pred: 0 (normal)"
        else:
            pred_text = f"Pred: {int(current_pred)}s before collision"

        # Calculate text position (bottom center)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        
        # Get text sizes
        (label_width, label_height), label_baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        (pred_width, pred_height), pred_baseline = cv2.getTextSize(pred_text, font, font_scale, font_thickness)
        
        # Calculate spacing and total width
        spacing = 10  # pixels between label and prediction text
        total_width = label_width + spacing + pred_width
        text_height = max(label_height, pred_height)
        
        # Calculate starting positions (centered)
        start_x = (frame.shape[1] - total_width) // 2
        text_y = frame.shape[0] - 20  # 20 pixels from bottom
        
        label_x = start_x
        pred_x = start_x + label_width + spacing
        label_x = start_x
        pred_x = start_x + label_width + spacing
        
        # Add background rectangle for better text visibility
        cv2.rectangle(frame, 
                     (start_x - 5, text_y - text_height - 5), 
                     (start_x + total_width + 5, text_y + max(label_baseline, pred_baseline) + 5), 
                     (255, 255, 255), -1)  # White background
        
        # Add the label text (dark green)
        cv2.putText(frame, label_text, (label_x, text_y), font, font_scale, (0, 128, 0), font_thickness)  # Dark green color
        
        # Add the prediction text (blue)
        cv2.putText(frame, pred_text, (pred_x, text_y), font, font_scale, (0, 0, 255), font_thickness)  # Blue color

        # Combine the video frame (on top) with the graph (below)
        combined = np.vstack((frame, graph_img))

        # Convert RGB to BGR for OpenCV and write frame
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    out.release()
    plt.close(fig)
    print(f"Visualization saved to {output_dir}")

def read_annotations_dada(data_path, mode="train"):
    ego_categories = [str(cat) for cat in list(range(1, 19)) + [61, 62]]
    video_ext = ".png"
    clip_timesteps = []
    clip_binary_labels = []
    clip_cat_labels = []
    clip_ego = []
    clip_night = []
    clip_toa = []
    clip_acc = []
    clip_anomaly_end = []  # Add this to track anomaly end frames

    errors = []

    print(f"mode: {mode}")

    if mode == "train":
        anno_path = "DADA2K_my_split/training.txt"
    elif mode == "val":
        anno_path = "DADA2K_my_split/validation.txt"
    else:  
        anno_path = "DADA2K_my_split/validation.txt"

    with open(os.path.join(data_path, anno_path), 'r') as file:
        clip_names = [line.rstrip() for line in file]

    df = pd.read_csv(os.path.join(data_path, "annotation", "full_anno.csv"))

    for clip in tqdm(clip_names, "Part 1/2. Reading and checking clips"):
        clip_type, clip_subfolder = clip.split("/")
        row = df[(df["video"] == int(clip_subfolder)) & (df["type"] == int(clip_type))]
        info = f"clip: {clip}, type: {clip_type}, subfolder: {clip_subfolder}, rows found: {row}"
        
        assert len(row) == 1, f"Multiple results! \n{info}"
        if len(row) != 1:
            errors.append(info)
        row = row.iloc[0]
        
        with zipfile.ZipFile(os.path.join(data_path, "frames", clip, "images.zip"), 'r') as zipf:
            framenames = natsorted([f for f in zipf.namelist() if os.path.splitext(f)[1] == video_ext])
        timesteps = natsorted([int(os.path.splitext(f)[0].split("_")[-1]) for f in framenames])
        
        if_acc_video = int(row["whether an accident occurred (1/0)"])
        st = int(row["abnormal start frame"])
        en = int(row["abnormal end frame"])
        toa = int(row["accident frame"])

        # Create new labeling scheme
        new_labels = []
        if toa > -1:
            for t in range(len(timesteps)):
                if (t < toa - 150) or (t > en):
                    new_labels.append(0)  # Normal frames
                elif toa <= t <= en:
                    new_labels.append(-1)  # Collision to anomaly end
                elif toa - 150 <= t <= toa - 121:
                    new_labels.append(5)  # 5 seconds before
                elif toa - 120 <= t <= toa - 91:
                    new_labels.append(4)  # 4 seconds before
                elif toa - 90 <= t <= toa - 61:
                    new_labels.append(3)  # 3 seconds before
                elif toa - 60 <= t <= toa - 31:
                    new_labels.append(2)  # 2 seconds before
                elif toa - 30 <= t <= toa - 1:
                    new_labels.append(1)  # 1 second before
                else:
                    new_labels.append(0)  # Normal frames
        else:
            new_labels = [0 for _ in timesteps]  # All normal if no accident

        binary_labels = new_labels
        cat_labels = [l * int(clip_type) for l in binary_labels]
        if_ego = clip_type in ego_categories
        if_night = int(row["light(day,night)1-2"]) == 2

        clip_timesteps.append(timesteps)
        clip_binary_labels.append(binary_labels)
        clip_cat_labels.append(cat_labels)
        clip_ego.append(if_ego)
        clip_night.append(if_night)
        clip_toa.append(toa)
        clip_acc.append(if_acc_video)
        clip_anomaly_end.append(en)

    clip_acc = np.array(clip_acc)
    valid_idx = np.where(clip_acc == 1)[0]
    clip_names = [clip_names[i] for i in valid_idx]
    clip_timesteps = [clip_timesteps[i] for i in valid_idx]
    clip_binary_labels = [clip_binary_labels[i] for i in valid_idx]
    clip_cat_labels = [clip_cat_labels[i] for i in valid_idx]
    clip_ego = [clip_ego[i] for i in valid_idx]
    clip_night = [clip_night[i] for i in valid_idx]
    clip_toa = [clip_toa[i] for i in valid_idx]
    clip_anomaly_end = [clip_anomaly_end[i] for i in valid_idx]

    for line in errors:
        print(line)
    if len(errors) > 0:
        print(f"\n====\nerrors: {len(errors)}. You can add saving the error list in the code.")
        exit(0)

    assert len(clip_names) == len(clip_timesteps) == len(clip_binary_labels)

    return clip_names, clip_timesteps, clip_binary_labels, clip_ego, clip_night, clip_toa, clip_anomaly_end


def process_clip(clip_key, preds_grouped, labels_grouped, clip_names, clip_timesteps, folder, phase, fps, width, graph_height, output_path):
    clip_id = clip_key[0]
    clip_name = clip_names[clip_id]
    print(f"clip name: {clip_name}, clip_id: {clip_id}, clip_key: {clip_key}")

    start_idx = 45
    timesteps = [i for i in range(1, len(clip_timesteps[clip_id]) + 1)] 
    filenames = [f"{str(ts).zfill(4)}.png" for ts in timesteps]

    images = []
    with zipfile.ZipFile(os.path.join(folder, "frames", clip_name, "images.zip"), 'r') as zipf:
        for fname in filenames:
            with zipf.open(fname) as file:
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
                images.append(img)

    # Create predictions with NaN padding
    preds_cat = np.concatenate((np.array([np.nan for _ in range(start_idx)]), preds_grouped[clip_key]))
    
    # Create ground truth labels for pre-prediction frames
    pre_pred_labels = [] 
    for i in range(0, start_idx):
        if clip_key[1] - 150 <= i <= clip_key[1] - 121:
            pre_pred_labels.append(5)
        elif clip_key[1] - 120 <= i <= clip_key[1] - 91:
            pre_pred_labels.append(4)
        elif clip_key[1] - 90 <= i <= clip_key[1] - 61:
            pre_pred_labels.append(3)
        elif clip_key[1] - 60 <= i <= clip_key[1] - 31:
            pre_pred_labels.append(2)
        elif clip_key[1] - 30 <= i <= clip_key[1] - 1:
            pre_pred_labels.append(1)
        else:
            pre_pred_labels.append(0)
    
    ground_truth_cat = np.concatenate((pre_pred_labels, labels_grouped[clip_key]))

    visualize_predictions_with_clip_and_anomaly(
        preds_cat,
        ground_truth_cat,
        images,
        clip_keys=clip_key,
        clip_names=clip_name,
        dataset='dada',
        phase=phase,
        output_path=output_path,
        fps=fps,
        width=width,
        graph_height=graph_height
    )

def create_and_visualize_videos(preds, labels, clip_infos, output_path="/home/ltran/vmae_predictions", fps=30, phase="val", width=800, graph_height=400):
    preds_grouped = defaultdict(list)
    labels_grouped = defaultdict(list)

    # Apply argmax to predictions (shape: N,6 -> N)
    preds_argmax = torch.argmax(preds, dim=1)

    # Group predictions and labels by clip
    for pred, (clip_id, toa), label in zip(preds_argmax.cpu(), clip_infos.cpu(), labels.cpu()):
        preds_grouped[(int(clip_id), int(toa))].append(int(pred))
        labels_grouped[(int(clip_id), int(toa))].append(int(label))

    # Load DADA2K annotation info
    folder = "/media/datasets_sveta/DADA2000"
    clip_names, clip_timesteps, _, _, _, _, _ = read_annotations_dada(folder, mode=phase)

    # Parallelize video creation
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for clip_key in preds_grouped.keys():
            futures.append(
                executor.submit(
                    process_clip,
                    clip_key, preds_grouped, labels_grouped,
                    clip_names, clip_timesteps,
                    folder, phase, fps, width, graph_height, output_path
                )
            )
        # Wait for all to finish
        for future in concurrent.futures.as_completed(futures):
            future.result()  # To catch exceptions