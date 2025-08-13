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
    clip_frames,  # expects a list or array of RGB frames (H, W, 3)
    clip_keys,
    dataset='dota',
    clip_names = "",
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
        - orange-highlighted anomaly regions,
        - a moving red line indicating the current frame.
    
    Args:
        predictions (np.ndarray): Prediction values.
        ground_truth (np.ndarray): Ground truth binary labels (0 or 1) for DoTA 
                                    and tuple with ground truth binary labels and time of accident frame for DADAD2K.
        clip_frames (np.ndarray): Array of RGB frames (T, H, W, 3) corresponding to the video clip.
        output_path (str): Path to save output video.
        fps (int): Frames per second of output.
        graph_height (int): Height in pixels for the graph.
        width (int): Width of output video.
    """
    print(f"len clip_frames: {len(clip_frames)}, len predictions: {len(predictions)}, len ground_truth: {len(ground_truth)}")
    assert len(predictions) == len(ground_truth) == len(clip_frames), \
        "Predictions, ground truth, and clip_frames must all be the same length."
    
    output_dir = os.path.join(output_path, dataset, phase)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_frames = len(predictions)
    
    # Resize clip frames to desired width
    resized_clip_frames = []
    for frame in clip_frames:
        resized = cv2.resize(frame, (width, width * frame.shape[0] // frame.shape[1]))  # preserve aspect ratio
        resized_clip_frames.append(resized)
    frame_height = resized_clip_frames[0].shape[0]
    total_height = frame_height + graph_height

    # Set up matplotlib figure and canvas
    fig, ax = plt.subplots(figsize=(width / 100, graph_height / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Setup video writer

    print(f"clip_keys: {clip_keys}, type: {type(clip_keys)}")

    if isinstance(clip_keys, tuple):
        filename = f"{dataset}_{phase}_{clip_names.split("/")[0]}_{clip_names.split("/")[1]}_sorted.mp4"
    else:
        filename = f"{dataset}_{phase}_{clip_names}_sorted.mp4"

    print(f"filename: {filename}")
    full_path = os.path.join(output_dir, filename)
    print(f"full output path: {os.path.join(output_dir, filename)}")
    out = cv2.VideoWriter(full_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, total_height))

    def draw_anomaly_background(ax, ground_truth):
        in_anomaly = False
        start = 0
        for i, val in enumerate(ground_truth):
            if val == 1 and not in_anomaly:
                start = i
                in_anomaly = True
            elif val == 0 and in_anomaly:
                ax.axvspan(start, i, facecolor='orange', alpha=0.3)
                in_anomaly = False
        if in_anomaly:
            ax.axvspan(start, len(ground_truth), facecolor='orange', alpha=0.3)

    # Generate each video frame
    for i in range(num_frames):
        ax.clear()

        # Plot with anomaly background
        draw_anomaly_background(ax, ground_truth)
        ax.plot(predictions, label='Prediction', color='blue')
        if isinstance(clip_keys, tuple):
            ax.plot(ground_truth, label='Ground Truth', color='green')
            if (clip_keys[1]-45) > 0:    
                ax.axvline(x=(clip_keys[1]), label='Time of Accident Frame', color='red')
        else:
            ax.plot(ground_truth, label='Ground Truth', color='green')

        # xticks = np.arange(0, num_frames, step=20)
        
        # if dataset == "dota":
        #     xticklabels = xticks + 15
        # else:
        #     xticklabels = xticks + 46

        # ax.set_xticks(xticks)
        # ax.set_xticklabels(xticklabels)

        ax.axvline(x=i, color='black', linewidth=2, label='Current Frame')
        ax.set_xlim(0, num_frames)
        ax.set_ylim(0, 1)
        ax.set_title("Anomaly Detection - Prediction vs Ground Truth")
        ax.axhline(y=0.5, color='orange', linestyle="--", linewidth=2, label='Threshold')
        ax.set_xlabel('Frame number')
        ax.set_ylabel('Score')

        # Render plot to image
        canvas.draw()
        buf = canvas.buffer_rgba()
        graph_img = np.asarray(buf)[..., :3]
        graph_img = cv2.resize(graph_img, (width, graph_height))

        # Get and resize current video frame
        frame = resized_clip_frames[i]

        # Combine the video frame (on top) with the graph (below)
        combined = np.vstack((frame, graph_img))  # shape: (total_height, width, 3)

        # Convert RGB to BGR for OpenCV and write frame
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    out.release()
    plt.close(fig)
    print(f"Visualization saved to {output_dir}")

def read_annotations_dota(data_path, multi_class=False, mode="train"):
    clip_names = None
    clip_timesteps = []
    clip_binary_labels = []
    clip_ego = []
    clip_night = []

    if mode == "train":
        anno_path = 'train_split.txt'
    elif mode == "val":
        anno_path = 'val_split.txt'
    else:  
        anno_path = 'val_split.txt'

    with open(os.path.join(data_path, "dataset", anno_path), 'r') as file:
        # /media/ltran/Data/datasets_sveta/DoTA_refined/dataset/train_split.txt
        clip_names = [line.rstrip() for line in file]
    for clip in clip_names:
        # print(f"clip: {clip}")
        clip_anno_path = os.path.join(data_path, "dataset", "annotations", f"{clip}.json")
        # clip_anno_path: /media/ltran/Data/datasets_sveta/DoTA_refined/dataset/annotations/ha-IeID24As_001886.json
        with open(clip_anno_path) as f:
            anno = json.load(f)
            # sort is not required since we read already sorted timesteps from annotations
            timesteps = natsorted([int(os.path.splitext(os.path.basename(frame_label["image_path"]))[0]) for frame_label
                                in anno["labels"]])
            # print("timesteps: ", timesteps)
            cat_labels = [int(frame_label["accident_id"]) for frame_label in anno["labels"]]
            if_ego = anno["ego_involve"]
            if_night = anno["night"]

        # here the binary frame labels are created based on the accident category labels for each frame
        binary_labels = [1 if l > 0 else 0 for l in cat_labels]

        if multi_class:
            if if_ego:
                binary_labels = [1 if l > 0 else 0 for l in cat_labels]
            else:
                binary_labels = [2 if l > 0 else 0 for l in cat_labels]

        clip_timesteps.append(timesteps)
        clip_binary_labels.append(binary_labels)
        clip_ego.append(if_ego)
        clip_night.append(if_night)

    assert len(clip_names) == len(clip_timesteps) == len(clip_binary_labels)

    return clip_names, clip_timesteps, clip_binary_labels, clip_ego, clip_night

def read_annotations_dada(data_path, multi_class=False, mode="train"):
        ego_categories = [str(cat) for cat in list(range(1, 19)) + [61, 62]]
        video_ext=".png"
        clip_timesteps = []
        clip_binary_labels = []
        clip_cat_labels = []
        clip_ego = []
        clip_night = []
        clip_toa = []
        clip_acc = []

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
            description_csv = row["texts"]
            assert len(row) == 1, f"Multiple results! \n{info}"
            if len(row) != 1:
                errors.append(info)
            row = row.iloc[0]
            with zipfile.ZipFile(os.path.join(data_path, "frames", clip, "images.zip"), 'r') as zipf:
                framenames = natsorted([f for f in zipf.namelist() if os.path.splitext(f)[1]==video_ext])
            timesteps = natsorted([int(os.path.splitext(f)[0].split("_")[-1]) for f in framenames])
            
            if_acc_video = int(row["whether an accident occurred (1/0)"])
            st = int(row["abnormal start frame"])
            en = int(row["abnormal end frame"])
            toa = int(row["accident frame"])

            if multi_class:
                if if_ego:
                    binary_labels = [1 if st <= t <= en else 0 for t in timesteps]
                else:
                    binary_labels = [2 if st <= t <= en else 0 for t in timesteps]
            else:
                if st > -1 and en > -1:
                    binary_labels = [1 if st <= t <= en else 0 for t in timesteps]
                else:
                    binary_labels = [0 for t in timesteps]

            cat_labels = [l*int(clip_type) for l in binary_labels]
            if_ego = clip_type in ego_categories
            if_night = int(row["light(day,night)1-2"]) == 2
            toa = int(row["accident frame"])

            clip_timesteps.append(timesteps)
            clip_binary_labels.append(binary_labels)
            clip_cat_labels.append(cat_labels)
            clip_ego.append(if_ego)
            clip_night.append(if_night)
            clip_toa.append(toa)
            clip_acc.append(if_acc_video)

        for line in errors:
            print(line)
        if len(errors) > 0:
            print(f"\n====\nerrors: {len(errors)}. You can add saving the error list in the code.")
            exit(0)

        assert len(clip_names) == len(clip_timesteps) == len(clip_binary_labels)

        return clip_names, clip_timesteps, clip_binary_labels, clip_ego, clip_night, clip_toa 


def process_clip(clip_key, preds_grouped, labels_grouped, clip_names, clip_timesteps, dataset, folder, phase, fps, width, graph_height, output_path):
    # Find the index of the clip in clip_names
    if isinstance(clip_key, tuple):
        clip_id = clip_key[0]
    else:
        clip_id = clip_key
    clip_name = clip_names[clip_id]
    print(f"clip name: {clip_name}, clip_id: {clip_id}, clip_key: {clip_key}")

    start_idx = 15 if dataset == "dota" else 45
    timesteps = [i for i in range(1, len(clip_timesteps[clip_id])+1)] 

    if dataset == "dota":
        filenames = [f"{str(ts).zfill(6)}.jpg" for ts in timesteps]
    else:
        filenames = [f"{str(ts).zfill(4)}.png" for ts in timesteps]

    images = []
    with zipfile.ZipFile(os.path.join(folder, "frames", clip_name, "images.zip"), 'r') as zipf:
        for fname in filenames:
            with zipf.open(fname) as file:
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
                images.append(img)

    preds_cat = np.concatenate((np.array([np.nan for _ in range(start_idx-1)]), preds_grouped[clip_key]))
    # testing 5 1s bins of 30 frames -> label bin 5 starts at frame 30
    ground_truth_cat = np.concatenate((np.array([0 for _ in range(start_idx-1)]), labels_grouped[clip_key]))

    predictions = preds_cat
    ground_truth = ground_truth_cat


    visualize_predictions_with_clip_and_anomaly(
        predictions,
        ground_truth,
        images,
        clip_keys=clip_key,
        dataset=dataset,
        clip_names=clip_name,
        phase=phase,
        output_path=output_path,
        fps=fps,
        width=width,
        graph_height=graph_height
    )

def create_and_visualize_videos(preds, labels, clip_infos, dataset="dota", output_path="/home/ltran/vmae_predictions", fps=30, phase="val", width=800, graph_height=400):
        
    preds_grouped = defaultdict(list)
    labels_grouped = defaultdict(list)

    preds = torch.nn.functional.softmax(preds, dim=1)

    # Group predictions and labels by clip
    if clip_infos.ndim > 1:
        for pred, (clip_id, toa), label in zip(preds[:,1].cpu(), clip_infos.cpu(), labels.cpu()):
            preds_grouped[(int(clip_id), int(toa))].append(pred)
            labels_grouped[(int(clip_id), int(toa))].append(label)
    else:
        for pred, clip_id, label in zip(preds[:,1].cpu(), clip_infos.cpu(), labels.cpu()):
            preds_grouped[int(clip_id)].append(pred)
            labels_grouped[int(clip_id)].append(label)

    # Load annotation info
    if dataset == "dota":
        folder = "/media/datasets_sveta/DoTA_refined"
        clip_names, clip_timesteps, _, _, _ = read_annotations_dota(folder, multi_class=False, mode=phase)
    else:
        folder = "/media/datasets_sveta/DADA2000"
        clip_names, clip_timesteps, _, _, _, _ = read_annotations_dada(folder, multi_class=False, mode=phase)

    # Parallelize video creation
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for clip_key in preds_grouped.keys():
            futures.append(
                executor.submit(
                    process_clip,
                    clip_key, preds_grouped, labels_grouped,
                    clip_names, clip_timesteps,
                    dataset, folder, phase, fps, width, graph_height, output_path
                )
            )
        # Optionally, wait for all to finish
        for future in concurrent.futures.as_completed(futures):
            future.result()  # To catch exceptions
