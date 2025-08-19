import argparse
import logging
import os

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MIN_INTERVAL_TQDM = 1.0
DEEP_SORT_MAX_AGE = 15
MIN_TRACK_AREA = 4096
N_INIT = 4
EMBEDDER_WTS = r"D:\work\track\best_model_osnet.pth.tar" # r"D:\work\track\deep-object-reid\model.pth.tar-29"
EMBEDDER = "torchreid_v2"
MAX_COSINE_DISTANCE = 0.2
MAX_IOU_DISTANCE = 0.5
IOU_THRESHOLD = 0.7


def compute_iou(boxA, boxB):
    """
    Вычисляет IOU между двумя прямоугольниками.
    boxA, boxB: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def merge_tracks(tracks, appearance_thresh=0.15, iou_thresh=0.5):
    merged = []
    used = set()
    for i, t1 in enumerate(tracks):
        if i in used:
            continue
        for j, t2 in enumerate(tracks):
            if i == j or j in used:
                continue
            # appearance
            f1 = t1.get_avg_feature()
            f2 = t2.get_avg_feature()
            if f1 is not None and f2 is not None:
                cos_dist = 1 - np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
                # iou
                ltrb1 = t1.to_ltrb()
                ltrb2 = t2.to_ltrb()
                iou = compute_iou(ltrb1, ltrb2)
                if cos_dist < appearance_thresh and iou > iou_thresh:
                    # объединить (например, оставить один трек)
                    used.add(j)
        merged.append(t1)
    return merged


def parse_video_folder(video_folder):
    videos = []
    for dirpath, _, files in os.walk(video_folder):
        for filename in files:
            if filename.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                videos.append(os.path.join(dirpath, filename))
    return videos


def show_tracks_from_videos(video_folder, device="cuda", debug=False):
    """
    Обработка каждого видео, вывод результата трекинга на экран.
    """
    model = YOLO("yolo12x.pt").to(device)
    videos = parse_video_folder(video_folder)
    logger.info(f"Found {len(videos)} videos...")

    pbar = tqdm(videos, mininterval=MIN_INTERVAL_TQDM)
    for video_path in pbar:
        video_name = os.path.basename(video_path)
        cap = cv2.VideoCapture(video_path)
        tracker = DeepSort(
            max_age=DEEP_SORT_MAX_AGE,
            n_init=N_INIT,
            embedder=EMBEDDER,
            embedder_wts=EMBEDDER_WTS,
            max_iou_distance=MAX_IOU_DISTANCE,
            max_cosine_distance=MAX_COSINE_DISTANCE,
            gating_only_position=False,
        )
        
        if debug:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            path = os.path.splitext(video_path)[0] + "_tracked.avi"
            out = cv2.VideoWriter(path, fourcc, 20.0, (1280, 720))    

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_height, frame_width = frame.shape[:2]
            results = model.predict(frame, verbose=False, classes=0, iou=IOU_THRESHOLD)[0]  # только люди

            bbs = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int)
                w, h = x2 - x1, y2 - y1
                conf = box.conf.item()
                cls = box.cls.item()
                bbs.append(([x1, y1, w, h], conf, cls))

            tracks = tracker.update_tracks(bbs, frame=frame)
            tracks = merge_tracks(tracks, MAX_COSINE_DISTANCE, IOU_THRESHOLD)

            # Отрисовка треков
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                if track.time_since_update > 2:
                    continue

                track_id = track.track_id
                # ltrb = track.to_ltrb(orig=True)
                ltrb = track.get_smoothed_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # ограничиваем рамку в пределах кадра
                x1 = max(0, min(x1, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                x2 = max(1, min(x2, frame_width))
                y2 = max(1, min(y2, frame_height))

                if (x2 -   x1) * (y2 - y1) < MIN_TRACK_AREA:
                    continue

                # рамка и ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if debug:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # показ кадра
            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow(f"Tracking - {video_name}", frame)
            if debug:
                out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        if debug:
            out.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track and show videos")
    parser.add_argument("--folder", type=str, required=True, help="Path to folder with videos")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save output videos")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cuda or cpu)")
    
    args = parser.parse_args()
    logger.info(f"Using device: {args.device}")
    show_tracks_from_videos(args.folder, args.device, args.debug)