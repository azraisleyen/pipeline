
from pathlib import Path
import cv2

from ..core.utils import ensure_dir, list_files_by_extensions


class FrameLoader:
    def __init__(self, video_path, frames_dir):
        self.video_path = Path(video_path)
        self.frames_dir = Path(frames_dir)
        ensure_dir(self.frames_dir)

    def frames_exist(self):
        frame_paths = list_files_by_extensions(
            self.frames_dir,
            [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
        )
        return len(frame_paths) > 0

    def extract_frames(self):
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video bulunamadı: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {self.video_path}")

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out_path = self.frames_dir / f"frame_{frame_id:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            frame_id += 1

        cap.release()
        print(f"Toplam çıkarılan frame sayısı: {frame_id}")

    def get_frame_list(self):
        if self.frames_exist():
            return list_files_by_extensions(
                self.frames_dir,
                [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
            )

        self.extract_frames()

        return list_files_by_extensions(
            self.frames_dir,
            [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
        )
