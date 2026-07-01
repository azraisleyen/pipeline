from pathlib import Path
import cv2
class ImageDirectorySource:
    def __init__(self,path): self.paths=sorted([p for p in Path(path).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}])
    def __iter__(self):
        for i,p in enumerate(self.paths,1): yield i,p.stem,cv2.imread(str(p))
class VideoFrameSource:
    def __init__(self,path): self.path=path
    def __iter__(self):
        cap=cv2.VideoCapture(str(self.path)); i=1
        while True:
            ok,frame=cap.read();
            if not ok: break
            yield i,f'frame_{i:06d}',frame; i+=1
        cap.release()
