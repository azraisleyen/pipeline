def ensure_bgr(frame):
    if frame is None: raise ValueError('Frame is None')
    if getattr(frame,'ndim',None)!=3 or frame.shape[2] not in (3,4): raise ValueError(f'Expected HxWx3/4 frame, got {getattr(frame,"shape",None)}')
    return frame[:,:,:3]
def crop(frame,bbox):
    x1,y1,x2,y2=bbox; return frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
