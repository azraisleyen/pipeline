def clamp_bbox(bbox, width=None, height=None):
    x1,y1,x2,y2=[int(round(float(v))) for v in bbox]
    if width is not None: x1=max(0,min(x1,width)); x2=max(0,min(x2,width))
    if height is not None: y1=max(0,min(y1,height)); y2=max(0,min(y2,height))
    return (x1,y1,x2,y2)
def bbox_area(b): return max(0,int(b[2])-int(b[0]))*max(0,int(b[3])-int(b[1]))
def center(b): return ((b[0]+b[2])/2.0,(b[1]+b[3])/2.0)
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=bbox_area((ix1,iy1,ix2,iy2)); den=bbox_area(a)+bbox_area(b)-inter
    return inter/den if den>0 else 0.0
