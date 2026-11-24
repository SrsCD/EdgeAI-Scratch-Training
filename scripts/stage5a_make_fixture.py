"""Generate small geometric video fixtures; they are not real CCTV examples."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/experimental/stage5a_fixture'
OUT.mkdir(parents=True,exist_ok=True)
frames=[]
for i in range(12):
    f=np.full((240,320,3),35,dtype=np.uint8)
    if i:
        cv2.rectangle(f,(45+i*4,55),(95+i*4,175),(80,180,210),-1)
        cv2.circle(f,(70+i*4,42),20,(120,175,205),-1)
    if i>=6:
        cv2.rectangle(f,(190,115),(260,175),(120,80,50),-1)
    frames.append(f)
for suffix,fourcc in (('.mp4','mp4v'),('.avi','MJPG'),('.mov','mp4v')):
    path=OUT/('synthetic_cctv'+suffix)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*fourcc),10.0,(320,240))
    if not writer.isOpened():
        print('codec unavailable',suffix);continue
    for frame in frames:writer.write(frame)
    writer.release()
    cap=cv2.VideoCapture(str(path));n=0
    while True:
        ok,_=cap.read()
        if not ok:break
        n+=1
    print(path.name,'decoded_frames',n,'fps',cap.get(cv2.CAP_PROP_FPS))
    cap.release()
rgb=[Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)) for f in frames]
rgb[0].save(OUT/'synthetic_cctv.gif',save_all=True,append_images=rgb[1:],duration=100,loop=0)
print('synthetic_cctv.gif frames',len(rgb))
