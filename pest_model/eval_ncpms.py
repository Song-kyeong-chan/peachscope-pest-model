import json, collections, torch, timm
from pathlib import Path
from PIL import Image, ImageOps
from torchvision import transforms as T
from ultralytics import YOLO, YOLOWorld
O=Path.home()/"peachscope-data/ncpms"; m=json.load(open(O/'fruit_images.json'))
CL=["healthy","shot_hole","brown_rot"]
tf=T.Compose([T.Resize(288),T.CenterCrop(288),T.ToTensor(),T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
models={}
for name,p in [("2B_convnext","results/exp2b_clean/convnext_tiny.pt"),("exp2_convnext","results/exp2_crops/convnext_tiny.pt")]:
    net=timm.create_model("convnext_tiny",num_classes=3); net.load_state_dict(torch.load(p,map_location="cpu")); net.eval(); models[name]=net
fd=YOLO("results/exp2_yolo/yolo11n_fruit/weights/best.pt")
yw=YOLOWorld("yolov8l-worldv2.pt"); yw.set_classes(["peach","fruit"])
rows=[]; crops=[]
for e in m:
    im=ImageOps.exif_transpose(Image.open(O/e['file'])).convert('RGB'); W,H=im.size
    boxes=[]; src="none"
    for det,nm,c in [(fd,"yolo11n",0.25),(yw,"world",0.1)]:
        r=det.predict(im,conf=c,verbose=False)[0]
        b=[x for x in r.boxes.xyxy.tolist() if min(x[2]-x[0],x[3]-x[1])>=0.12*min(W,H)]
        if b: boxes=b; src=nm; break
    if boxes:
        # 가장 큰 과실 (NCPMS 사진은 대상 과실이 크게 찍힘)
        x0,y0,x1,y1=max(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1])); pw,ph=(x1-x0)*.1,(y1-y0)*.1
        c=im.crop((max(0,x0-pw),max(0,y0-ph),min(W,x1+pw),min(H,y1+ph)))
    else: c=im
    crops.append(c)
    row={'file':e['file'],'cls':e['cls'],'det':src,'nfruit':len(boxes)}
    with torch.no_grad():
        for nm,net in models.items():
            pc=torch.softmax(net(tf(c)[None]),1)[0]; pf=torch.softmax(net(tf(im)[None]),1)[0]
            row[nm+'_crop']=CL[int(pc.argmax())]; row[nm+'_crop_p']=[round(float(v),3) for v in pc]
            row[nm+'_full']=CL[int(pf.argmax())]
    rows.append(row)
json.dump(rows,open(O/'eval_rows.json','w'),ensure_ascii=False,indent=1)
print('detector used',collections.Counter(r['det'] for r in rows))
for col in ['2B_convnext_crop','2B_convnext_full','exp2_convnext_crop']:
    print('\n==',col)
    t=collections.defaultdict(collections.Counter)
    for r in rows: t[r['cls']][r[col]]+=1
    for k in ['brown_rot','shot_hole','scab','anthracnose','gray_mold','insect_damage','stink_bug','planthopper','viroid']:
        if k in t: n=sum(t[k].values()); print(f"  {k:14s} n={n:2d}", dict(t[k]), f"disease-detected={1-t[k]['healthy']/n:.2f}")
# save crop sheet
from PIL import ImageDraw, ImageFont
F=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",15); S,G=200,10
sh=Image.new('RGB',(S*G,S*((len(crops)+G-1)//G)),'white'); d=ImageDraw.Draw(sh)
for j,(c,r) in enumerate(zip(crops,rows)):
    c=c.copy(); c.thumbnail((S-4,S-4)); x,y=(j%G)*S+2,(j//G)*S+2; sh.paste(c,(x,y))
    d.rectangle([x,y,x+196,y+18],fill='black'); d.text((x+2,y),f"{j} {r['cls'][:6]}>{r['2B_convnext_crop'][:6]}",fill='yellow',font=F)
sh.save('str(O/'pred_sheet.jpg')',quality=85)
