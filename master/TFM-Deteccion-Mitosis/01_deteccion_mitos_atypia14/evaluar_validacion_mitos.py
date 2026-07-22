#!/usr/bin/env python3
import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import sys, json, math
from collections import defaultdict
from pathlib import Path

VERS = Path(f'{WORKSPACE}/Versiones_Paciente/mitos14')
MPP = {'aperio': 0.2455, 'hamamatsu': 0.2274}
DIST_UM = 7.5
DIST_PX = {sc: DIST_UM / m for sc, m in MPP.items()}

def predecir(modelo, ckpt, image_paths, thr=0.05):
    out = defaultdict(list)
    if modelo == 'rfdetr':
        import rfdetr
        model = rfdetr.RFDETRSmall(pretrain_weights=str(ckpt))
        for i in range(0, len(image_paths), 8):
            res = model.predict([str(p) for p in image_paths[i:i+8]], threshold=thr)
            res = res if isinstance(res, list) else [res]
            for p, det in zip(image_paths[i:i+8], res):
                if det is None: continue
                for box, conf in zip(det.xyxy, det.confidence):
                    out[p.name].append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(conf)))
    else:
        from ultralytics import YOLO
        model = YOLO(str(ckpt))
        for i in range(0, len(image_paths), 64):
            batch = [str(p) for p in image_paths[i:i+64]]
            for p, r in zip(image_paths[i:i+64], model.predict(batch, conf=thr, imgsz=512, verbose=False)):
                b = r.boxes
                if b is None: continue
                for box, conf in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
                    out[p.name].append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(conf)))
    return out

def cen(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1]); ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    iw=max(0.0,ix2-ix1); ih=max(0.0,iy2-iy1); inter=iw*ih
    if inter<=0: return 0.0
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def sweep(items, crit):
    filas=[]
    for thr in [round(0.05*i, 2) for i in range(1, 19)]:
        TP=FP=FN=0
        for d, gts, ps in items:
            used=[False]*len(gts)
            for pb in sorted([p for p in ps if p[4] >= thr], key=lambda z: -z[4]):
                best=-1
                if crit=='distancia':
                    pcx,pcy=cen(pb); bd=d
                    for gi,g in enumerate(gts):
                        if used[gi]: continue
                        gcx,gcy=cen(g); dd=math.hypot(pcx-gcx,pcy-gcy)
                        if dd<=bd: bd=dd; best=gi
                else:
                    bi=0.5
                    for gi,g in enumerate(gts):
                        if used[gi]: continue
                        i=iou(pb[:4],g)
                        if i>=bi: bi=i; best=gi
                if best>=0: used[best]=True; TP+=1
                else: FP+=1
            FN+=len(gts)-sum(used)
        Pr=TP/(TP+FP) if TP+FP else 0.0; Re=TP/(TP+FN) if TP+FN else 0.0
        F1=2*Pr*Re/(Pr+Re) if Pr+Re else 0.0
        filas.append({'conf':thr,'TP':TP,'FP':FP,'FN':FN,
                      'precision':round(Pr,4),'recall':round(Re,4),'f1':round(F1,4)})
    return filas

def main():
    modelo, ckpt, fold, salida, nombre = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]), sys.argv[5]
    salida.mkdir(parents=True, exist_ok=True)
    items = []
    n_gt = 0
    for sc, pre in [('aperio', 'A'), ('hamamatsu', 'H')]:
        pac = pre + fold
        cdir = VERS / sc / pac / 'coverage'
        if not (cdir / '_annotations.coco.json').exists():
            print(f'  [WARN] sin coverage {sc}/{pac}', flush=True); continue
        coco = json.loads((cdir / '_annotations.coco.json').read_text())
        gt_by_img = defaultdict(list)
        for a in coco['annotations']:
            if a['category_id'] == 1:
                x, y, w, h = a['bbox']; gt_by_img[a['image_id']].append((x, y, x+w, y+h))
        imgs = [cdir / 'images' / im['file_name'] for im in coco['images']]
        preds = predecir(modelo, ckpt, imgs)
        d = DIST_PX[sc]
        for im in coco['images']:
            gts = gt_by_img[im['id']]; n_gt += len(gts)
            ps = preds.get(im['file_name'], [])
            items.append((d, gts, ps))
        print(f'  {sc}/{pac}: {len(coco["images"])} parches, {sum(len(gt_by_img[i]) for i in gt_by_img)} mit', flush=True)

    res = {'nombre': nombre, 'modelo': modelo, 'fold': fold, 'n_gt': n_gt}
    for crit in ['distancia', 'iou']:
        filas = sweep(items, crit)
        at = lambda c: min(filas, key=lambda r: abs(r['conf']-c))
        fbest = max(filas, key=lambda r: r['f1'])
        res[crit] = {'sweep': filas, 'conf_0.30': at(0.30), 'conf_0.50': at(0.50), 'best_f1': fbest}
        print(f'  [{nombre}|{crit}] @0.30 R{at(0.30)["recall"]} @0.50 R{at(0.50)["recall"]} | best F1 {fbest["f1"]}@{fbest["conf"]}', flush=True)
    (salida / f'{nombre}_resumen.json').write_text(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
