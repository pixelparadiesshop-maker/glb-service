#!/usr/bin/env python3
"""
GLB-Service – verpackt die Cover-Pipeline als Web-API für n8n (Cloud).

Endpoints
  GET  /            -> Healthcheck ("ok")
  POST /glb         -> Scan rein, GLB raus (binär, model/gltf-binary)

POST /glb akzeptiert ENTWEDER
  a) multipart/form-data: file=<scan.jpg>, tag=PS4, flip=false
  b) application/json:     {"image_base64":"...", "tag":"PS4", "flip":false}

Antwort: die GLB-Datei als Download (Content-Type model/gltf-binary).
Bei Maß-Plausibilitätsfehler kommt Header  X-Case-Warning  mit zurück.

Lokal testen:
  pip install fastapi "uvicorn[standard]" python-multipart pillow numpy trimesh
  uvicorn glb_service:app --host 0.0.0.0 --port 8000
"""
import io, base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Body, Response, HTTPException

app = FastAPI(title="Cover-GLB-Service")

# ── Plattform-Tag (aus deinem n8n-Workflow) -> Hüllenmaße (Panel-B, Höhe, Spine) mm ──
TAG_DIMS = {
    # Blu-ray-Keepcase
    "PS3": (135,170,14), "PS4": (135,170,14), "PS5": (135,170,14),
    "XONE": (135,170,14), "XSX": (135,170,14),
    # DVD-Keepcase (größer)
    "PS2": (135,190,14), "X360": (135,190,14), "WII": (135,190,14),
    "WIIU": (135,190,14), "XBOX": (135,190,14),
    # Sonderformate
    "GC": (123,170,14), "SWITCH": (104,170,10), "SWITCH2": (104,170,10),
}

def trim_margins(im, thresh=18):
    a = np.asarray(im.convert("L")).astype(int)
    h, w = a.shape
    gr = np.abs(np.diff(a.mean(axis=1))); gc = np.abs(np.diff(a.mean(axis=0)))
    lh, lw = int(h*0.02), int(w*0.02)
    edge = lambda g, l: (int(np.where(g[:l] > thresh)[0][-1]) + 1) if len(np.where(g[:l] > thresh)[0]) else 0
    return im.crop((edge(gc, lw), edge(gr, lh),
                    w - edge(gc[::-1], lw), h - edge(gr[::-1], lh)))

def build_glb(im: Image.Image, tag: str, flip: bool):
    if tag not in TAG_DIMS:
        raise HTTPException(400, f"Unbekannter Plattform-Tag '{tag}'. Erlaubt: {', '.join(TAG_DIMS)}")
    pw, ph, sp = TAG_DIMS[tag]
    im = im.convert("RGB")
    if im.height > im.width:
        im = im.transpose(Image.ROTATE_90)
    if flip:
        im = im.transpose(Image.ROTATE_180)
    im = trim_margins(im)
    W, H = im.size

    spine_w = H * sp / ph
    cx = W / 2
    x0, x1 = round(cx - spine_w/2), round(cx + spine_w/2)
    back, spine, front = im.crop((0,0,x0,H)), im.crop((x0,0,x1,H)), im.crop((x1,0,W,H))

    ratio, target = front.width/front.height, pw/ph
    warning = ""
    if abs(ratio-target)/target > 0.08:
        warning = f"Panel-Ratio {ratio:.3f} vs Soll {target:.3f} – Scan evtl. unvollstaendig"

    AW, AH = 2048, 2048
    total = back.width + spine.width + front.width + 56
    scale = min(AW/total, AH/back.height, 1.0)
    rs = lambda i: i.resize((max(1,round(i.width*scale)), max(1,round(i.height*scale))), Image.LANCZOS)
    back, spine, front = rs(back), rs(spine), rs(front)
    Hpx = back.height

    atlas = Image.new("RGB", (AW, AH), (27,63,143))
    atlas.paste(back,(0,0)); atlas.paste(spine,(back.width,0))
    atlas.paste(front,(back.width+spine.width,0))
    vh = Hpx/AH
    ub1 = back.width/AW; us1 = (back.width+spine.width)/AW
    uf1 = (back.width+spine.width+front.width)/AW
    up0, up1 = uf1+8/AW, min(1.0, uf1+40/AW)
    X, Y, Z = pw/2000, ph/2000, sp/2000

    verts, uvs, faces = [], [], []
    def quad(TL,TR,BR,BL,u0,u1,v0=0.,v1=1.):
        i = len(verts); verts.extend([TL,TR,BR,BL])
        a, b = v0*vh, v1*vh
        uvs.extend([(u0,1-a),(u1,1-a),(u1,1-b),(u0,1-b)])
        faces.extend([(i,i+3,i+2),(i,i+2,i+1)])
    quad((-X,+Y,+Z),(+X,+Y,+Z),(+X,-Y,+Z),(-X,-Y,+Z), us1, uf1)
    quad((+X,+Y,-Z),(-X,+Y,-Z),(-X,-Y,-Z),(+X,-Y,-Z), 0.0, ub1)
    quad((-X,+Y,-Z),(-X,+Y,+Z),(-X,-Y,+Z),(-X,-Y,-Z), ub1, us1)
    quad((+X,+Y,+Z),(+X,+Y,-Z),(+X,-Y,-Z),(+X,-Y,+Z), up0, up1,.3,.7)
    quad((-X,+Y,-Z),(+X,+Y,-Z),(+X,+Y,+Z),(-X,+Y,+Z), up0, up1,.3,.7)
    quad((-X,-Y,+Z),(+X,-Y,+Z),(+X,-Y,-Z),(-X,-Y,-Z), up0, up1,.3,.7)

    import trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial
    atlas.format = "JPEG"
    _save = atlas.save
    atlas.save = lambda fp, **kw: _save(fp, format="JPEG", quality=92)
    mesh = trimesh.Trimesh(np.array(verts), np.array(faces), process=False)
    mesh.visual = TextureVisuals(uv=np.array(uvs),
        material=PBRMaterial(baseColorTexture=atlas, metallicFactor=0.0,
                             roughnessFactor=0.55, name=tag))
    glb_bytes = trimesh.Scene([mesh]).export(file_type="glb")
    return glb_bytes, warning

@app.get("/")
def health():
    return {"status": "ok", "tags": list(TAG_DIMS)}

@app.post("/glb")
async def glb(
    file: UploadFile = File(None),
    tag: str = Form(None),
    flip: str = Form("false"),
    payload: dict = Body(None),
):
    # Eingabe a) multipart  oder  b) JSON
    if file is not None:
        data = await file.read()
        t = (tag or "PS4").upper()
        fl = str(flip).lower() in ("1","true","yes","ja")
    elif payload:
        b64 = payload.get("image_base64", "")
        if "," in b64: b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        t = (payload.get("tag") or "PS4").upper()
        fl = str(payload.get("flip", False)).lower() in ("1","true","yes","ja")
    else:
        raise HTTPException(400, "Kein Bild übergeben (file= oder image_base64=).")

    try:
        im = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Datei ist kein gültiges Bild.")

    glb_bytes, warning = build_glb(im, t, fl)
    headers = {"Content-Disposition": 'attachment; filename="case.glb"'}
    if warning:
        headers["X-Case-Warning"] = warning
    return Response(content=glb_bytes, media_type="model/gltf-binary", headers=headers)
