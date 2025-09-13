import os, warnings, time, glob, tempfile, threading
warnings.filterwarnings("ignore")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import gradio as gr

# ---------- optional deps ----------
try:
    import nibabel as nib
    HAVE_NIB = True
except Exception:
    HAVE_NIB = False

try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ---------- model predictor ----------
PREDICTOR = None
DEVICE = "cuda:0"
CKPT = (
    "/mnt/hdd3/hjang/scripts/MICCAI_2025_CLIP/checkpoints/"
    "nnUNetInteractionTrainer__nnUNetPlans__3d_fullres_scratch"
)
for env in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    os.environ.setdefault(env, tempfile.mkdtemp(prefix=f"{env}_"))

def _init_predictor_once():
    """Load model once + (best-effort) CUDA warm-up."""
    global PREDICTOR
    if PREDICTOR is not None:
        return True
    try:
        import torch
        from Interactive_MEN_RT_predictor import InteractiveMENRTPredictor
        dev = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        pred = InteractiveMENRTPredictor(
            device=dev, use_torch_compile=False, do_autozoom=False, verbose=False
        )
        pred.initialize_from_trained_model_folder(
            model_training_output_dir=CKPT, use_fold=0, checkpoint_name="checkpoint_best.pth"
        )
        PREDICTOR = pred
        # GPU warm-up (best effort)
        try:
            if torch.cuda.is_available():
                x = np.zeros((1, 8, 8, 8), np.float32)
                pred.reset_interactions()
                pred.set_image(x)
                pred.set_target_buffer(np.zeros_like(x[0], np.float32))
                pred._finish_preprocessing_and_initialize_interactions()
                torch.cuda.synchronize()
        except Exception:
            pass
        print("[MODEL] ready")
        return True
    except Exception as e:
        print(f"[MODEL] init failed: {e}")
        return False

def preload_model_in_background():
    threading.Thread(target=_init_predictor_once, daemon=True).start()

# ---------- config ----------
DATA_ROOT = Path("/mnt/hdd3/hjang/data/Meningioma/Interactive_MEN_RT/data/val")
EXAMPLES = [
    "BraTS-MEN-RT-0071-1",
    "BraTS-MEN-RT-0223-1",
    "BraTS-MEN-RT-0280-1",
    "BraTS-MEN-RT-0422-1",
    "BraTS-MEN-RT-0436-1",
]
RENDER_PX_DEFAULT = 384
MAX_RENDER_PX = 1024
ROT_CCW = True  # 90 degree CCW

# colors
ACCENT_HEX   = "#1e90ff"
CROSS_RGB    = (30, 144, 255)
GT_RGBA_FILL = (255, 215,   0, 128)  # alpha=0.5
PR_RGBA_FILL = (255,  60,  60, 128)  # alpha=0.5
SEED_RGB     = (89, 224, 154)

# ---------- state ----------
class State:
    def __init__(self):
        self.vol=None; self.shape=None
        self.gt=None;  self.pred=None
        self.case_id=None; self.loaded=False
        self.cross={"x":0,"y":0,"z":0}
        self.slice={"axial":0,"sagittal":0,"coronal":0}
        self.seeds=[]           # [(x,y,z), ...]
        self.seed_views=[]      # ["axial"/"sagittal"/"coronal", ...] (seeds and index synchronized)
        self.render_px=RENDER_PX_DEFAULT
        self.disp_wh={"axial":(RENDER_PX_DEFAULT,RENDER_PX_DEFAULT),
                      "sagittal":(RENDER_PX_DEFAULT,RENDER_PX_DEFAULT),
                      "coronal":(RENDER_PX_DEFAULT,RENDER_PX_DEFAULT)}
        self.active_view="axial"   # recent clicked view
S = State()

# ---------- utils ----------
def _norm01(a):
    a=a.astype(np.float32)
    p2,p98=np.percentile(a,2),np.percentile(a,98)
    if p98<=p2: p2,p98=float(a.min()),float(a.max()) or 1.0
    return np.clip((a-p2)/max(p98-p2,1e-6),0,1)

def _resize_slice_nearest(arr2d,w,h):
    im=Image.fromarray(arr2d); im=im.resize((w,h),Image.NEAREST); return np.array(im)

def _rot90_if_needed(img_or_np):
    if not ROT_CCW: return img_or_np
    if isinstance(img_or_np, Image.Image): return img_or_np.rotate(90, expand=True)
    return np.rot90(img_or_np, k=1)

# ---------- IO ----------
def _load_png_stack(case_dir):
    pngs = sorted(glob.glob(str(case_dir / "png_axial" / "*.png")))
    if not pngs:
        pngs = sorted(glob.glob(str(case_dir / "png_axial" / "*.jpg")))
    if not pngs: return None
    t0=time.time()
    arr=[np.array(Image.open(p).convert("L")) for p in pngs]
    vol=np.stack(arr,axis=2).astype(np.float32)
    vol=_norm01(vol)
    print(f"[PIL] {len(pngs)} slices -> {vol.shape} in {time.time()-t0:.2f}s")
    return vol

def _load_nifti(case_dir,case_id,ds=1):
    if not HAVE_NIB: return None
    p=case_dir/f"{case_id}_t1c.nii.gz"
    if not p.exists(): return None
    t0=time.time()
    nii=nib.load(str(p))
    arr=np.asanyarray(nii.dataobj[::ds,::ds,::ds],dtype=np.float32)
    arr=_norm01(arr)
    print(f"[NIfTI] {case_id} -> {arr.shape} in {time.time()-t0:.2f}s")
    return arr

def _resample_mask_to_vol_shape(mask_xyz, vol_shape_xyz):
    mx,my,mz=mask_xyz.shape; vx,vy,vz=vol_shape_xyz
    out=np.zeros((vx,vy,vz),dtype=np.uint8)
    for k in range(vz):
        src_k=int(round(k*(mz-1)/max(vz-1,1)))
        sl=(mask_xyz[:,:,src_k]>0).astype(np.uint8)*255
        im=Image.fromarray(sl).resize((vy,vx),Image.NEAREST)
        out[:,:,k]=(np.array(im)>0).astype(np.uint8)
    return out

def _load_gt(case_dir,case_id,vol_shape):
    candidates = [
        f"{case_id}_gtv.nii.gz", f"{case_id}_seg.nii.gz", f"{case_id}_gt.nii.gz",
        "gtv.nii.gz", "seg.nii.gz", "gt.nii.gz",
        f"{case_id}_gtv.nii", f"{case_id}_seg.nii", f"{case_id}_gt.nii",
    ]
    if HAVE_NIB:
        for name in candidates:
            p = case_dir/name
            if p.exists():
                try:
                    m=np.asanyarray(nib.load(str(p)).dataobj,dtype=np.uint8)
                    print(f"[GT] found {p.name} raw={m.shape}")
                    m=_resample_mask_to_vol_shape(m,vol_shape)
                    print(f"[GT] resized -> {m.shape}")
                    return (m>0).astype(np.uint8)
                except Exception as e:
                    print(f"[GT] load error {p.name}: {e}")
    print("[GT] not found.")
    return None

def load_case(case_id):
    case_dir=DATA_ROOT/case_id
    vol=_load_png_stack(case_dir) or _load_nifti(case_dir,case_id,ds=1)
    if vol is None:
        Z=96
        x=np.linspace(-1,1,RENDER_PX_DEFAULT)[:,None,]
        y=np.linspace(-1,1,RENDER_PX_DEFAULT)[None,:,]
        z=np.linspace(-1,1,Z)[None,None,:]
        vol=np.exp(-(x**2+y**2+z**2)*6).astype(np.float32)
        print("[VOL] dummy")
    S.vol=vol; S.shape=vol.shape; S.pred=None; S.seeds=[]; S.seed_views=[]; S.case_id=case_id
    X,Y,Z=S.shape
    S.cross={"x":X//2,"y":Y//2,"z":Z//2}
    S.slice={"sagittal":S.cross["x"],"coronal":S.cross["y"],"axial":S.cross["z"]}
    S.gt=_load_gt(case_dir,case_id,S.shape)
    S.render_px=RENDER_PX_DEFAULT
    S.active_view="axial"
    S.loaded=True
    print(f"[LOAD] {case_id} | shape={S.shape}")

# ---------- 2D (90° CCW) ----------
def _slice2d(view):
    if view=="axial":       sl=S.vol[:,:,S.slice["axial"]]
    elif view=="sagittal":  sl=S.vol[S.slice["sagittal"],:,:].T
    else:                   sl=S.vol[:,S.slice["coronal"],:].T
    return _rot90_if_needed(sl)

def _cross_pix_on_rot(view,w,h,x=None,y=None,z=None):
    X,Y,Z=S.shape
    if x is None: x=S.cross["x"]
    if y is None: y=S.cross["y"]
    if z is None: z=S.cross["z"]
    if view=="axial":
        u=int(round(x*(w-1)/max(X-1,1)))
        v=int(round((Y-1-y)*(h-1)/max(Y-1,1)))
    elif view=="sagittal":
        u=int(round(z*(w-1)/max(Z-1,1)))
        v=int(round((Y-1-y)*(h-1)/max(Y-1,1)))
    else:
        u=int(round(z*(w-1)/max(Z-1,1)))
        v=int(round((X-1-x)*(h-1)/max(X-1,1)))
    return u,v

def _draw_cross(img_draw, view, w, h):
    u,v=_cross_pix_on_rot(view,w,h)
    img_draw.line([(u,0),(u,h)],fill=CROSS_RGB,width=1)
    img_draw.line([(0,v),(w,v)],fill=CROSS_RGB,width=1)
    img_draw.ellipse((u-5,v-5,u+5,v+5),fill=(255,255,255),outline=ACCENT_HEX,width=2)

def render_top(view):
    if not S.loaded: return None
    sl=_slice2d(view)
    im=Image.fromarray((sl*255).astype(np.uint8)).resize((S.render_px,S.render_px),Image.BILINEAR).convert("RGB")
    w=h=S.render_px
    dr=ImageDraw.Draw(im)
    _draw_cross(dr, view, w, h)
    S.disp_wh[view]=im.size
    return im

# ---- Axial overlays (mask also 90° CCW) ----
def _axial_mask2d_rot(mask3d):
    if mask3d is None: return None
    m = mask3d[:,:,S.slice["axial"]].astype(np.uint8)
    m = _rot90_if_needed(m)
    return m

def _axial_overlay_fill(mask3d, rgba):
    sl = _rot90_if_needed(S.vol[:,:,S.slice["axial"]])
    base=Image.fromarray((sl*255).astype(np.uint8)).resize((S.render_px,S.render_px),Image.BILINEAR).convert("RGBA")
    m2d = _axial_mask2d_rot(mask3d)
    if m2d is None: return base.convert("RGB")
    m2d = _resize_slice_nearest((m2d>0).astype(np.uint8), S.render_px, S.render_px)
    over=np.zeros((S.render_px,S.render_px,4),dtype=np.uint8); over[m2d>0]=rgba
    return Image.alpha_composite(base,Image.fromarray(over,"RGBA")).convert("RGB")

# ---------- Interaction 2D ----------
def _interaction_2d():
    view = S.active_view
    if not S.loaded: return None
    sl=_slice2d(view)
    im=Image.fromarray((sl*255).astype(np.uint8)).resize((S.render_px,S.render_px),Image.BILINEAR).convert("RGB")
    w=h=S.render_px
    dr=ImageDraw.Draw(im)

    # cross
    _draw_cross(dr, view, w, h)

    # seeds (only show on the plane)
    tol=0
    for i, (x,y,z) in enumerate(S.seeds):
        on_plane = (
            (view=="axial"    and abs(z - S.slice["axial"])   <= tol) or
            (view=="sagittal" and abs(x - S.slice["sagittal"])<= tol) or
            (view=="coronal"  and abs(y - S.slice["coronal"]) <= tol)
        )
        if not on_plane: continue
        u,v=_cross_pix_on_rot(view,w,h,x,y,z)
        r=4
        dr.ellipse((u-r,v-r,u+r,v+r), fill=SEED_RGB, outline=(40,140,100), width=1)
        # number label(small)
        dr.text((u+6, v-8), f"{i+1}", fill=(30,30,30))
    return im

# ===================== segmentation ========================================
def _segment_with_model():
    if PREDICTOR is None and not _init_predictor_once():
        return None, "model-init-failed"
    try:
        img = S.vol[None].astype(np.float32)
        PREDICTOR.reset_interactions()
        PREDICTOR.set_image(img)
        PREDICTOR.set_target_buffer(np.zeros_like(img[0], np.float32))
        PREDICTOR._finish_preprocessing_and_initialize_interactions()
        PREDICTOR._predict_without_interaction()
        pred = (PREDICTOR.target_buffer.astype(np.float32) > 0.5).astype(np.uint8)
        if pred.shape != S.shape:
            print(f"[MODEL] resize pred {pred.shape}->{S.shape}")
            pred = _resample_mask_to_vol_shape(pred, S.shape)
        return pred, "ok"
    except Exception as e:
        print(f"[MODEL] inference failed: {e}")
        return None, "model-error"

def _segment_fallback():
    if not S.seeds or not HAVE_SCIPY:
        return None, "no-seeds-or-scipy"
    X,Y,Z=S.shape
    field=np.zeros((X,Y,Z),dtype=np.float32)
    for (x,y,z) in S.seeds: field[x,y,z]=1.0
    t0=time.time()
    prob=ndi.gaussian_filter(field,sigma=6.0)
    if prob.max()>0: prob/=prob.max()
    nz=prob[prob>0]; thr=np.percentile(nz,70) if nz.size else 0.5
    mask=prob>=max(thr,1e-3)
    lab,nlab=ndi.label(mask.astype(np.uint8))
    if nlab>1:
        keep=np.zeros(nlab+1,np.uint8)
        for (x,y,z) in S.seeds: keep[lab[x,y,z]]=1
        mask=keep[lab]>0
    mask=ndi.binary_closing(mask,iterations=1); mask=ndi.binary_opening(mask,iterations=1)
    print(f"[FB] seg {time.time()-t0:.3f}s | vox={int(mask.sum())}")
    return (mask>0).astype(np.uint8), "ok"

def do_segment():
    pred, tag = _segment_with_model()
    if pred is None:
        pred, tag2 = _segment_fallback(); tag = f"{tag}->{tag2}"
    S.pred = pred if pred is not None else None
    print(f"[SEG] done: {tag}")
    return "OK" if S.pred is not None else "Failed"

# ---------- debug widgets helpers ----------
def _seed_rows():
    """DataFrame rows: [[#, view, x, y, z], ...]"""
    rows=[]
    for i,(x,y,z) in enumerate(S.seeds):
        v = S.seed_views[i] if i < len(S.seed_views) else ""
        rows.append([i+1, v, x, y, z])
    return rows

def _seed_dropdown_options():
    """Dropdown options & default value"""
    opts=[]
    for i,(x,y,z) in enumerate(S.seeds):
        v = S.seed_views[i] if i < len(S.seed_views) else "axial"
        opts.append(f"{v} → point {i+1} → ({x},{y},{z})")
    return opts

def _debug_widgets(current_idx=None):
    rows = _seed_rows()
    df_upd = gr.update(value=rows)
    opts = _seed_dropdown_options()
    if current_idx is None:
        val = (opts[-1] if opts else None)
    else:
        val = (opts[current_idx] if (0 <= current_idx < len(opts)) else (opts[-1] if opts else None))
    dd_upd = gr.update(choices=opts, value=val)
    return df_upd, dd_upd

# ---------- pack outputs ----------
def _figs_and_imgs():
    top_ax=render_top("axial")
    top_sg=render_top("sagittal")
    top_co=render_top("coronal")
    ax_gt = _axial_overlay_fill(S.gt, GT_RGBA_FILL)
    ax_pr = _axial_overlay_fill(S.pred, PR_RGBA_FILL)
    inter2d = _interaction_2d()
    return top_ax, top_sg, top_co, ax_gt, ax_pr, inter2d

def _bar_ranges_and_values():
    X,Y,Z=S.shape
    return (gr.update(minimum=0,maximum=Z-1,value=S.slice["axial"],visible=True),
            gr.update(minimum=0,maximum=X-1,value=S.slice["sagittal"],visible=True),
            gr.update(minimum=0,maximum=Y-1,value=S.slice["coronal"],visible=True))

# ---------- robust event parsing ----------
def _parse_evt_xy(evt):
    """
    Gradio 3/4 호환: evt.index, evt.x/y, dict(evt['index'], evt['x']/['y']) 모두 대응
    """
    if evt is None: return None
    try:
        if hasattr(evt, "index") and evt.index is not None:
            ix = evt.index
            if isinstance(ix, (list, tuple)) and len(ix) >= 2:
                return int(ix[0]), int(ix[1])
        if hasattr(evt, "x") and hasattr(evt, "y"):
            return int(getattr(evt, "x")), int(getattr(evt, "y"))
        if isinstance(evt, dict):
            if "index" in evt and evt["index"] is not None:
                ix = evt["index"]
                if isinstance(ix, (list, tuple)) and len(ix) >= 2:
                    return int(ix[0]), int(ix[1])
            if "x" in evt and "y" in evt:
                return int(evt["x"]), int(evt["y"])
            d = evt.get("evt", {}).get("data", {})
            if "x" in d and "y" in d:
                return int(d["x"]), int(d["y"])
    except Exception:
        pass
    return None

def _disp_to_vol(view,u,v):
    X,Y,Z=S.shape; w,h=S.disp_wh[view]
    if w<=0 or h<=0: w=h=S.render_px
    if view=="axial":
        x = int(round(u * (X-1) / max(w-1,1)))
        y = int(round((Y-1) - v * (Y-1) / max(h-1,1)))
        z = S.slice["axial"]
    elif view=="sagittal":
        z = int(round(u * (Z-1) / max(w-1,1)))
        y = int(round((Y-1) - v * (Y-1) / max(h-1,1)))
        x = S.slice["sagittal"]
    else:
        z = int(round(u * (Z-1) / max(w-1,1)))
        x = int(round((X-1) - v * (X-1) / max(h-1,1)))
        y = S.slice["coronal"]
    x=max(0,min(X-1,x)); y=max(0,min(Y-1,y)); z=max(0,min(Z-1,z))
    return x,y,z

# ---------- thumbnails ----------
def _thumb_from_case(case_id,px=96):
    case_dir=DATA_ROOT/case_id
    pngs = sorted(glob.glob(str(case_dir / "png_axial" / "*.png")))
    if not pngs:
        pngs = sorted(glob.glob(str(case_dir / "png_axial" / "*.jpg")))
    if pngs:
        im = Image.open(pngs[len(pngs)//2]).convert("L")
        arr = _norm01(np.array(im).astype(np.float32))
        im = Image.fromarray((arr*255).astype(np.uint8)).resize((px,px),Image.BILINEAR)
    else:
        vol = _load_png_stack(case_dir) or _load_nifti(case_dir,case_id,ds=2)
        if vol is None:
            im = Image.new("L",(px,px),30)
        else:
            mid = vol[:,:,vol.shape[2]//2]
            im = Image.fromarray((mid*255).astype(np.uint8)).resize((px,px),Image.BILINEAR)
    im = _rot90_if_needed(im)
    return ImageOps.expand(im,border=1,fill=200)

# ---------- callbacks ----------
def on_load(case_id):
    load_case(case_id)
    preload_model_in_background()
    imgs=_figs_and_imgs(); bars=_bar_ranges_and_values(); dbg=_debug_widgets()
    return (*imgs,*bars,*dbg)

def on_gallery_select(evt: gr.events.SelectData):
    idx=0
    if hasattr(evt,"index"):
        ix=evt.index; idx=int(ix[0] if isinstance(ix,(list,tuple)) else ix)
    idx=max(0,min(len(EXAMPLES)-1,idx))
    cid=EXAMPLES[idx]
    out=on_load(cid)
    return (*out, gr.update(value=cid))

def _click_common(view, evt: gr.SelectData):
    if not S.loaded: return (*_figs_and_imgs(),*_bar_ranges_and_values(),*_debug_widgets())
    xy=_parse_evt_xy(evt)
    print(f"[UI] click@{view}: {getattr(evt,'index',None)} -> {xy}")
    if xy is None:  # prevent None on some versions: no-op update
        return (*_figs_and_imgs(),*_bar_ranges_and_values(),*_debug_widgets())
    u,v=xy
    x,y,z=_disp_to_vol(view,u,v)
    S.seeds.append((x,y,z)); S.seed_views.append(view)
    print(f"[UI] seed+ {(x,y,z)} total={len(S.seeds)}")
    S.cross={"x":x,"y":y,"z":z}
    S.slice={"sagittal":x,"coronal":y,"axial":z}
    S.active_view=view
    imgs=_figs_and_imgs(); bars=_bar_ranges_and_values(); dbg=_debug_widgets(current_idx=len(S.seeds)-1)
    return (*imgs,*bars,*dbg)

def on_axial_select(evt: gr.SelectData):    return _click_common("axial", evt)
def on_sagittal_select(evt: gr.SelectData): return _click_common("sagittal", evt)
def on_coronal_select(evt: gr.SelectData):  return _click_common("coronal", evt)

def on_seg_button():
    msg=do_segment(); print(f"[UI] Segment -> {msg}")
    return (*_figs_and_imgs(),*_bar_ranges_and_values(),*_debug_widgets())

def on_clear():
    S.seeds=[]; S.seed_views=[]; S.pred=None; print("[UI] cleared seeds/pred")
    return (*_figs_and_imgs(),*_bar_ranges_and_values(),*_debug_widgets())

def on_undo():
    if S.seeds:
        popped=S.seeds.pop(); pv=S.seed_views.pop() if S.seed_views else "axial"
        print(f"[UI] undo seed {popped} ({pv})")
    else:
        print("[UI] undo: no seeds")
    return (*_figs_and_imgs(),*_bar_ranges_and_values(),*_debug_widgets())

def on_z_release(z_idx):
    S.slice["axial"]=int(z_idx); S.cross["z"]=S.slice["axial"]; S.active_view="axial"
    print(f"[UI] Z -> {S.slice['axial']}")
    return (*_figs_and_imgs(),*_bar_ranges_and_values(),*_debug_widgets())

def on_x_release(x_idx):
    S.slice["sagittal"]=int(x_idx); S.cross["x"]=S.slice["sagittal"]; S.active_view="sagittal"
    print(f"[UI] X -> {S.slice['sagittal']}")
    return (*_figs_and_imgs(),*_bar_ranges_and_values(),*_debug_widgets())

def on_y_release(y_idx):
    S.slice["coronal"]=int(y_idx); S.cross["y"]=S.slice["coronal"]; S.active_view="coronal"
    print(f"[UI] Y -> {S.slice['coronal']}")
    return (*_figs_and_imgs(),*_bar_ranges_and_values(),*_debug_widgets())

def _jump_to_idx(idx:int):
    """Common: jump to selected index from log."""
    if not (0 <= idx < len(S.seeds)): 
        return (*_figs_and_imgs(),*_bar_ranges_and_values(),*_debug_widgets())
    x,y,z = S.seeds[idx]
    view  = S.seed_views[idx] if idx < len(S.seed_views) else "axial"
    S.cross={"x":x,"y":y,"z":z}
    S.slice={"sagittal":x,"coronal":y,"axial":z}
    S.active_view=view
    return (*_figs_and_imgs(),*_bar_ranges_and_values(),*_debug_widgets(current_idx=idx))

def on_seed_df_select(evt):
    """DataFrame select: evt.index -> (row, col)"""
    try:
        row = int(evt.index[0]) if hasattr(evt, "index") else 0
    except Exception:
        row = 0
    return _jump_to_idx(row)

def on_seed_dd_change(val):
    """Dropdown change: id parsing"""
    if not val:
        return (*_figs_and_imgs(),*_bar_ranges_and_values(),*_debug_widgets())
    try:
        p = val.split("point")[1].strip()
        num = int(p.split("→")[0].strip())
        idx = num - 1
    except Exception:
        idx = len(S.seeds)-1
    return _jump_to_idx(idx)

# ---------- UI ----------
css = """
:root {
  --bg:#f7fbff; --card:#ffffff; --accent:#1e90ff; --shadow:0 8px 26px rgba(45,156,219,.12);
}
.gradio-container{font-family:Inter,ui-sans-serif,system-ui;background:var(--bg)}
.round{background:var(--card);border-radius:16px;padding:10px;box-shadow:var(--shadow);border:1px solid #e8f0fb}
.section{font-weight:700;color:#114b8b;margin:2px 0 8px}
.tiny .gr-slider input[type="range"]{height:6px}
.tiny .gr-form{gap:6px}
.smallnote{color:#3e6285;font-size:12px;margin-top:6px}
.dfshort { max-height: 190px; overflow: auto; }
.dfshort table { font-size: 12px; }
"""

# thumbnails are created before UI (definition/dependency order guaranteed)
THUMBS=[_thumb_from_case(cid,px=96) for cid in EXAMPLES]

with gr.Blocks(css=css, title="Interactive 2D (3x2 layout)", theme=gr.themes.Soft(), analytics_enabled=False) as demo:
    gr.Markdown("<h3 style='color:#114b8b'>Interactive-MEN-RT Segmentation</h3><div class='smallnote'>Domain-Specialized Interactive Segmentation Framework for Meningioma Radiotherapy Planning</div>")

    with gr.Row():
        with gr.Column(scale=1, min_width=280, elem_classes=["round"]):
            gr.Markdown("<div class='section'>Samples &amp; Tools</div>")
            gallery = gr.Gallery(value=THUMBS, columns=len(EXAMPLES), height=110, allow_preview=False, preview=False, show_label=False)
            case_dd = gr.Dropdown(choices=EXAMPLES, value=EXAMPLES[0], label="Case")
            with gr.Row():
                seg_btn = gr.Button("Segment", variant="primary")
                undo_btn = gr.Button("Undo")
                clr_btn = gr.Button("Clear")

            gr.Markdown("<div class='section' style='margin-top:8px'>Interactions</div>")
            seeds_df = gr.Dataframe(
                headers=["#", "view", "x", "y", "z"],
                value=[],
                datatype=["number","str","number","number","number"],
                interactive=False,
                wrap=True,
                row_count=(0, "dynamic")
            )
            seed_dd = gr.Dropdown(choices=[], value=None, label="Go to (select a point)")

        with gr.Column(scale=5):
            with gr.Row():
                with gr.Column(elem_classes=["round"]):
                    axial = gr.Image(type="pil", interactive=True, height=RENDER_PX_DEFAULT+8, label="Axial (Z)")
                    z_bar = gr.Slider(0,1,value=0,step=1,label="Z", elem_classes=["tiny"])
                with gr.Column(elem_classes=["round"]):
                    sagittal = gr.Image(type="pil", interactive=True, height=RENDER_PX_DEFAULT+8, label="Sagittal (X)")
                    x_bar = gr.Slider(0,1,value=0,step=1,label="X", elem_classes=["tiny"])
                with gr.Column(elem_classes=["round"]):
                    coronal = gr.Image(type="pil", interactive=True, height=RENDER_PX_DEFAULT+8, label="Coronal (Y)")
                    y_bar = gr.Slider(0,1,value=0,step=1,label="Y", elem_classes=["tiny"])
            with gr.Row():
                out_ax_gt = gr.Image(type="pil", interactive=False, height=RENDER_PX_DEFAULT+8, label="Axial GT (fill)")
                out_ax_pr = gr.Image(type="pil", interactive=False, height=RENDER_PX_DEFAULT+8, label="Axial Pred (fill)")
                inter2d   = gr.Image(type="pil", interactive=False, height=RENDER_PX_DEFAULT+8, label="Interaction (2D)")

    # initial
    demo.load(lambda: on_load(EXAMPLES[0]), [],
              [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd])

    # case change
    case_dd.change(lambda cid: on_load(cid), [case_dd],
                   [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd])

    # gallery select
    gallery.select(on_gallery_select, [],
                   [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd, case_dd])

    # clicks — pure callbacks that only receive event objects (avoid coordinate None issue)
    axial.select(on_axial_select, [],
                 [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd])
    sagittal.select(on_sagittal_select, [],
                    [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd])
    coronal.select(on_coronal_select, [],
                   [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd])

    # bars
    z_bar.release(on_z_release, [z_bar],
                  [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd])
    x_bar.release(on_x_release, [x_bar],
                  [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd])
    y_bar.release(on_y_release, [y_bar],
                  [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd])

    # buttons
    seg_btn.click(on_seg_button, [],
                  [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd])
    clr_btn.click(on_clear, [],
                  [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd])
    undo_btn.click(on_undo, [],
                   [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd])

    # debug list selection
    seeds_df.select(on_seed_df_select, [],
                    [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd])
    seed_dd.change(on_seed_dd_change, [seed_dd],
                   [axial, sagittal, coronal, out_ax_gt, out_ax_pr, inter2d, z_bar, x_bar, y_bar, seeds_df, seed_dd])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, share=False,
                allowed_paths=[str(DATA_ROOT)])
