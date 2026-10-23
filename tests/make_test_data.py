import numpy as np, json, os, subprocess
# Generates phone-like Moon photos (4032x3024 JPEG) and videos with known positions for tests/e2e.js.
from PIL import Image, ImageFilter
rng = np.random.default_rng(3)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gen')

# ---- ground-truth moon, 1024 px ----
S = 1024; R = S/2
y, x = np.mgrid[0:S, 0:S].astype(np.float32)
nx, ny = (x-R)/R, (y-R)/R; r2 = nx*nx+ny*ny; disk = r2 <= 1
def fbm(octaves):
    acc = np.zeros((S, S), np.float32); amp = 1
    for o in range(octaves):
        g = 4*2**o
        small = rng.random((g+1, g+1)).astype(np.float32)
        acc += amp*np.asarray(Image.fromarray(small).resize((S, S), Image.BICUBIC)); amp *= 0.5
    return acc/ (2-2**(1-octaves))
m = fbm(5)
alb = np.where(m > 0.58, 0.5, 0.82) + (fbm(6)-0.5)*0.12
for _ in range(700):
    cx, cy = rng.random(2)*S; cr = (rng.random()**3)*S*0.06+3
    d = np.hypot(x-cx, y-cy)/cr
    alb -= 0.12*np.clip(1-d, 0, 1)**0.5 * (d < 1)
    alb += 0.10*np.exp(-((d-1)/0.12)**2)
limb = 0.55+0.45*np.sqrt(np.clip(1-r2, 0, 1))
edge = -0.4*np.sqrt(np.clip(1-ny*ny, 0, 1))
term = np.clip((nx-edge)*12+0.5, 0, 1)
moon = np.clip(alb*limb*term, 0, 1)*disk
rgb = np.stack([moon*1.03, moon, moon*0.94], -1)
Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(f'{OUT}/truth.png')
moonA = disk.astype(np.float32)

def frame(W, H, diam, cx, cy, blur, gain, noise, light=False, cloud=False):
    sky = np.linspace(14, 34, H, dtype=np.float32)[:, None, None]*np.ones((1, W, 3), np.float32)*[0.8, 0.85, 1.0]
    img = sky.copy()
    if not cloud:
        m = Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).resize((diam, diam), Image.LANCZOS)
        if blur > 0: m = m.filter(ImageFilter.GaussianBlur(blur))
        a = np.asarray(Image.fromarray((moonA*255).astype(np.uint8)).resize((diam, diam), Image.LANCZOS).filter(ImageFilter.GaussianBlur(max(blur, 0.5))))/255.0
        mm = np.asarray(m).astype(np.float32)*gain
        x0, y0 = int(round(cx-diam/2)), int(round(cy-diam/2))
        sub = img[y0:y0+diam, x0:x0+diam]
        img[y0:y0+diam, x0:x0+diam] = sub*(1-a[..., None]) + np.maximum(sub, mm)*a[..., None]
        cx, cy = x0+diam/2, y0+diam/2
    if light:
        d = np.hypot(*np.mgrid[0:H, 0:W][::-1].astype(np.float32) - np.array([W*0.9, H*0.88])[:, None, None])
        img += (255*np.exp(-(d/18)**2))[..., None]
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)), (cx, cy)

truth = {}
os.makedirs(f'{OUT}/photos', exist_ok=True)
W, H = 4032, 3024
for i in range(24):
    cx = W/2 + rng.uniform(-300, 300); cy = H/2 + rng.uniform(-250, 250)
    blur = rng.uniform(0.6, 3.2); gain = rng.uniform(0.85, 1.08)
    im, c = frame(W, H, 400, cx, cy, blur, gain, 7, light=(i % 5 == 0), cloud=(i in (7, 19)))
    name = f'IMG_{4100+i}.jpg'
    if i == 3:  # portrait-stored with EXIF rotate tag, like a phone held sideways
        exif = Image.Exif(); exif[0x0112] = 6
        im.transpose(Image.ROTATE_90).save(f'{OUT}/photos/{name}', quality=88, exif=exif)
    else:
        im.save(f'{OUT}/photos/{name}', quality=88)
    truth[name] = {'cx': c[0], 'cy': c[1], 'blur': blur, 'cloud': i in (7, 19)}
json.dump(truth, open(f'{OUT}/truth.json', 'w'))

# ---- video: 1920x1080, moon 320 px, smooth hand-shake ----
os.makedirs(f'{OUT}/vframes', exist_ok=True)
px, py = 960.0, 540.0
for k in range(90):
    px += rng.normal(0, 4); py += rng.normal(0, 4)
    im, _ = frame(1920, 1080, 320, px, py, rng.uniform(0.6, 2.5), rng.uniform(0.95, 1.02), 5)
    im.save(f'{OUT}/vframes/{k:03d}.png')
for args, out in [(['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20'], 'moon_h264.mp4'),
                  (['-c:v', 'libvpx-vp9', '-b:v', '4M'], 'moon_vp9.webm')]:
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', '30', '-i', f'{OUT}/vframes/%03d.png', *args, f'{OUT}/{out}'], check=True)
print('ok')
