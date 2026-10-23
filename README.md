# Mobile-Stakker

**Lunar Stacker** is a single-page web app for stacking phone photos of the Moon. It runs entirely in the browser, so nothing is uploaded.

## Use it

1. Open the hosted page on your phone: https://joeyphatsjr.github.io/Mobile-Stakker/ (deployed by `.github/workflows/pages.yml` on every push to `main`).
2. Tap **Choose photos or a video** and pick a burst of Moon shots (20–60 is ideal) or a short video.
3. Adjust **Keep sharpest**, sharpening and levels, then **Save image**.

## How it works

1. **Find the Moon.** Each frame is thresholded and the largest bright blob is taken as the Moon.
2. **Crop.** Every frame is cropped to the same square around the Moon.
3. **Score.** Sharpness is the variance of the Laplacian over the lit disk. The sharpest frame becomes the reference.
4. **Align.** FFT phase correlation gives a coarse shift, then a sum-of-squared-differences search refines it to sub-pixel accuracy.
5. **Stack.** The best N% of frames are averaged, with sigma clipping to reject outliers.
6. **Upscale 2× (default: AI).** ESRGAN-slim (an RDN trained on DIV2K, ~1 MB, run with TensorFlow.js in tiles) proposes detail, then iterative back-projection corrects it until shrinking it back reproduces the stack exactly, so it can't keep detail the photos contradict. Falls back to bicubic + back-projection if the model can't load. "Frame shifts" is drizzle-style multi-frame super-resolution.
7. **Denoise.** Starlet (à trous wavelet) shrinkage. Noise per scale is measured in the empty sky and scaled for brighter pixels by the photon-noise curve measured from frame-to-frame scatter.
8. **Deblur.** The blur is measured from the Moon's own limb (a perfectly sharp edge in reality): a robust circle fit, then the sky-side width of the averaged edge profile. Damped Richardson–Lucy (White 1994) undoes it, and pixels stop changing once they fit their data within the noise. A runtime guard checks the result for a dark ring at the limb or amplified sky grain and backs off if either appears.
9. **Process.** Black point, gamma, highlights and colour saturation for "mineral moon" shots. Extra unsharp sharpening is available but off by default, because it creates halos.

### What the enhancement does and doesn't do

Measured against the ground-truth Moon in `tests/` (`node tests/quality.js`), the default pipeline recovers real detail (fine-detail correlation with the true Moon 0.84 → 0.91 for a 5× phone-zoom shot) with no halo, less sky noise, and no hatching in smooth areas. Raw AI output, without the back-projection step, scored *worse* than plain bicubic and added a glow ring, which is why it's always held to the data. It can't invent resolution your photos don't contain.

Limitations: it corrects drift but not field rotation or zoom changes between frames.

## Tests

`tests/make_test_data.py` renders 24 phone-style 4032×3024 JPEGs (small Moon, sky glow, a streetlight, shake, blur, noise, two clouded-out frames, one EXIF-rotated) plus H.264 and VP9 videos, all with known Moon positions. `tests/e2e.js` serves the app, emulates a phone in Playwright, uploads them and checks frame rejection, sub-pixel alignment against ground truth, noise reduction, every control, saving, and video frame extraction.

`tests/quality.js` scores the enhancement against the true Moon on two sets (a tiny undersampled 5×-zoom Moon and a larger blur-limited one) and fails on less real detail, a softer limb, halos, extra noise, distorted brightness or patterning in smooth areas.

```sh
pip install numpy pillow   # ffmpeg must be on PATH
python3 tests/make_test_data.py
npm i playwright @tensorflow/tfjs@4.22.0 @upscalerjs/esrgan-slim@1.0.0
node tests/e2e.js && node tests/quality.js
```
