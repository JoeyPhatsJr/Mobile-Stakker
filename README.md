# Mobile-Stakker

**Lunar Stacker** is a single-page web app for stacking phone photos of the Moon. It runs entirely in the browser, so nothing is uploaded.

## Use it

1. Open `index.html` on your phone. To host it, enable GitHub Pages under *Settings → Pages → Source: GitHub Actions*; the workflow deploys on every push to `main`.
2. Tap **Choose photos or a video** and pick a burst of Moon shots (20–60 is ideal) or a short video.
3. Adjust **Keep sharpest**, sharpening and levels, then **Save image**.

## How it works

1. **Find the Moon.** Each frame is thresholded and the largest bright blob is taken as the Moon.
2. **Crop.** Every frame is cropped to the same square around the Moon.
3. **Score.** Sharpness is the variance of the Laplacian over the lit disk. The sharpest frame becomes the reference.
4. **Align.** FFT phase correlation gives a coarse shift, then a sum-of-squared-differences search refines it to sub-pixel accuracy.
5. **Stack.** The best N% of frames are averaged, with sigma clipping to reject outliers.
6. **Process.** Two-scale unsharp masking (fine detail and craters), black point, gamma, highlights and colour saturation for "mineral moon" shots.

Limitations: it corrects drift but not field rotation or zoom changes between frames.
