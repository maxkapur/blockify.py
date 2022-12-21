# blockify.py

`blockify.py` is a Python library that decomposes an image into rectangular blocks
of solid color. Its intended use is to compute very lightweight SVG approximations
to an input image that can be displayed while the image is loading.

Inspired by <https://github.com/fogleman/primitive>, but `blockify.py` chooses a different
set of tradeoffs. In particular, by partitioning the image into nonoverlapping "panes"
instead of allowing transparent, intersecting polygons, we sacrifice fidelity to the original
image in favor of a truly small filesize in the vector approximation and fast computation time.

## Examples

Preliminary imports:

```python
import blockify
import matplotlib.pyplot as plt

plt.style.use("dark_background")
```

Main interface (see below for a discussion of the options):

```python
teapot_blockified = blockify.blockify_image(
    "./test_images/teapot.jpg",
    250,
    alternator_type="strict",
    strategy="largest_pane"
)

def plot_comparison(blockified_image):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].imshow(blockified_image.image_array)
    ax[0].set_title("Original image")
    ax[1].imshow(blockified_image.to_array())
    ax[1].set_title("Blockified image")

plot_comparison(teapot_blockified)
```

![Comparison of an image of a teapot and its blockified approximation](./test_images/teapot_blockified_comparison.png)

Another example:

```python
sculpture_blockified = blockify.blockify_image(
    "./test_images/sculpture.jpg", 
    350,
    alternator_type="randomized",
    strategy="largest_pane"
)
plot_comparison(sculpture_blockified)
```

![Comparison of an image of an abstract sculpture and its blockified approximation](./test_images/sculpture_blockified_comparison.png)

## Usage

Call the function `blockify.blockify_image(filename, n)`, where `filename` is a path to an image file (must be readable by `imageio`) and `n` is the number of panes desired. The optional keyword arguments `alternator_type` and `strategy` are described below.

The function returns a `BlockifiedImage` instance with methods `.to_array()` and `.to_png(filename)`. `.to_svg(filename)` is on the to-do list. You can also use `.image_array` to inspect the original image array. 

## How it works

To split an image horizontally into two panes (left and right):

1. For each possible split between `0` and the image's width, compute the sum of squared deviations between each pixel hue and the average hue within the left and right panes. (We do this in linear time by adapting a rolling-variance function from Wikipedia.)
2. Choose the value that minimizes the sum of the squared deviations between the left and right panes.
3. Replace pixels to the left (right) of the split point with the average hue in the left (right) pane.

Splitting an image vertically is analogous.

To decompose an image into `n` panes, we apply the "split an image" routine recursively, alternating between horizontal and vertical splits. There are many possible strategies for selecting which pane to split next; `blockify.py` implements the following as options to `blockify.blockify_image()`:

- `strategy="variance"` splits the pane that has the highest squared deviation from its average hue (i.e., the highest variance weighted by the pane area)
- `strategy="largest_pane"` always splits the pane of largest area.

There are also several strategies for alternating between horizontal and vertical splits:
- `alternator_type="strict"`: Strict alternation.
- `alternator_type="bresenham"`: Alternate according to Bresenham's line algorithm, performing one type of split by default and the other type whenever the fractional part overflows. This means that, for example, "tall" images get more vertical splits than horizontal splits, in proportion to the aspect ratio, which is aesthetically pleasing.
- `alternator_type="randomized"`: Randomly select splits with probability matching the aspect ratio.

## Attribution

The test images are from <a href="https://unsplash.com/@lvenfoto?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Zhang liven</a> on <a href="https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>.