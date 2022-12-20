"""
`blockify.py` is a Python library that decomposes an image into rectangular blocks
of solid color. Its intended use is to compute very lightweight SVG approximations
to an input image that can be displayed while the image is loading.

Inspired by https://github.com/fogleman/primitive, but `blockify.py` chooses a
tradeoff weighted more heavily on "output sparsity" than "fidelity to the original image."

Author: Max Kapur
Email:  maxkapur@gmail.com
Date:   December 2022
"""


import numpy as np
import imageio
from heapq import heappush, heappop


class Pane:
    """
    Instances of this class represent rectangular subsets of an image.
    """

    def __init__(self, x0, x1, y0, y1):
        """
        Instantiate a new `Pane` with the given x- and y-coordinates.
        """
        assert x0 <= x1
        assert y0 <= y1

        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

        self.size = (x1 - x0) * (y1 - y0)

    def __gt__(self, other):
        # Need to overload > because when there are many panes on the heap in
        # _compute_recursive_split...(), ties can occur between the loss function
        # values and Python proceeds to the next element of the tuple for comparison.

        # Just use lexicographical comparison on coords.
        return (self.x0, self.x1, self.y0, self.y1) > (other.x0, other.x1, other.y0, other.y1)

    def split(self, split_point, left_right):
        """
        Split the pane either horizontally (`left_right = True`) or vertically
        (`left_right = False`) at the given split point. Return the two `Pane`s
        that result.
        """
        if left_right:
            return (
                Pane(self.x0, self.x1, self.y0, self.y0 + split_point),
                Pane(self.x0, self.x1, self.y0 + split_point, self.y1)
            )
        else:
            return (
                Pane(self.x0, self.x0 + split_point, self.y0, self.y1),
                Pane(self.x0 + split_point, self.x1, self.y0, self.y1)
            )

    def slice_image(self, image_array):
        """
        Return the slice of the image represented by the `image_array` corresponding
        to the current `Pane`.
        """
        return image_array[self.x0:self.x1, self.y0:self.y1, :]


def pane_loss(image_array):
    """
    Compute the loss, i.e. sum of squared deviations from the average hue, within 
    a given slice.
    """
    return np.var(image_array) * image_array.size


def _optimal_split_left_right(image_array):
    # For each split point, compute the sum of squared deviations from the average
    # hue within the left and right panes. The optimal split point is the one that
    # minimizes the sum of these quantities. (Equivalently, the sum of the variance
    # in each pane weighted by the pane sizes.)

    # Using an adaptation of Chan et al.'s parallel algorithm for computing the variance,
    # we can compute all 2n variances in O(n) time, where n is the width of the image.

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm

    means = image_array.mean(axis=(0, 2))
    n_for_means = image_array.shape[0] * image_array.shape[2]
    sum_squared_deviations = image_array.var(axis=(0, 2)) * n_for_means
    n_split_points = image_array.shape[1]

    loss_left_pane = np.zeros(n_split_points)
    loss_right_pane = np.zeros(n_split_points)

    delta = 0.0
    curr_mean = 0.0
    curr_sum_squared_deviations = 0.0
    for i in range(n_split_points-1):
        delta = means[i] - curr_mean
        curr_mean += delta / (i + 1)
        curr_sum_squared_deviations += sum_squared_deviations[i] + \
            delta**2 * i * n_for_means / (i+1)
        loss_left_pane[i+1] = curr_sum_squared_deviations

    delta = 0.0
    curr_mean = 0.0
    curr_sum_squared_deviations = 0.0
    for i in range(n_split_points):
        delta = means[-i-1] - curr_mean
        curr_mean += delta / (i + 1)
        curr_sum_squared_deviations += \
            sum_squared_deviations[-i-1] + delta**2 * i * n_for_means / (i+1)
        loss_right_pane[-i-1] = curr_sum_squared_deviations

    best = np.argmin(loss_left_pane + loss_right_pane)
    return best, (loss_left_pane[best], loss_right_pane[best])


def optimal_split(image_array, left_right):
    """
    Compute the optimal horizontal (`left_right = True`) or vertical (`left_right = False`)
    split of the given image.
    """
    if left_right:
        return _optimal_split_left_right(image_array)
    else:
        return _optimal_split_left_right(image_array.transpose(1, 0, 2))


def _fill_with_mean_hue(image_array):
    # Replace each pixel hue with the mean hue over the slice
    if image_array.size > 0:
        image_array[:] = image_array.mean(axis=(0, 1))


def split_inplace(image_array, split_point, left_right):
    """
    Perform the given horizontal (`left_right = True`) or vertical (`left_right = False`)
    split by changing each pixel hue to the average hue in that pane. Modifies `image_array`
    in place.
    """
    last = image_array.shape[1] if left_right else image_array.shape[0]
    if (split_point <= 0) or (split_point >= last):
        _fill_with_mean_hue(image_array)
    elif left_right:
        _fill_with_mean_hue(image_array[:, :split_point, :])
        _fill_with_mean_hue(image_array[:, split_point:, :])
    else:
        _fill_with_mean_hue(image_array[:split_point, :, :])
        _fill_with_mean_hue(image_array[split_point:, :, :])


def _strict_alternator(x, y, n):
    # Simply alternate between horizontal and vertical splits,
    # but start by splitting the image along the smaller axis.
    left_right = x < y
    for _ in range(n):
        yield left_right
        left_right = not left_right


def _bresenham_alternator(x, y, n):
    # Iterator for determining whether to perform a horizontal or vertical
    # split in proportion to the image's aspect ratio. If the image is
    # tall (x > y), returns False more often than True (i.e. fewer vertical splits).

    # Based on Bresenham's line algorithm.

    flip = x < y
    x, y = (y, x) if flip else (x, y)

    v = y
    for _ in range(n):
        v += y
        if v > x:
            yield not flip
            v -= x
        else:
            yield flip


def _randomized_alternator(x, y, n):
    # Randomized iterator for determining whether to perform a horizontal or
    # vertical split in proportion to the image's aspect ratio. If the image is
    # tall (x > y), returns False more often than True (i.e. fewer vertical splits).

    flip = x < y
    aspect_ratio = x / y if flip else y / x

    for _ in range(n):
        yield flip ^ (np.random.rand() < aspect_ratio)


def _compute_recursive_split_variance_strategy(image_array, alternator):
    # Split the image into n panels, alternating between horizontal
    # and vertical splits, selecting the largest panel each time.

    # Return a list (in heap order) of Pane objects describing the split
    # locations.

    initial_pane = Pane(0, image_array.shape[0], 0, image_array.shape[1])

    # Binary min heap; root element always stores the largest pane
    # Panes themselves exactly partition the original image
    panes = [
        (-pane_loss(image_array), initial_pane)
    ]

    for left_right in alternator:
        # Select the pane with the highest loss; heuristically, this will benefit
        # most from splitting.
        _, pane = heappop(panes)
        split_point, (loss0, loss1) = optimal_split(
            pane.slice_image(image_array),
            left_right
        )
        pane0, pane1 = pane.split(split_point, left_right)
        heappush(panes, (-loss0, pane0))
        heappush(panes, (-loss1, pane1))

    return panes


def _compute_recursive_split_largest_pane_strategy(image_array, alternator):
    # Split the image into n panels, alternating between horizontal
    # and vertical splits, selecting the lossiest panel each time.

    # Return a list (in heap order) of Pane objects describing the split
    # locations.

    initial_pane = Pane(0, image_array.shape[0], 0, image_array.shape[1])

    # Binary min heap; root element always stores the lossiest pane
    # Panes themselves exactly partition the original image
    panes = [
        (-initial_pane.size, initial_pane)
    ]

    for left_right in alternator:
        # Select the pane with the highest loss; heuristically, this will benefit
        # most from splitting.
        _, pane = heappop(panes)
        split_point, _ = optimal_split(
            pane.slice_image(image_array),
            left_right
        )
        pane0, pane1 = pane.split(split_point, left_right)
        heappush(panes, (-pane0.size, pane0))
        heappush(panes, (-pane1.size, pane1))

    return panes


class BlockifiedImage:
    """
    Class representing an image that has been partitioned into nonoverlapping panes.
    """

    def __init__(self, panes, image_array, alternator_type, strategy) -> None:
        """
        Instantiate a `BlockifiedImage` with the given `panes` and original image
        represented by the `image_array`. `alternator_type` and `strategy` are
        stored for reference purposes.
        """
        self.panes = panes
        self.image_array = image_array
        self.alternator_type = alternator_type
        self.strategy = strategy

    def __repr__(self) -> str:
        a = f"Blockified image instance with {len(self.panes)} panes"
        b = f"Original image dimensions: {'×'.join(map(str, self.image_array.shape))}"
        c = f"Alternator type: {self.alternator_type}"
        d = f"Strategy: {self.strategy}"
        return "\n".join((a, b, c, d))

    def to_array(self):
        """
        Rasterize the blockified image by copying it to a new array and replacing 
        each pixel hue with the average hue in its pane.
        """
        split_image = np.copy(self.image_array)
        for _, pane in self.panes:
            _fill_with_mean_hue(pane.slice_image(split_image))
        return split_image

    def to_svg(self, filename):
        # TODO
        raise NotImplementedError

    def to_png(self, filename):
        """
        Rasterize the blockified image and save it to `filename`. 
        """
        split_image = self.to_array()
        imageio.imsave(filename, split_image)


_alternator_types = {
    "strict": _strict_alternator,
    "bresenham": _bresenham_alternator,
    "randomized": _randomized_alternator
}


_split_strategies = {
    "variance": _compute_recursive_split_variance_strategy,
    "largest_pane": _compute_recursive_split_largest_pane_strategy
}


def blockify_image(filename, n, alternator_type="bresenham", strategy="variance"):
    """
    Blockify the image located at `filename` into `n` panes. Vertical and horizontal
    splits are selected according to the `alternator_type`, and the split pane
    is selected using the `strategy`. 
    """
    # TODO: Document alternator types and split strategies

    image_array = imageio.imread(filename)

    # TODO: Allow user to supply own alternator
    alternator = _alternator_types[alternator_type](
        image_array.shape[0],
        image_array.shape[1],
        n-1
    )

    panes = _split_strategies[strategy](
        image_array,
        alternator
    )

    return BlockifiedImage(
        panes,
        image_array,
        alternator_type,
        strategy
    )
