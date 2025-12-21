import numpy as np

from lgff.datasets.single_loader import SingleObjectDataset


def test_compute_roi_intrinsic_no_pad():
    K = np.array([[1000.0, 0, 320.0], [0, 1000.0, 240.0], [0, 0, 1]], dtype=np.float32)
    roi = (100, 50, 200, 100)  # x0, y0, w, h
    K_new = SingleObjectDataset.compute_roi_intrinsic(K, roi, out_size=(400, 200), pad_xy=(0, 0))

    assert np.isclose(K_new[0, 0], 1000 * 400 / 200)
    assert np.isclose(K_new[1, 1], 1000 * 200 / 100)
    assert np.isclose(K_new[0, 2], (320.0 - 100) * (400 / 200))
    assert np.isclose(K_new[1, 2], (240.0 - 50) * (200 / 100))


def test_compute_roi_intrinsic_with_pad():
    K = np.array([[800.0, 0, 200.0], [0, 800.0, 150.0], [0, 0, 1]], dtype=np.float32)
    roi = (20, 10, 100, 50)
    pad_xy = (5, 7)
    K_new = SingleObjectDataset.compute_roi_intrinsic(K, roi, out_size=(200, 100), pad_xy=pad_xy)

    sx = 200 / 100
    sy = 100 / 50
    assert np.isclose(K_new[0, 0], 800 * sx)
    assert np.isclose(K_new[1, 1], 800 * sy)
    assert np.isclose(K_new[0, 2], (200 - 20) * sx + pad_xy[0])
    assert np.isclose(K_new[1, 2], (150 - 10) * sy + pad_xy[1])


def test_mask_validity_thresholds():
    ds = object.__new__(SingleObjectDataset)
    ds.min_mask_nonzero_ratio = 0.1
    mask_zero = np.zeros((2, 2), dtype=bool)
    assert not ds._is_mask_valid(mask_zero)

    mask_sparse = np.array([[1, 0], [0, 0]], dtype=bool)
    assert not ds._is_mask_valid(mask_sparse)

    mask_ok = np.array([[1, 0], [0, 1]], dtype=bool)
    assert ds._is_mask_valid(mask_ok)
