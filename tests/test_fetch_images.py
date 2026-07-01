import io
import zipfile

from benchmark import fetch_images


def test_image_ids_are_at_least_eight_and_unique():
    ids = fetch_images.IMAGE_IDS
    assert len(ids) >= 8
    assert len(set(ids)) == len(ids)


def test_url_for_builds_coco_val2017_url():
    assert fetch_images.url_for(139) == "http://images.cocodataset.org/val2017/000000000139.jpg"


def test_target_dir_points_at_repo_data():
    p = fetch_images.target_dir()
    assert p.name == "benchmark_images"
    assert p.parent.name == "data"


def test_dist_dir_points_at_repo_data():
    p = fetch_images.dist_dir()
    assert p.name == "dist_images"
    assert p.parent.name == "data"


def test_dist_zip_url_is_coco128():
    assert fetch_images.DIST_ZIP_URL.endswith("coco128.zip")


def _make_zip(path, names):
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            # a tiny valid-enough byte payload; extraction copies bytes verbatim
            zf.writestr(name, b"\xff\xd8\xff\xe0jpegbytes")


def test_extract_jpgs_collects_sorted_and_limited(tmp_path):
    zpath = tmp_path / "coco128.zip"
    _make_zip(zpath, [
        "coco128/images/train2017/000000000009.jpg",
        "coco128/images/train2017/000000000025.jpg",
        "coco128/images/train2017/000000000030.jpg",
        "coco128/labels/train2017/000000000009.txt",  # non-jpg, must be ignored
    ])
    out = tmp_path / "dist_images"
    got = fetch_images._extract_jpgs(zpath, out, limit=2)
    assert [p.name for p in got] == ["000000000009.jpg", "000000000025.jpg"]
    assert all(p.exists() for p in got)
