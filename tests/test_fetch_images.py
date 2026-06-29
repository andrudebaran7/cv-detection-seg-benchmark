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
