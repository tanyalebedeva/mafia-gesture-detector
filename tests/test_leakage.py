from mafia_cv.leakage import group_split, leakage_report, original_stem, video_key


def test_original_stem_strips_roboflow_suffix():
    assert original_stem("IMG_0596_MOV-0003_jpg.rf.4f2c9a1b.jpg") == "IMG_0596_MOV-0003"
    assert original_stem("plain_photo.png") == "plain_photo"


def test_video_key_groups_frames_of_one_video():
    a = video_key("IMG_0596_MOV-0003_jpg.rf.aaa.jpg")
    b = video_key("IMG_0596_MOV-0017_jpg.rf.bbb.jpg")
    c = video_key("IMG_0601_MOV-0003_jpg.rf.ccc.jpg")
    assert a == b == "IMG_0596_MOV"
    assert c != a


def test_single_photos_are_not_merged_into_one_group():
    # IMG_0596 и IMG_0597 - разные снимки, а не кадры видео "IMG"
    assert video_key("IMG_0596_jpg.rf.x.jpg") != video_key("IMG_0597_jpg.rf.y.jpg")


def test_leakage_report_detects_shared_videos():
    files = {
        "train": ["v1-0001_jpg.rf.a.jpg", "v1-0002_jpg.rf.b.jpg", "v2-0001_jpg.rf.c.jpg"],
        "valid": ["v3-0001_jpg.rf.d.jpg"],
        "test": ["v1-0003_jpg.rf.e.jpg", "v4-0001_jpg.rf.f.jpg"],
    }
    rep = leakage_report(files)
    assert rep["test"]["images_from_train_videos"] == 1
    assert rep["test"]["share_from_train_videos"] == 0.5
    assert rep["valid"]["images_from_train_videos"] == 0


def test_group_split_keeps_groups_together_and_roughly_hits_ratios():
    groups = {f"video{i}": 10 for i in range(30)}
    assign = group_split(groups, seed=0)
    assert set(assign) == set(groups)
    sizes = {s: sum(groups[g] for g, sp in assign.items() if sp == s) for s in ("train", "valid", "test")}
    assert sizes["train"] == 240
    assert sizes["valid"] == 30
    assert sizes["test"] == 30
