from src.data.build_features import split_dataset


def make_fake_dataset(n_per_class=100):
    good = [(f"good_{i}.png", 0) for i in range(n_per_class)]
    bad = [(f"bad_{i}.png", 1) for i in range(n_per_class)]
    return good + bad


def test_split_sizes_are_70_15_15():
    train, val, test = split_dataset(make_fake_dataset())

    assert (len(train), len(val), len(test)) == (140, 30, 30)


def test_split_has_no_image_in_two_sets():
    train, val, test = split_dataset(make_fake_dataset())

    train_paths = {path for path, _ in train}
    val_paths = {path for path, _ in val}
    test_paths = {path for path, _ in test}

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)