import torch as t
import os


def main(
    in_file_name="coastal_sea_data_preprocessed_cur.pt", out_dir: str = "data"
):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(os.path.join(out_dir, "test")):
        os.mkdir(os.path.join(out_dir, "test"))
    if not os.path.exists(os.path.join(out_dir, "train")):
        os.mkdir(os.path.join(out_dir, "train"))

    data = t.load(in_file_name)
    test_size = int(0.2 * len(data))
    to_cut = (test_size + 16) // 2
    test_1 = data[:to_cut]
    test_2 = data[-to_cut:]

    train = data[to_cut:-to_cut]

    assert len(train) + len(test_1) + len(test_2) == len(data), "whoops"

    t.save(test_1, os.path.join(out_dir, "test", "1.pt"))
    t.save(test_2, os.path.join(out_dir, "test", "2.pt"))
    t.save(train, os.path.join(out_dir, "train", "1.pt"))


if __name__ == "__main__":
    main()
