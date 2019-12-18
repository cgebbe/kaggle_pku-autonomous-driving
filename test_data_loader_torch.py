import data_loader
import data_loader_torch
import numpy as np
import copy


def test_roll_conversion():
    """ Test conversion roll > roll_new > roll
    """
    list_roll_test = [-np.pi, -1, 0, 1, +np.pi - 0.001]
    for roll_org in list_roll_test:
        roll_new = data_loader_torch.convert_roll_to_roll_new(roll_org)
        roll = data_loader_torch.convert_roll_new_to_roll(roll_new)
        # roll += 0.1
        assert np.allclose(roll, roll_org)


def test_item_to_mat_conversion():
    """ Test conversion mat > item > mat
    NOT testing item > mat > item, because comparing items more tedious (first need association!),
    """
    # get "original" mat
    dataset = data_loader.DataSet('../data/train')
    list_ids = dataset.list_ids
    item_org = dataset.load_item(list_ids[0], flag_load_mask=False)
    dataset_torch = data_loader_torch.DataSetTorch(dataset)
    mat_org = dataset_torch.convert_item_to_mat(item_org)

    # convert back
    item = dataset_torch.convert_mat_to_item(mat_org)
    item.img = item_org.img  # need this for back conversion!
    mat = dataset_torch.convert_item_to_mat(item)

    assert np.allclose(mat, mat_org)


if __name__ == '__main__':
    test_roll_conversion()
    test_item_to_mat_conversion()
    print("=== Finished")
