import data_loader


def test_car_to_string():
    """ Test that conversion string -> car -> string works
    """
    dataset = data_loader.DataSet(path_csv='../data/train.csv',
                                  path_folder_images='../data/train_images',
                                  path_folder_masks='../data/train_masks',
                                  )
    cars_as_string = dataset.df_cars.loc[0, 'PredictionString']
    item = data_loader.DataItem()
    item.set_cars_from_string(cars_as_string)
    cars_as_string2 = item.get_cars_as_string()
    # cars_as_string2 += '4'
    assert cars_as_string == cars_as_string2


if __name__ == '__main__':
    test_car_to_string()
    print("=== Finished")
