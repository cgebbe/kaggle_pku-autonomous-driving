from tqdm.notebook import tqdm as tqdm
import torch


def predict(model,
            device,
            dataset_torch,
            params,
            ):
    data_loader = torch.DataLoader(dataset=dataset_torch,
                                   batch_size=params['batch_size'],
                                   shuffle=False,
                                   num_workers=0,
                                   )

    # perform predictions
    predictions = []
    model.eval()
    for img, _, _ in tqdm(data_loader):
        with torch.no_grad():
            output = model(img.to(device))
        output = output.data.cpu().numpy()
        for mat in output:
            item = dataset_torch.convert_mat_to_item(mat)
            if params['flag_plot']:
                item.plot()

            string = item.get_cars_as_string(flag_submission=True)
            predictions.append(string)

    # predictions to csv
    df_out = dataset_torch.dataset.df_cars
    df_out['PredictionString'] = predictions
    return df_out
