#  Download weights file from gdrive
import requests
import os


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        print("saving response")
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


if __name__ == '__main__':
    # params
    id_file = '1ebpi0TV6_fmsIwmK9cWR55TSvO3ozIM4'
    #path_dst = '~/output/20200119_focal_v3_kaggle_nb/model_9.pth'
    path_dst = 'model.pth'

    # run file
    #os.makedirs(os.path.dirname(path_dst), exist_ok=True)
    download_file_from_google_drive(id_file, path_dst)
    print("=== Finished")
