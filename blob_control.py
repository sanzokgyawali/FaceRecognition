from azure.storage.blob import BlobClient
from azure.storage.blob import ContainerClient


account_name = 'nancy'
account_key = 'lwyozYQUfrsNnCb/L/L0J4S0gauQ5DgBAxd3LewU5Lw76Poe+9oWDzEpaD/zUbAsarsEa6U+M4SDfuoitSaJdQ=='
account_url = 'https://nancy.blob.core.windows.net/'
container_name = "nancy-unidentified-face"
connection_string = "DefaultEndpointsProtocol=https;AccountName={};AccountKey={};EndpointSuffix=core.windows.net".format(account_name, account_key)


def get_container_list():
    container = ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name)

    blob_list = container.list_blobs()
    file_list = []
    for blob in blob_list:
        file_list.append(blob.name)

    print(file_list)
    return file_list


def upload_blob(file_name, blob_name):
    blob = BlobClient.from_connection_string(conn_str=connection_string, container_name=container_name,
                                             blob_name=blob_name)

    with open(file_name, "rb") as data:
        blob.upload_blob(data)


def download_blob(file_name, blob_name):
    blob = BlobClient.from_connection_string(conn_str=connection_string, container_name=container_name,
                                             blob_name=blob_name)

    with open(file_name, "wb") as my_blob:
        blob_data = blob.download_blob()
        blob_data.readinto(my_blob)


def delete_blob(blob_name):
    blob = BlobClient.from_connection_string(conn_str=connection_string, container_name=container_name,
                                             blob_name=blob_name)
    blob.delete_blob()


if __name__ == '__main__':
    # upload_blob('Readme.md', 'test_read,txt')
    # get_container_list()
    # download_blob('test1.md', 'test_read,txt')
    # delete_blob('1599243530.9070566.jpg')
    get_container_list()
