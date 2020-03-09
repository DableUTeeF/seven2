from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

if __name__ == '__main__':
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")

    src_images = '/media/palm/data/7/images'
    src_anns = '/media/palm/data/7/anns'

    dest_images = {"title": "images", "id": "1n6ksAS1RSBViP-kUlDg9yWEySMFIiSfc"}
    dest_anns = {"title": "anns", "id": "1Sz9DVf28lRB-KATMfhWdhZbOxSZPsh24"}

    drive = GoogleDrive(gauth)

    file_list = drive.ListFile({'q': "'1YyC93dAAnCthmh34s9Ek-gY_Lzf8u1Sl' in parents and trashed=false"}).GetList()
    for file1 in file_list:
        print('title: %s, id: %s' % (file1['title'], file1['id']))

    uploaded = []
    if os.path.exists('/home/palm/PycharmProjects/Seven/stuff/uploaded.txt'):
        uploaded = open('/home/palm/PycharmProjects/Seven/stuff/uploaded.txt').read().split('\n')
    # try:
    #     for image in os.listdir(src_images):
    #         if image in uploaded:
    #             continue
    #         textfile = drive.CreateFile({'title': image, "parents": [{"kind": "drive#fileLink", "id": dest_images['id']}]})
    #         textfile.SetContentFile(os.path.join(src_images, image))
    #         textfile.Upload()
    #         uploaded.append(image)
    #         print('Uploaded:', image)
    #         # break
    #
    #     for ann in os.listdir(src_anns):
    #         if ann in uploaded:
    #             continue
    #         textfile = drive.CreateFile({'title': ann, "parents": [{"kind": "drive#fileLink", "id": dest_anns['id']}]})
    #         textfile.SetContentFile(os.path.join(src_anns, ann))
    #         textfile.Upload()
    #         uploaded.append(ann)
    #         print('Uploaded:', ann)
    #
    #         # break
    # except KeyboardInterrupt:
    #     with open('/home/palm/PycharmProjects/Seven/stuff/uploaded.txt', 'w') as wr:
    #         for u in uploaded:
    #             wr.write(u)
    #             wr.write('\n')
    # except Exception as e:
    #     print(e)
    #     with open('/home/palm/PycharmProjects/Seven/stuff/uploaded.txt', 'w') as wr:
    #         for u in uploaded:
    #             wr.write(u)
    #             wr.write('\n')
