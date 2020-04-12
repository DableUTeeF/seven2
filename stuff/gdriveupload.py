from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

if __name__ == '__main__':
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("stuff/mycreds.txt")
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
    gauth.SaveCredentialsFile("stuff/mycreds.txt")

    src_snapshot = 'snapshots/pairs/11'

    dest_images = {"title": "pairs", "id": "17r6Yv5Jt8hbBU_PJ7gN-W9Wt_NaaAPvL"}

    drive = GoogleDrive(gauth)

    # file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    file_list = drive.ListFile({'q': "'17r6Yv5Jt8hbBU_PJ7gN-W9Wt_NaaAPvL' in parents and trashed=false"}).GetList()
    for file1 in file_list:
        print('title: %s, id: %s' % (file1['title'], file1['id']))

    # exit()
    try:
        for files in os.listdir(src_snapshot):
            textfile = drive.CreateFile({'title': files, "parents": [{"kind": "drive#fileLink", "id": dest_images['id']}]})
            textfile.SetContentFile(os.path.join(src_snapshot, files))
            textfile.Upload()
            print('Uploaded:', files)
    except Exception as e:
        print(e)
