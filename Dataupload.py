import os
import pandas as pd
class Dataupload:
    def _init_(self,user_id,project_id):
        self.user_id = user_id
        self.project_id = project_id
    def upload_images(self,filename):
        fn = os.path.basename(filename)
        open(fn,'wb').write(filename.file.read())
    def upload_label(self,label_file):
        fn = os.path.basename(label_file)
        open(fn,'wb').write(label_file)
        self.label_data = pd.read_csv(label_file.file.read())




