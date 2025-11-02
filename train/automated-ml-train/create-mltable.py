import mltable
import os

csv_path = os.path.abspath('../../data/creditcard.csv')

paths = [
    {'file': csv_path}
]

train_table = mltable.from_delimited_files(paths=paths, delimiter=',', header='all_files_same_headers')
train_table.save('./train_data')
