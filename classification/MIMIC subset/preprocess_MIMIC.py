import pandas as pd
import os

cur_dir = os.getcwd()
csv_path = os.path.join(cur_dir, "MIMIC-JPG-SAMPLE-LABELS.csv")

mimic_df = pd.read_csv("/Users/mpb/Image_Analytics/MIMIC-CXR-JPG-SAMPLE-1K/MIMIC-JPG-SAMPLE-LABELS.csv")

substring = "2.0.0"

# Only a certain part of file_path
new_paths = [path.split(substring, 1)[1].lstrip() for path in mimic_df['file_path']]
mimic_df['file_path'] = new_paths


columns_to_remove = ['dicom_id','subject_id','study_id','weight_lbs','height_ins','BMI_kgm2','gender','age','path']

# Remove the specified columns
mimic_df = mimic_df.drop(columns=columns_to_remove)

# forward slashes for me
mimic_df['file_path'] = mimic_df['file_path'].str.replace('\\', '/')

# new csv
mimic_df.to_csv(os.path.join(cur_dir, "MIMIC-CXR-JPG-SAMPLE-1K/mimic_small.csv"))





