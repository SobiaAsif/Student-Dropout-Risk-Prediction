import pandas as pd

df = pd.read_csv("data/student_burnout_dropout_dataset_2.csv")

categorical_columns = [
    "Gender",
    "Department",
    "Residence_Type",
    "Part_Time_Job",
    "Family_Income_Bracket",
    "Counseling_Access",
    "Burnout_Level"
]

for col in categorical_columns:
    print(f"\n{col}:")
    print(sorted(df[col].dropna().unique()))