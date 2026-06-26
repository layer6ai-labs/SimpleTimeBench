import pandas as pd
from IPython.display import display

folder_path = "data/Hahn_survey"
# Specify the input .txt file and output .csv file paths
input_txt_file = f"{folder_path}/LD2011_2014.txt"
output_csv_file = input_txt_file.replace('.txt', '.csv')
# Read the .txt file into a DataFrame
# Assuming the .txt file is tab-delimited or space-delimited
df = pd.read_csv(input_txt_file, delimiter=";")  # Change delimiter if needed

# Save the DataFrame to a .csv file
df.to_csv(output_csv_file, index=False)

print(f"File converted and saved as {output_csv_file}")

display(df)