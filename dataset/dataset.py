import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_table("dataset_lnc.txt")

# Split the dataset (80% for training and 20% for testing)
train, test = train_test_split(df, test_size=0.2 , random_state=103, stratify=df['label'])

print(test['label'].value_counts())
# Save the splits into separate files
train.to_csv('lnc_train.txt', sep='\t', index=False)
test.to_csv('lnc_test.txt', sep='\t', index=False)

# Load the dataset
df_lnc = pd.read_table("lnc_test.txt")

# Define the categories and number of samples per category
categories_to_sample = [0, 1, 2, 3, 4]
samples_per_category = 20

# Create a list to store the sampled data
sampled_data = []

# Extract samples for each category
for category in categories_to_sample:
    category_data = df_lnc[df_lnc['label'] == category]
    if len(category_data) >= samples_per_category:
        sampled_data.append(category_data.sample(n=samples_per_category))
    else:
        sampled_data.append(category_data)

# Concatenate all the sampled data into one DataFrame
sampled_df = pd.concat(sampled_data)

# Save the sampled data to a new file
sampled_df.to_csv('independent_test_dataset.txt', sep='\t', index=False)

df_lnc= pd.read_table("independent_test_dataset.txt")
# Assuming df_lnc is already defined as in your dataset.
to_sample_1 = [0,2,3,4]
to_sample_2 = [0,2,4]
to_sample_3 = [1,2,4]

df_lnc1 = df_lnc[df_lnc['label'].isin(to_sample_1)]
df_lnc1.to_csv('independent_task1.txt', sep='\t', index=False)
df_lnc2 = df_lnc[df_lnc['label'].isin(to_sample_2)]
df_lnc2.to_csv('independent_task2.txt', sep='\t', index=False)
df_lnc3 = df_lnc[df_lnc['label'].isin(to_sample_3)]
df_lnc3.to_csv('independent_task3.txt', sep='\t', index=False)