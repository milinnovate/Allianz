import os
import pandas as pd

# Define directories
input_directory = "/Users/milindsoni/Documents/projects/Allianz/presentationdata/output2"  # Update with the correct path to your 'output2' directory
holiday_directory = "/Users/milindsoni/Documents/projects/Allianz/presentationdata/output2/holidays"  # Update with the correct path to your 'holidays' directory
result_directory = "/Users/milindsoni/Documents/projects/Allianz/presentationdata/output2/processedwithhol"  # Update with the correct path for saving final results

if not os.path.exists(result_directory):
    os.makedirs(result_directory)

def mark_holidays_and_long_weekends(df, holidays):
    # Ensure 'Date' in holidays is the correct datetime format
    holidays['Date'] = pd.to_datetime(holidays['Date'], format='%d-%m-%Y', errors='coerce')
    df['Day'] = pd.to_datetime(df['Day'], errors='coerce')

    # Mark holidays in the main dataframe
    df['Holidays'] = df['Day'].isin(holidays['Date']).astype(int)

    # Prepare to mark long weekends
    df.sort_values('Day', inplace=True)  # Ensure dates are sorted
    df['Long Weekend'] = 0

    # Iterate through the DataFrame to mark long weekends
    for idx, row in df.iterrows():
        if row['Holidays'] == 1:
            current_date = row['Day']
            dow = current_date.dayofweek

            # Check and mark long weekends that include Friday, Saturday, Sunday
            if dow == 4:  # Friday
                if (idx + 1 < len(df) and df.iloc[idx + 1]['Holidays'] == 1 and df.iloc[idx + 1]['Day'].dayofweek == 5 and
                    idx + 2 < len(df) and df.iloc[idx + 2]['Holidays'] == 1 and df.iloc[idx + 2]['Day'].dayofweek == 6):
                    df.loc[idx: idx + 2, 'Long Weekend'] = 1
            
            # Check and mark long weekends that include Sunday, Monday, Tuesday
            if dow == 0:  # Monday
                if (idx - 1 >= 0 and df.iloc[idx - 1]['Holidays'] == 1 and df.iloc[idx - 1]['Day'].dayofweek == 6 and
                    idx + 1 < len(df) and df.iloc[idx + 1]['Holidays'] == 1 and df.iloc[idx + 1]['Day'].dayofweek == 1):
                    df.loc[idx - 1: idx + 1, 'Long Weekend'] = 1

    return df

if not os.path.exists(result_directory):
    os.makedirs(result_directory)

# Process each file
for file_name in os.listdir(input_directory):
    if file_name.startswith('enhanced') and file_name.endswith('.csv'):
        district_name = file_name.split('_')[1]
        holiday_file_name = district_name.replace(' ', '_') + '_holidays_2023.csv'

        df = pd.read_csv(os.path.join(input_directory, file_name))
        holiday_file_path = os.path.join(holiday_directory, holiday_file_name)
        if os.path.exists(holiday_file_path):
            holidays = pd.read_csv(holiday_file_path)
            df = mark_holidays_and_long_weekends(df, holidays)
            df.to_csv(os.path.join(result_directory, f'final_{file_name}'), index=False)
        else:
            print(f"Holiday file not found for {district_name}: {holiday_file_path}")

print("Processing complete.")