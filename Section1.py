import numpy as np
import pandas as pd
import schedule
import time


def process_csv(path_in, path_out):
    """
    :param path_in: source data file path
    :param path_out: target data file path
    :return: None
    """
    # Read name and price as string
    df = pd.read_csv(path_in, dtype={'name': str, 'price': str})
    # Drop rows with empty name
    df = df[df['name'].notna()]

    first_names = []
    last_names = []
    for full_name in df['name'].values:
        # Remove honorific and suffix such as
        #    'Mr.', 'Mrs.', 'Jr.', 'Sr.', 'I', 'II', 'III', 'PhD', 'MD', 'DDS'
        full_name_split = full_name.split()
        for sub_str in full_name_split:
            if not (sub_str.isalpha() and sub_str[1:].islower()):
                full_name_split.remove(sub_str)

        last_names.append(full_name_split[-1])
        first_names.append(' '.join(full_name_split[0: -1]))

    # Convert prices in string to numerical values
    prices = np.float32(df['price'].values)
    # Write to a CSV file
    df_out = pd.DataFrame({'first_name': first_names,
                           'last_name': last_names,
                           'price': prices,
                           'above_100': (prices > 100)})
    df_out.to_csv(path_out, index=False)


if __name__ == '__main__':
    # Repeat the task at 1pm everyday
    def daily_job():
        process_csv('dataset1.csv', 'dataset1_out.csv')
        process_csv('dataset2.csv', 'dataset2_out.csv')

    schedule.every().day.at('13:00').do(daily_job)

    while True:
        schedule.run_pending()
        time.sleep(1)



