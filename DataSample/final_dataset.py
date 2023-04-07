import pandas as pd
import numpy as np
from faker import Faker


def transform_downloaded_datasets():
    # Read CSV files
    credit_card_data = pd.read_csv('creditcard.csv')
    paysim_data = pd.read_csv('paysim.csv')

    # Preprocess and standardize the Credit Card dataset
    credit_card_data['transaction_type'] = credit_card_data['Class'].apply(
        lambda x: 'genuine' if x == 0 else 'fraudulent')
    credit_card_data['user_id'] = credit_card_data.index.map(lambda x: f"user_{x}")
    credit_card_data['timestamp'] = pd.to_datetime(credit_card_data['Time'], unit='s')
    credit_card_data['merchant'] = 'Unknown'  # We don't have merchant info in this dataset
    credit_card_data_standardized = credit_card_data[['user_id', 'Amount', 'timestamp', 'merchant', 'transaction_type']]
    credit_card_data_standardized.columns = ['user_id', 'transaction_amount', 'timestamp', 'merchant',
                                             'transaction_type']

    # Preprocess and standardize the Paysim dataset
    paysim_data = paysim_data[paysim_data['type'] == 'CASH_OUT']  # Filter to include only CASH_OUT transactions
    paysim_data['transaction_type'] = paysim_data['isFraud'].apply(lambda x: 'genuine' if x == 0 else 'fraudulent')

    # Create a new DataFrame with only the required columns
    paysim_data_standardized = pd.DataFrame()
    paysim_data_standardized['user_id'] = paysim_data['nameOrig']
    paysim_data_standardized['transaction_amount'] = paysim_data['amount']
    paysim_data_standardized['timestamp'] = pd.to_timedelta(paysim_data['step'], unit='h')
    paysim_data_standardized['merchant'] = paysim_data['nameDest']
    paysim_data_standardized['transaction_type'] = paysim_data['transaction_type']

    return credit_card_data_standardized, paysim_data_standardized


def generate_synthetic_data(num_records):
    fake = Faker()

    data = []
    for _ in range(num_records):
        user_id = fake.uuid4()
        transaction_amount = round(np.random.uniform(1, 10000), 2)
        timestamp = fake.date_time_this_year()
        merchant = fake.company()
        transaction_type = np.random.choice(['genuine', 'fraudulent'], p=[0.99, 0.01])

        data.append([user_id, transaction_amount, timestamp, merchant, transaction_type])

    columns = ['user_id', 'transaction_amount', 'timestamp', 'merchant', 'transaction_type']
    return pd.DataFrame(data, columns=columns)


def save_data(data, filename):
    data.to_csv(filename, index=False)


synthetic_data = generate_synthetic_data(10000)
credit_card_data_standardized, paysim_data_standardized = transform_downloaded_datasets()

combined_data = pd.concat([synthetic_data, credit_card_data_standardized, paysim_data_standardized], axis=0,
                          ignore_index=True)

save_data(combined_data, 'combined_data.csv')
