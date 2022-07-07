import pickle
import pandas as pd
import argparse

def run(year, mm):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PUlocationID', 'DOlocationID']

    def read_data(filename):
        df = pd.read_parquet(filename)
        
        df['duration'] = df.dropOff_datetime - df.pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df

    date_str = f'{year:04d}-{mm:02d}'
    print(date_str)
    
    #input_file = f'../data/fhv_tripdata_{date_str}.parquet'
    #input_file = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{date_str}.parquet'
    input_file = f'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/fhv_tripdata_{date_str}.parquet'
    print(f'Downloading data from {input_file}...')
    df = read_data(input_file)
   
    print('Preprocessing data (vectorize)...')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    print('Applying the model...')
    y_pred = lr.predict(X_val)

    print('Mean duration prediction:', y_pred.mean())
    
    # df['ride_id'] = f'{year:04d}/{mm:02d}_' + df.index.astype('str')
    # df_result = pd.DataFrame()
    # df_result['ride_id'] = df['ride_id']
    # df_result['duration_preds'] = y_pred


    # output_file = f'../data/{date_str}_preds.parquet'
    # print('Saving the result to {output_file}...')
    
    # df_result.to_parquet(
    #     output_file,
    #     engine='pyarrow',
    #     compression=None,
    #     index=False
    # )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--year', type=int, default=2021, help="YYYY")
    parser.add_argument('-m', '--month', type=int, default=2, help="MM")
    args = parser.parse_args()

    run(args.year, args.month)

    