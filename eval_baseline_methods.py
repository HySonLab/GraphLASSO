import argparse
import numpy as np
import pandas as pd
import os
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
#import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#print(os.getcwd())
from lib import utils
from lib.metrics import masked_rmse_np, masked_mape_np, masked_mae_np
from lib.utils import StandardScaler


def historical_average_predict(df, period=12 * 24 * 7, test_ratio=0.3, null_val=0.):
    """
    Calculates the historical average of sensor reading.
    :param df:
    :param period: default 1 week.
    :param test_ratio:
    :param null_val: default 0.
    :return:
    """
    n_sample, n_sensor = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    y_test = df[-n_test:]
    y_predict = pd.DataFrame.copy(y_test)

    for i in range(n_train, min(n_sample, n_train + period)):
        inds = [j for j in range(i % period, n_train, period)]
        historical = df.iloc[inds, :]
        y_predict.iloc[i - n_train, :] = historical[historical != null_val].mean()
    # Copy each period.
    for i in range(n_train + period, n_sample, period):
        size = min(period, n_sample - i)
        start = i - n_train
        y_predict.iloc[start:start + size, :] = y_predict.iloc[start - period: start + size - period, :].values
    return y_predict, y_test


def static_predict(df, n_forward, test_ratio=0.3):
    """
    Assumes $x^{t+1} = x^{t}$
    :param df:
    :param n_forward:
    :param test_ratio:
    :return:
    """
    test_num = int(round(df.shape[0] * test_ratio))
    y_test = df[-test_num:]
    y_predict = df.shift(n_forward).iloc[-test_num:]
    return y_predict, y_test


def predict(model,df, n_forwards=(1, 3), n_lags=4, n_noise=3, test_ratio=0.3):
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    :param df: pandas.DataFrame, index: time, columns: sensor id, content: data.
    :param n_forwards: a tuple of horizons.
    :param n_lags: the order of the VAR model.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    n_sample, n_output = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    df_train, df_test = df[:n_train], df[n_train:]

    print(df.shape)
    print("mean", df_test.values.mean(), "std: ", df_test.values.std())
    print("mean", df_train.values.mean(), "std: ", df_train.values.std())

    scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
    data = scaler.transform(df_train.values)
    #data=df_train.copy()
    if model=="var":
        var_model = VAR(data)
        var_result = var_model.fit(n_lags)
    elif model=="varma":
        var_model=VARMAX(data,order=(n_lags,n_noise), trend ='t')
        var_result = var_model.fit()
    
    max_n_forwards = np.max(n_forwards)
    # Do forecasting.
    result = np.zeros(shape=(len(n_forwards), n_test, n_output))
    start = n_train - n_lags - max_n_forwards + 1
    for input_ind in range(start, n_sample - n_lags):
        prediction = var_result.forecast(scaler.transform(df.values[input_ind: input_ind + n_lags]), max_n_forwards)
        #prediction[prediction<0]=0
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]

    df_predicts = []
    for i, n_forward in enumerate(n_forwards):
        df_predict = pd.DataFrame(scaler.inverse_transform(result[i]), index=df_test.index, columns=df_test.columns)
        #df_predict[df_predict < 0] = 0
        df_predicts.append(df_predict)
    
    return df_predicts, df_test


# def eval_static(traffic_reading_df):
#     logger.info('Static')
#     horizons = [1, 3, 6, 12]
#     logger.info('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
#     for horizon in horizons:
#         y_predict, y_test = static_predict(traffic_reading_df, n_forward=horizon, test_ratio=0.2)
#         rmse = masked_rmse_np(preds=y_predict.values, labels=y_test, null_val=0)
#         mape = masked_mape_np(preds=y_predict.values, labels=y_test.values, null_val=0)
#         mae = masked_mae_np(preds=y_predict.values, labels=y_test.values, null_val=0)
#         line = 'Static\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
#         logger.info(line)


# def eval_historical_average(traffic_reading_df, period):
#     y_predict, y_test = historical_average_predict(traffic_reading_df, period=period, test_ratio=0.2)
#     rmse = masked_rmse_np(preds=y_predict.values, labels=y_test.values, null_val=0)
#     mape = masked_mape_np(preds=y_predict.values, labels=y_test.values, null_val=0)
#     mae = masked_mae_np(preds=y_predict.values, labels=y_test.values, null_val=0)
#     logger.info('Historical Average')
#     logger.info('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
#     for horizon in [1, 3, 6, 12]:
#         line = 'HA\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
#         logger.info(line)


def eval_var(model,traffic_reading_df, n_lags=7, n_noise=3,n_forwards=[1, 3, 6, 12]):
    
    y_predicts, y_test = predict(model,traffic_reading_df, n_forwards=n_forwards, n_lags=n_lags,n_noise=n_noise,
                                     test_ratio=0.2)
    logger.info('VAR (lag=%d)' % n_lags)
    logger.info('Model\tHorizon\tRMSE\tMAPE\tMAE')
    for i, horizon in enumerate(n_forwards):
        rmse = masked_rmse_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        mape = masked_mape_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        mae = masked_mae_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        line = 'VAR\t%d\t%.4f\t%.4f\t%.4f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def main(args):
    if args.reading_filename[-4:]=='.csv':
        reading_df = pd.read_csv(args.reading_filename)
        
    elif args.reading_filename[-3:]=='.h5':
        reading_df = pd.read_hdf(args.reading_filename)
    elif args.reading_filename[-4:] =='.txt':
        reading_df = pd.read_csv(args.reading_filename, header=None) 
        reading_df=reading_df.iloc[:,:100]
    #print(traffic_reading_df.mean(),traffic_reading_df.std())
    #eval_static(traffic_reading_df)
    #eval_historical_average(traffic_reading_df, period=7 * 24 * 12)
    if args.n_forwards =="weekly":
        n_fw=[1,3,7,14]
    elif  args.n_forwards =="daily":
        n_fw=[1,3,6,12]
    model="var"
    if model=="varma":
        dif=reading_df.diff()
        reading_df = dif.iloc[1:,:]
    print("dataset: ", args.reading_filename)
    eval_var(model,reading_df, n_lags=args.n_lags, n_noise=2,n_forwards=n_fw)

    y_pred_ha, y_test_ha  = historical_average_predict(reading_df,period = args.n_lags)

    print(masked_mae_np(y_pred_ha, y_test_ha))


if __name__ == '__main__':
    logger = utils.get_logger('data/model', 'Baseline')
    parser = argparse.ArgumentParser()
    parser.add_argument('--reading_filename', default="data/CA1_Food1.csv", type=str,
                        help='Path to the Dataframe.')
    parser.add_argument('--n_forwards',default="daily", type=str, help="Number to forecast") # "daily" is minutes, "weekly" is days
    parser.add_argument('--n_lags',default=3, type=int, help="N_lags")
    parser.add_argument('--standarlize',default=True, type=bool, help="Normalization")
    args = parser.parse_args()
    main(args)
