from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None, csv= True, has_index=False
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    if has_index:
        df=df.iloc[:,1:]

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
   
    data_list = [data]
    
 

    if csv:
        print("shape:", data.shape)
        # time_ind=list(df.iloc[:,0].apply(lambda x: int(x[2:])))
        # time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        # print(time_in_day.shape)
        # data_list.append(time_in_day)
    else:
        if add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if add_day_in_week:
            day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
            day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
            data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    

    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    # print(x.shape)
    # print(x_t[0,...])
    # print(y_t[0,...])

    return x, y


def generate_train_val_test(args):
    if args.df_filename[-3:] =='.h5':
        df = pd.read_hdf(args.df_filename)
    elif args.df_filename[-4:] =='.csv':
        df = pd.read_csv(args.df_filename)

    if "dept_CA1.csv" in args.df_filename:
        index=True
    else:
        index=False
    # 0 is the latest observed sample.
    input_length=args.input_len
    output_length=args.output_len

    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        # here to change the input and output length
        np.concatenate((np.arange(-input_length+1, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1,output_length+1, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    len_data = df.shape[0]
    t_test = round(len_data * 0.9)
    t_train = round(len_data * 0.7)
    train =df.iloc[:t_train,:]
    val =df.iloc[t_train: t_test,:]
    test=df.iloc[t_test: ,:]

    x_train, y_train = generate_graph_seq2seq_io_data(train,x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
        has_index=index)
    x_val, y_val = generate_graph_seq2seq_io_data(val,x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
        has_index=index)
    x_test, y_test = generate_graph_seq2seq_io_data(test,x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
        has_index=index)


    # x, y = generate_graph_seq2seq_io_data(
    #     df,
    #     x_offsets=x_offsets,
    #     y_offsets=y_offsets,
    #     add_time_in_day=True,
    #     add_day_in_week=False,
    #     has_index=index
    # )

    # print("x shape: ", x.shape, ", y shape: ", y.shape)
    # # Write the data into npz file.
    # # num_test = 6831, using the last 6831 examples as testing.
    # # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    
    # # train
    # x_train, y_train = x[:num_train], y[:num_train]
    # # val
    # x_val, y_val = (
    #     x[num_train: num_train + num_val],
    #     y[num_train: num_train + num_val],
    # )
    # # test
    # x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/CA1_Food1", help="Output directory."
    )
    parser.add_argument(
        "--df_filename",
        type=str,
        default="data/CA1_Food1.csv",
        help="Raw traffic readings.",
    )
    parser.add_argument(
        "--output_len", type=int, default= 7, help="Output len."
    )
    parser.add_argument(
        "--input_len", type=int, default= 14, help="Input len."
    )
    args = parser.parse_args()
    main(args)
