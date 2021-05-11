from pmdarima.arima import auto_arima
import pandas as pd
import datetime

# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):
    import pandas as pd

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return


if __name__ == "__main__":
    args = config()

    df_generation = pd.read_csv(args.generation)
    arima_generation = auto_arima(df_generation["generation"], seasonal=True)
    pred_generation = arima_generation.predict(n_periods=24)

    df_consumption = pd.read_csv(args.consumption)
    arima_consumption = auto_arima(df_consumption["consumption"], seasonal=True)
    pred_consumption = arima_consumption.predict(n_periods=24)

    # print(df_generation["time"].iloc[-1])
    
    data = []
    for i in range(24):
        today = datetime.datetime.strptime(df_generation["time"].iloc[-24 + i], "%Y-%m-%d %H:%M:%S")
        nextDay = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        print(nextDay)
        result = round(pred_generation[i] - pred_consumption[i], 2)
        print(result)
        if result < 0:
            data.append([nextDay, "buy", 2.5, (result * -1)])
        else:
            data.append([nextDay, "sell", 1.5, result])
    output(args.output, data)
