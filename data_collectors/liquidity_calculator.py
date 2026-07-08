import pandas as pd

liquidity_db = pd.read_csv("C:\\Users\\zmbur\\PycharmProjects\\contextAI\\support_files\\liquidity_db.csv")


def get_ticker_dict(ticker):
    ticker_data = liquidity_db.loc[ticker.upper()]
    ticker_dict = {
        # 'is_etf': ticker_data['is_etf'],
        'adtv': ticker_data['adtv'],
        # 'spread': ticker_data['spread'],
        'lqa_score': ticker_data['lqa_score']
    }
    return ticker_dict


def calculate_liquidity(adtv, lqa_score):
    """
    Function to calculate own liquidity score
    :param adtv: average daily volume for the ticker
    :param lqa_score: lqa_score for the ticker provided by Bloomberg
    :return: final liquidity_score from 0:100
    """
    high_adtv_threshold = 5_000_000  # 295 of the most liquid stocks - ends at TJX / SNOW
    # threshold defines pivot where we begin tailoring down size based on liquidity
    adtv_weight = 0.6
    lqa_weight = 0.4

    normalized_adtv = min(adtv / high_adtv_threshold, 1)
    normalized_lqa = lqa_score / 100

    score = (normalized_adtv * adtv_weight) + (normalized_lqa * lqa_weight)

    final_lqa_score = round(score * 100, 0)
    if final_lqa_score < 0:
        final_lqa_score = 0
    return final_lqa_score


def get_liquidity(ticker):
    ticker_dict = get_ticker_dict(ticker)
    lqa_score = calculate_liquidity(**ticker_dict)
    return lqa_score

# if __name__ == '__main__':
#     for index, row in liquidity_db.iterrows():
#         ticker = index
#         lqa = get_liquidity(**get_ticker_dict(ticker))
#         liquidity_db.at[index,'final_lqa'] = lqa
#     liquidity_db.to_csv('C:\\Users\\zmbur\\PycharmProjects\\orderPipe\\support_files\\liquidity_db_add.csv')
