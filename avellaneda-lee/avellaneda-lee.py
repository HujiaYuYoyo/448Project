import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import AR

# Parameters
dt = 1  #training interval, in seconds
training_size = 100

long_open = -1.25 #when to buy/sell
long_close = -0.50
short_open = 1.25;
short_close = 0.75

risk_free = 0 #exogeneous parameters
tran_cost = 0.0005


leverage = 1 #risk parameters
initial_wealth = 1000
limit_wealth = 100


def standardized_returns(midprices):
    log_midprices = np.log(midprices)
    logreturns = np.diff(log_midprices)
    if np.std(logreturns) > 0.0000001:
        return (logreturns - np.mean(logreturns)) / np.std(logreturns)
    else:
        return np.zeros((logreturns.shape))


def regress(returns1, returns2):
    x = returns1.reshape(-1, 1)
    y = returns2.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    a = model.intercept_[0]
    b = model.coef_[0, 0]
    residuals = y - model.predict(x)
    return residuals, a, b


def fitOU(residual):
    training_size = len(residual)
    ou = np.cumsum(residual)
    model = AR(ou)
    fittedmodel = model.fit(maxlag=1,disp=-1)
    a = fittedmodel.params[0]
    b = fittedmodel.params[1]
    var = fittedmodel.sigma2 #b should be in (0,1) so the mean reversion time (1/kappa) is >0
    #The reversion time is short enough if kappa > 2/(dt*training_size), i.e, at least half the training-period
    #This means  0 < b < np.exp(-2/training_size)

    if b > 0 and b < np.exp(-2/training_size):
        kappa = -np.log(b) / dt    
        m = a / (1 - np.exp(-kappa * dt))
        sigma = np.sqrt(var * 2 * kappa / (1 - np.exp(-2 * kappa * dt)))
        sigmaeq = np.sqrt(var / (1 - np.exp(-2 * kappa * dt)));
        return kappa, m, sigma, sigmaeq

    else:
        return -1,0,0,0


def sscore(m, sigmaeq):
    if sigmaeq != 0:
        return -m / sigmaeq
    elif m > 0:
        return 10000000
    else:
        return -10000000


def metrics(wealth):
    n = len(wealth)
    times = range(n)
    plt.plot(times, wealth, c='blue')
    plt.title('Evolution of the wealth')
    plt.xlabel('Seconds')
    plt.ylabel('Dollars')
    plt.show()

    log_wealth = np.log(wealth)
    list_logreturns = np.diff(log_wealth, axis=0)

    print(np.max(log_wealth))
    print(np.max(list_logreturns))

    plt.plot(range(n - 1), list_logreturns, c='blue')
    plt.title('Evolution of the log-returns')
    plt.xlabel('Seconds')
    plt.show()

    plt.hist(list_logreturns, bins='auto')
    plt.title('Distribution of the log-returns')
    plt.show()

    # Maybe do montecarlo and compute VaR = np.percentile(montecarlo_logreturns,5)

    sharpe = np.mean(list_logreturns) / np.std(list_logreturns)*np.sqrt(6*60*60) #6 hours of trading second-by-second
    print('The (daily) Sharpe ratio is:', sharpe)

    cum_return = (wealth[n - 1] - wealth[0]) / wealth[0]
    print('The total cumulative return is:', cum_return)
    return


def plots(scores, long_open, long_close, short_open, short_close):
    n = len(scores)
    times = range(n)
    plt.plot(times, scores, c='blue')
    plt.plot(times, np.full(n,long_open), c='green', label='long_open')
    plt.plot(times, np.full(n,long_close), c='red', label='long_close')
    plt.plot(times, np.full(n,short_open), c='olive', label='short_open')
    plt.plot(times, np.full(n,short_close), c='brown', label='short_close')
    plt.title('Evolution of the s-score')
    plt.xlabel('Seconds')
    plt.ylabel('S-score')
    plt.legend()
    plt.show()
    return


def tickers_to_dict(tickers):
    '''Function that take in a list of tickers and returns a dictionare with the prices (bid, ask, midprices), returns
    and normalized returns of that ticker.
    Assumes that there is a csv in the same folder named "ticker"-1.csv'''
    stocks = {}
    for ticker in tickers:
        stocks[ticker] = pd.read_csv('{}-1.csv'.format(ticker), parse_dates=True, index_col=0)
        stocks[ticker]['midprice'] = midprice(stocks[ticker]['direct.bid1'], stocks[ticker]['direct.ask1'])
    return stocks


def midprice(bid, ask):
    """ function to calculate midprice using the best bid and ask prices and no volume weighting """
    midprice = np.add(bid, ask) / 2.0
    return midprice


def execute(tickers):
    # get data of the pair of tickers during all period 

    stocks = tickers_to_dict(tickers)

    midprices1 = stocks[tickers[0]]['midprice']
    midprices2 = stocks[tickers[1]]['midprice']
    sellprice = np.column_stack((stocks[tickers[0]]['direct.bid1'], stocks[tickers[1]]['direct.bid1']))  # sell prices
    buyprice = np.column_stack((stocks[tickers[0]]['direct.ask1'], stocks[tickers[1]]['direct.ask1']))  # buy prices

    #initialize vectors storing the evolution of the strategy
    n = len(midprices1)
    position = np.zeros((n - training_size + 1, 2))
    wealth = np.full((n - training_size + 1, 1), initial_wealth)
    scores = [0]

    for t in range(n - training_size):
        if wealth[t] > limit_wealth:
            # Preprocess data in the training period
            returns1 = standardized_returns(midprices1[t:t + training_size])
            returns2 = standardized_returns(midprices2[t:t + training_size])

            residuals, a, b = regress(returns1, returns2)

            # Calibrate model in the training period
            kappa, m, sigma, sigmaeq = fitOU(residuals)
            increment = 0
            
        
            if kappa > 0: #sufficiently fast and non-degenerate mean-reversion, checked in the fitOU function
                s = sscore(m, sigmaeq)

                # Execute trading
                if position[t, 0] == 0:
                    if s < -long_open:
                        position[t + 1, 0] = leverage
                        position[t + 1, 1] = -leverage * np.ceil(b) #we round b to order an integer amount of stocks
                        increment = leverage * (-buyprice[training_size + t, 0] + np.ceil(b) * sellprice[training_size + t, 1])
                    elif s > short_open:
                        position[t + 1, 0] = - leverage
                        position[t + 1, 1] = leverage * np.ceil(b)
                        increment = leverage * (sellprice[training_size + t, 0] - np.ceil(b) * buyprice[training_size + t, 1])
                elif position[t, 0] > 0 and s > -short_close:
                    position[t + 1, :] = np.zeros(2)
                    increment = leverage * (sellprice[training_size + t, 0] + np.ceil(b) * buyprice[training_size + t, 1])
                elif position[t, 0] < 0 and s < long_close:
                    position[t + 1, :] = np.zeros(2)
                    increment = leverage * (-buyprice[training_size + t, 0] + np.ceil(b) * sellprice[training_size + t, 1])
                else:
                    position[t + 1, :] = position[t, :]  

            else:
                position[t + 1, :] = position[t, :]
                s = scores[t]

        else:
            position[t + 1, :] = position[t, :]
            s = scores[t]
            increment = 0

        # Compute change in wealth and score
        scores += [s]
        wealth[t+1] = wealth[t] - tran_cost * abs(position[t + 1, 0] - position[t, 0]) + increment

    print(wealth)
    print(position)

    # Metrics and plots
    metrics(wealth)
    plots(scores, long_open, long_close, short_open, short_close)

    return


tickers = ['csco', 'aapl']  # define the pair here
execute(tickers)

#if __name__ == "__main__":
#    main()

