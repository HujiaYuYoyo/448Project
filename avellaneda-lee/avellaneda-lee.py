import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARMA

#Parameters
dt = 1#/252
long_open = -1.25
long_close = -0.50
short_open = 1.25;
short_close = 0.75
risk_free = 0
tran_cost = 0.0005
leverage = 1
training_size = 100

#returns1 = np.array([0.0,0.1,-0.1,0,0,0.2])
#returns2 = 2*returns1


def standardized_returns(midprices):
    log_midprices = np.log(midprices)
    logreturns = np.diff(log_midprices)
    return (logreturns-np.mean(logreturns))/np.sd(logreturns)

def regress(returns1,returns2):
    x = returns1.reshape(-1,1)
    y = returns2.reshape(-1,1)
    model = LinearRegression()
    model.fit(x,y)
    a = model.intercept_[0]
    b = model.coef_[0,0]
    residuals = y-model.predict(x)
    return residuals, a,b

def fitOU(residual):
    ou = np.cumsum(residual)
    model = ARMA(ou, order=(1, 0)) 
    fittedmodel = model.fit(disp=-1)  
    a = fittedmodel.params[0]
    b = fittedmodel.params[1]
    var =  fittedmodel.sigma2
    kappa = -np.log(b)/dt
    m = a/(1-np.exp(-kappa*dt))
    sigma = np.sqrt(var*2*kappa/(1-np.exp(-2*kappa*dt)))
    sigmaeq = np.sqrt(var/(1-np.exp(-2*kappa*dt)));
    return kappa, m, sigma, sigmaeq

def sscore(m, sigmaeq):
    if sigmaeq != 0:
        return -m/sigmaeq
    elif m>0:
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
    list_logreturns = np.diff(log_wealth)
    
    plt.plot(range(n-1),list_logreturns, c='blue')
    plt.title('Evolution of the log-returns')
    plt.xlabel('Seconds')
    plt.show()
    
    plt.hist(list_logreturns, bins='auto')
    plt.title('Distribution of the log-returns')
    plt.show() 
    
    #Maybe do montecarlo and compute VaR = np.percentile(montecarlo_logreturns,5)
    
    sharpe = np.mean(list_logreturns)/np.sd(list_logreturns)
    print('The Sharpe ratio is:', sharpe)
    
    cum_return = (wealth[n-1]-wealth[0])/wealth[0]
    print('The total cumulative return is:', cum_return)
    return

def plots(scores, long_open, long_close, short_open, short_close, n, training_size):
    times = range(n-training_size+1)
    plt.plot(times, scores, c='blue')
    plt.plot(times, long_open*np.ones(n-training_size+1), c='green', label='long_open')
    plt.plot(times, long_close*np.ones(n-training_size+1), c='red', label='long_close')
    plt.plot(times, short_open*np.ones(n-training_size+1), c='olive', label='short_open')
    plt.plot(times, short_close*np.ones(n-training_size+1), c='brown', label='short_close')
    plt.title('Evolution of the s-score')
    plt.xlabel('Seconds')
    plt.ylabel('S-score')
    plt.legend()
    plt.show()
    return


def main():
    #Get data
    #get 1 second midprices, sellprice, buyprice of a (good) pair of stocks during all period
    #n= len(midprices)
    position = np.zeros((n-training_size+1,2))
    wealth = [0]   
    scores = []
    
    for t in range(n-training_size):
        #Preprocess data in the training period
        returns1 = standardized_returns(midprices1[t:training_size+t])
        returns2 = standardized_returns(midprices2[t:training_size+t])
        residuals, a,b = regress(returns1,returns2)
    
        #Calibrate model in the training period
        kappa, m, sigma, sigmaeq = fitOU(residual)
        #print("The mean reversion time is", 1/kappa)
        #print("Is the reversion time short enough?", kappa > 1/(2*dt*training_size))
        s = sscore(m,sigmaeq)
        scores += [s]
        
        #Execute trading 
        increment = 0
        if position[t,0] == 0:
            if s < -long_open:
                position[t+1,0] = leverage
                position[t+1,1] = -leverage * b
                increment = leverage*(-buyprice[training_size+t+1,0]+b*sellprice[training_size+t+1,1])
            elif s > short_open:
                position[t+1,0] = - leverage
                position[t+1,1] = leverage * b
                increment = leverage*(sellprice[training_size+t+1,0]-b*buyprice[training_size+t+1,1])
        elif position[t,0] > 0 and s > -short_close:
            position[t+1,:] = np.zeros(2)
            increment = leverage*(sellprice[training_size+t+1,0]+b*buyprice[training_size+t+1,1])
        elif position[t,0] < 0 and s < long_close:
            position[t+1,:] = np.zeros(2)
            increment = leverage*(-buyprice[training_size+t+1,0]+b*sellprice[training_size+t+1,1])
            
        #Compute change in wealth    
        wealth += [-tran_cost*abs(position[t+1,0]-position[t,0])+increment]
        
        #Metrics and plots
        metrics(wealth)
        plots(scores, long_open, long_close, short_open, short_close, n, training_size)        
    
    return

if __name__ == "__main__":
    main()

