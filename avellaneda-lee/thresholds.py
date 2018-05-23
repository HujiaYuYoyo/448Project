import numpy as np
from scipy import integrate 
from scipy import optimize

kappa = 1
rho = 0.01
c = 0.01
theta = 1
sigma = 0.5

def fplus(u,rho,epsilon,kappa, theta, sigma):
    return u**(rho/kappa-1)*np.exp(-np.sqrt(2*kappa/sigma**2)*(theta-epsilon)*u-u**2/2)

def fplus_der(u,rho,epsilon,kappa, theta, sigma):
    return np.sqrt(2*kappa/sigma**2)*u**(rho/kappa)*np.exp(-np.sqrt(2*kappa/sigma**2)*(theta-epsilon)*u-u**2/2)

def fminus(u,rho,epsilon,kappa, theta, sigma):
    return u**(rho/kappa-1)*np.exp(np.sqrt(2*kappa/sigma**2)*(theta-epsilon)*u-u**2/2)

def fminus_der(u,rho,epsilon,kappa, theta, sigma):
    return -np.sqrt(2*kappa/sigma**2)*u**(rho/kappa)*np.exp(np.sqrt(2*kappa/sigma**2)*(theta-epsilon)*u-u**2/2)

def Fplus(epsilon,rho, kappa, theta, sigma):
    integral,error = integrate.quad(fplus,0, np.inf, args = (rho,epsilon,kappa,theta,sigma,))
    return integral

def Fminus(epsilon,rho, kappa, theta, sigma):
    integral,error = integrate.quad(fminus,0, np.inf, args = (rho,epsilon,kappa,theta,sigma,))
    return integral

def Fplus_der(epsilon,rho, kappa, theta, sigma):
    integral,error = integrate.quad(fplus_der,0, np.inf, args = (rho,epsilon,kappa,theta,sigma,))
    return integral

def Fminus_der(epsilon,rho, kappa, theta, sigma):
    integral,error = integrate.quad(fminus_der,0, np.inf, args = (rho,epsilon,kappa,theta,sigma,))
    return integral

def long_close_function(epsilon,rho, kappa, theta, sigma,c):
    return (epsilon[0] - c)*Fplus_der(epsilon[0],rho, kappa, theta, sigma)-Fplus(epsilon[0],rho, kappa, theta, sigma)

def long_close(rho, kappa, theta, sigma, c): #epsilon^*+
    return optimize.root(long_close_function,[1.8], args=(rho, kappa, theta, sigma, c,), method = 'lm').x[0]

def short_close_function(epsilon,rho, kappa, theta, sigma,c):
    return (epsilon[0] + c)*Fminus_der(epsilon[0],rho, kappa, theta, sigma)-Fminus(epsilon[0],rho, kappa, theta, sigma)

def short_close(rho, kappa, theta, sigma, c): #epsilon^*-
    return optimize.root(short_close_function,[-1], args=(rho, kappa, theta, sigma, c,), method = 'lm').x[0]

def Hplus(epsilon,kappa,theta, sigma,rho,c):
    epsilonplus = long_close(rho, kappa, theta, sigma,c)
    if epsilon >= epsilonplus:
        return epsilon - c
    else:
        return (epsilonplus - c)*Fplus(epsilon,rho, kappa, theta, sigma)/Fplus(epsilonplus,rho, kappa, theta, sigma)
    
def Hplus_der(epsilon,kappa,theta, sigma,rho,c):
    epsilonplus = long_close(rho, kappa, theta, sigma,c)
    if epsilon >= epsilonplus:
        return 1
    else:
        return (epsilonplus - c)*Fplus_der(epsilon,rho, kappa, theta, sigma)/Fplus(epsilonplus,rho, kappa, theta, sigma)

def Hminus(epsilon,kappa,theta, sigma,rho, c):
    epsilonminus = short_close(rho, kappa, theta, sigma,c)
    if epsilon <= epsilonminus:
        return -epsilon - c
    else:
        return -(epsilonminus + c)*Fminus(epsilon,rho, kappa, theta, sigma)/Fminus(epsilonminus,rho, kappa, theta, sigma)
    
    
def Hminus_der(epsilon,kappa,theta, sigma,rho, c):
    epsilonminus = short_close(rho, kappa, theta, sigma,c)
    if epsilon <= epsilonminus:
        return -1
    else:
        return -(epsilonminus + c)*Fminus_der(epsilon,rho, kappa, theta, sigma)/Fminus(epsilonminus,rho, kappa, theta, sigma)

def long_short_open_function(epsilon,rho,kappa,theta,sigma, c):
    minusepsilon = epsilon[0]
    plusepsilon = epsilon[1]
    
    numA = Fminus(minusepsilon,rho, kappa, theta, sigma)*(Hplus(plusepsilon,kappa,theta,sigma,rho,c)-plusepsilon-c)-Fminus(plusepsilon,rho, kappa, theta, sigma)*(Hminus(minusepsilon,kappa,theta,sigma,rho,c)+minusepsilon-c)
    denA = Fplus(plusepsilon,rho, kappa, theta, sigma)*Fminus(minusepsilon,rho, kappa, theta, sigma)-Fplus(minusepsilon,rho, kappa, theta, sigma)*Fminus(plusepsilon,rho, kappa, theta, sigma)
    A = numA/denA

    numB = Fplus(minusepsilon,rho, kappa, theta, sigma)*(Hplus(plusepsilon,kappa,theta,sigma,rho,c)-plusepsilon-c)-Fplus(plusepsilon,rho, kappa, theta, sigma)*(Hminus(minusepsilon,kappa,theta,sigma,rho,c)+minusepsilon-c)
    denB = Fminus(plusepsilon,rho, kappa, theta, sigma)*Fplus(minusepsilon,rho, kappa, theta, sigma)-Fminus(minusepsilon,rho, kappa, theta, sigma)*Fplus(plusepsilon,rho, kappa, theta, sigma)
    B = numB/denB
    
    y_0 = A*Fplus_der(plusepsilon,rho, kappa, theta, sigma)+B*Fminus_der(plusepsilon,rho, kappa, theta, sigma)+1-Hplus_der(plusepsilon,kappa,theta, sigma,rho, c)
    y_1 = A*Fplus_der(minusepsilon,rho, kappa, theta, sigma)+B*Fminus_der(minusepsilon,rho, kappa, theta, sigma)-1-Hminus_der(minusepsilon,kappa,theta, sigma,rho, c)
    
    return [y_0,y_1]

def long_short_open(rho,kappa,theta,sigma, c):
    return optimize.root(long_short_open_function,[-0.18,1.5 ], args=(rho, kappa, theta, sigma, c,), method = 'hybr').x

def long_open(rho,kappa,theta,sigma, c):
    return short_close(rho,kappa,theta,sigma, c)

def short_open(rho,kappa,theta,sigma, c):
    return long_close(rho,kappa,theta,sigma, c)