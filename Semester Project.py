# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:20:25 2021

@author: nicol
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
import pylab as pp
from scipy import integrate, interpolate
from scipy import optimize
from scipy.optimize import minimize
from scipy.integrate import odeint

###########################
##formatting of data set###
########################### 

sars=pd.read_excel(r"C:\Users\nicol\Desktop\Uni Module\FS 21\BIO 394\Semester Project\Dataset\sars_2003_complete_dataset_clean2.xlsx")
#sars
#sars.info()
SarsChina=pd.read_excel(r'C:\Users\nicol\Desktop\Uni Module\FS 21\BIO 394\Semester Project\Dataset\Sars2003_China.xlsx')
SarsHK=pd.read_excel(r'C:\Users\nicol\Desktop\Uni Module\FS 21\BIO 394\Semester Project\Dataset\Sars2003_HongKong.xlsx')
SarsSingapore=pd.read_excel(r'C:\Users\nicol\Desktop\Uni Module\FS 21\BIO 394\Semester Project\Dataset\Sars2003_Singapore.xlsx')
SarsTaiwan=pd.read_excel(r'C:\Users\nicol\Desktop\Uni Module\FS 21\BIO 394\Semester Project\Dataset\Sars2003_Taiwan.xlsx')
SarsVietnam=pd.read_excel(r'C:\Users\nicol\Desktop\Uni Module\FS 21\BIO 394\Semester Project\Dataset\Sars2003_Vietnam.xlsx')

#SarsChina.plot(kind='line', x='Date', y='Cumulative number of case(s)')
#SarsHK.plot(kind='line', x='Date', y='Cumulative number of case(s)')
#SarsSingapore.plot(kind='line', x='Date', y='Cumulative number of case(s)')
#SarsTaiwan.plot(kind='line', x='Date', y='Cumulative number of case(s)')
#SarsVietnam.plot(kind='line', x='Date', y='Cumulative number of case(s)')

SarsHK['Date'] = pd.to_datetime(SarsHK['Date']).dt.date
SarsChina['Date'] = pd.to_datetime(SarsChina['Date']).dt.date
SarsSingapore['Date'] = pd.to_datetime(SarsSingapore['Date']).dt.date
SarsTaiwan['Date'] = pd.to_datetime(SarsTaiwan['Date']).dt.date
SarsVietnam['Date'] = pd.to_datetime(SarsVietnam['Date']).dt.date
#print(SarsHK['Date'][1])
#SarsChina=SarsChina[:-15]

sars['Date'] = pd.to_datetime(sars['Date']).dt.date
Date = []
SerNumb=[]
for i in sars["Date"]:
#    print(i)
    if i not in Date:
        Date.append(i)


for i in sars["Number"]:
#    print(i)
    if i not in SerNumb:
        SerNumb.append(i)
#print(SerNumb)

colum_names = [ 'Serial Number', 'Date', 'Cases China', 'Cases HK', 'Cases Singapore', 
               'Cases Taiwan', 'Cases Vietnam', 'Death China', 'Death HK', 'Death Singapore',
               'Death Taiwan', 'Death Vietnam', 'Recovered China', 'Recovered HK','Recovered Singapore',
               'Recovered Taiwan', 'Recovered Vietnam','Daily China', 'Daily HK', 'Daily Singapore',
               'Daily Taiwan', 'Daily Vietnam']
CumCas = pd.DataFrame(columns = colum_names, index = range(0,96))

#add series number to data frame
for index,rows in CumCas.iterrows():
#    print(index)
    CumCas['Serial Number'][index] = SerNumb[index]
#    for j in SerNumb:
#        print(index)
#        print(j)
#        if j == index:
#            print(CumCas[index])
#        print(CumCas['Serial Number'][index])

#        if index == SarsChina['Date'][j]:
#            CumCas['Serial Number'][index] = SarsChina['Number'][j]
            
#add date to data frame
count= 0
for index, rows in CumCas.iterrows():
    CumCas['Date'][index]=Date[count]
    count+=1
###############################         
#add case numbers to data grame  
###############################
#China
for index,rows in CumCas.iterrows():
    for num in range(len(SarsChina['Number'])):
#        print(num)
        if SarsChina['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Cases China'][index]= SarsChina['Cumulative number of case(s)'][num]
#HK
for index,rows in CumCas.iterrows():
    for num in range(len(SarsHK['Number'])):
#        print(num)
        if SarsHK['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Cases HK'][index]= SarsHK['Cumulative number of case(s)'][num]
#Taiwan
for index,rows in CumCas.iterrows():
    for num in range(len(SarsTaiwan['Number'])):
#        print(num)
        if SarsTaiwan['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Cases Taiwan'][index]= SarsTaiwan['Cumulative number of case(s)'][num]
#Vietnam
for index,rows in CumCas.iterrows():
    for num in range(len(SarsVietnam['Number'])):
#        print(num)
        if SarsVietnam['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Cases Vietnam'][index]= SarsVietnam['Cumulative number of case(s)'][num]        
#Singapore
for index,rows in CumCas.iterrows():
    for num in range(len(SarsSingapore['Number'])):
#        print(num)
        if SarsSingapore['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Cases Singapore'][index]= SarsSingapore['Cumulative number of case(s)'][num]    
########################
#add death to data frame   
########################
#China
for index,rows in CumCas.iterrows():
    for num in range(len(SarsChina['Number'])):
#        print(num)
        if SarsChina['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Death China'][index]= SarsChina['Number of deaths'][num]
#HK
for index,rows in CumCas.iterrows():
    for num in range(len(SarsHK['Number'])):
#        print(num)
        if SarsHK['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Death HK'][index]= SarsHK['Number of deaths'][num]
#Taiwan
for index,rows in CumCas.iterrows():
    for num in range(len(SarsTaiwan['Number'])):
#        print(num)
        if SarsTaiwan['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Death Taiwan'][index]= SarsTaiwan['Number of deaths'][num]
#Vietnam
for index,rows in CumCas.iterrows():
    for num in range(len(SarsVietnam['Number'])):
#        print(num)
        if SarsVietnam['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Death Vietnam'][index]= SarsVietnam['Number of deaths'][num]        
#Singapore
for index,rows in CumCas.iterrows():
    for num in range(len(SarsSingapore['Number'])):
#        print(num)
        if SarsSingapore['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Death Singapore'][index]= SarsSingapore['Number of deaths'][num]   
            
########################
#add recovered to data frame   
########################
#China
for index,rows in CumCas.iterrows():
    for num in range(len(SarsChina['Number'])):
#        print(num)
        if SarsChina['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Recovered China'][index]= SarsChina['Number recovered'][num]
#HK
for index,rows in CumCas.iterrows():
    for num in range(len(SarsHK['Number'])):
#        print(num)
        if SarsHK['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Recovered HK'][index]= SarsHK['Number recovered'][num]
#Taiwan
for index,rows in CumCas.iterrows():
    for num in range(len(SarsTaiwan['Number'])):
#        print(num)
        if SarsTaiwan['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Recovered Taiwan'][index]= SarsTaiwan['Number recovered'][num]
#Vietnam
for index,rows in CumCas.iterrows():
    for num in range(len(SarsVietnam['Number'])):
#        print(num)
        if SarsVietnam['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Recovered Vietnam'][index]= SarsVietnam['Number recovered'][num]        
#Singapore
for index,rows in CumCas.iterrows():
    for num in range(len(SarsSingapore['Number'])):
#        print(num)
        if SarsSingapore['Number'][num] == CumCas["Serial Number"][index]:
            CumCas['Recovered Singapore'][index]= SarsSingapore['Number recovered'][num]   
            
#Plot     
#CumCas.plot(x='Date', y=['Cases China','Cases HK','Cases Singapore','Cases Vietnam','Cases Taiwan'], kind = 'line')
#plt.title('Cumulative Cases')
#plt.xlabel('Date')
#plt.ylabel('Number of Cases')
#
##China Plot
#CumCas.plot(x='Date', y=['Cases China','Death China','Recovered China'])
#plt.title('Outbreak China')
#plt.xlabel('Date')
#plt.ylabel('Number of People')

#################
###daily cases###
#################

#China
for index,rows in CumCas.iterrows():
#    print(index)
#    print(CumCas['Cases China'][index])
    if index == 0:
        pass
    else:
         CumCas['Daily China'][index] = CumCas['Cases China'][index]-CumCas['Cases China'][index-1]
#    if res<0:
#        CumCas['Daily China'][index] = 0
#    else:
#         CumCas['Daily China'][index] = res

#HK
for index,rows in CumCas.iterrows():
#    print(index)
#    print(CumCas['Cases China'][index])
    if index == 0:
        pass
    else:
         CumCas['Daily HK'][index] = CumCas['Cases HK'][index]-CumCas['Cases HK'][index-1]

#Taiwan
for index,rows in CumCas.iterrows():
#    print(index)
#    print(CumCas['Cases China'][index])
    if index == 0:
        pass
    else:
         CumCas['Daily Taiwan'][index] = CumCas['Cases Taiwan'][index]-CumCas['Cases Taiwan'][index-1]


#Vietnam
for index,rows in CumCas.iterrows():
#    print(index)
#    print(CumCas['Cases China'][index])
    if index == 0:
        pass
    else:
         CumCas['Daily Vietnam'][index] = CumCas['Cases Vietnam'][index]-CumCas['Cases Vietnam'][index-1]
   

#Singapore
for index,rows in CumCas.iterrows():
#    print(index)
#    print(CumCas['Cases China'][index])
    if index == 0:
        pass
    else:
         CumCas['Daily Singapore'][index] = CumCas['Cases Singapore'][index]-CumCas['Cases Singapore'][index-1]

    
#Daily Cases Plot
CumCas.plot(x='Date', y=['Daily China','Daily HK', 'Daily Singapore', 'Daily Taiwan','Daily Vietnam'])
plt.title('Cases Per Day')
plt.xlabel('Date')
plt.ylabel('Cases')

#i copied the CumCas data set into an excel file so I have the completed-------
#data frame at hand

compset = pd.read_excel(r"C:\Users\nicol\Desktop\Uni Module\FS 21\BIO 394\Semester Project\Dataset\Working_Dataset.xlsx")
compset
compset = compset.replace(np.nan, 0)
num = compset._get_numeric_data()
num[num < 0]=0

import numpy as np

##initialize the data----------------------------------------------------------
x_data = np.array(range(0,96), dtype=float)
y_data = np.array(compset['Daily Singapore'], dtype=float)

#Functions---------------------------------------------------------------------

def f_ODE(y, x, p): 
    """define the ODE system in terms of 
        dependent variable y,
        independent variable t, and
        optinal parmaeters, in this case a single variable k """
    r"""S,I,R, p[0] = beta, p[1] = gamma, y[0] = S, y[1] = I,"""
    return (-p[0] * y[0] * y[1], p[0]*y[0]*y[1]-p[1]*y[1], p[1]*y[1])     
def my_ls_func(x,teta):
    """definition of function for LS fit
        x gives evaluation points,
        teta is an array of parameters to be varied for fit"""
    # create an alias to f which passes the optional params    
    f2 = lambda y,t: f_ODE(y, t, teta)
    # calculate ode solution, retuen values for each entry of "x"
    r = integrate.odeint(f2,y0,x)
    #in this case, we only need one of the dependent variable values
    return r[:,1]
def f_resid(p):
    """ function to pass to optimize.leastsq
        The routine will square and sum the values returned by 
        this function""" 
    return my_ls_func(x_data,p)-y_data
#solve the system - the solution is in variable c------------------------------
p_guess = [0.000001, 0.01]  #initial guess for params
y0 = [1000000 ,1,0] #inital conditions for ODEs
(c,kvg) = optimize.leastsq(f_resid, p_guess) #get params
print ("parameter values are ",c)

#calculate R0------------------------------------------------------------------
R0 = (c[0]*y0[0])/c[1]
print("The reproduction number was", R0)

#evaluate goodness of fit------------------------------------------------------
p, cov, infodict, mesg, ier = optimize.leastsq(f_resid, p_guess, full_output = True)
y =  np.array(compset['Daily Singapore'], dtype=float)
ss_err = (infodict['fvec']**2).sum()
ss_tot = ((y-y.mean())**2).sum()
rsquared=1-(ss_err/ss_tot)
print(rsquared)

#Plot--------------------------------------------------------------------------
# fit ODE results to interpolating spline just for fun
xeval=np.linspace(min(x_data), max(x_data),30) 
gls = interpolate.UnivariateSpline(xeval, my_ls_func(xeval,c), k=3, s=0)

#pick a few more points for a very smooth curve, then plot 
#   data and curve fit
xeval=np.linspace(min(x_data), max(x_data),200)
#Plot of the data as red dots and fit as blue line
pp.plot(x_data, y_data,'.r',xeval,gls(xeval),'-b')
pp.xlabel('Time in Days',{"fontsize":16})
pp.ylabel("Cases",{"fontsize":16})
pp.title("Model Fit for Singapore")
pp.legend(('Data Singapore','fit'),loc=0)
pp.show()

#Results-----------------------------------------------------------------------

#Hong Kong
"""Hong Kong had a beta of 7.41671353e-05 and a gamma of 7.15455594e+00
which leads to a reproduction number of 1.0366420503364424 (Rsquared = 0.36)
Initial conditions: p_guess = [0.000001, 0.01],y0 = [100000 ,1,0]  """
#China
"""China had a beta of 2.86898682e-05 and a gamma of 2.68446793e+00
which leads to a reproduction number of 1.0679185572168077 (Rsquared = 0.51)
Initial conditions:p_guess = [0.00001, 0.01],y0 = [100000 ,1,0] """
#Singapore
"""Singapore had a beta of 1.3899666e-04 and a gamma of 1.3755240e+01
which leads to a reproduction number of 1.0104997084928151 (Rsquared = 0.37)
Initial Values = p_guess = [0.00001, 0.01], y0 = [100000 ,1,0]"""
#Taiwan
"""Taiwan had a beta of 1.34606773e-04 and a gamma of 1.26301529e+00
which leads to a reproduction number of 1.065757275082623 (Rsquared = 0.33)
Initial Values:p_guess = [0.000001, 0.01], y0 = [10000 ,1,0]"""






















