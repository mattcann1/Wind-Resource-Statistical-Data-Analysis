# -----------------------------------------------------------#
# (C) 2020 Matthew Cann
# Released under MIT Public License (MIT)
# email mcann@uwaterloo.ca
# -----------------------------------------------------------


# IMPORTS......................................................................
import math
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import exponweib 
from scipy.stats import rayleigh
import os 

# DATA PROCESSING...............................................................
Directory = 'E:\Documents\Waterloo-Masters\ME 765' # Main Directory Working in
os.chdir(Directory)
data = pd.read_csv('Met_Data_2016.csv')#Read wind data from file'Met_Data_2016.csv' 

#ORGANZING DATA MONTHLY........................................................
#Makes empthy data sets for each month
firstmonth_data = pd.DataFrame(columns=['Avg Wind Speed @ 80m [m/s]',
                    'Avg Wind Speed @ 50m [m/s]','Avg Wind Speed @ 20m [m/s]',
                    'Avg Wind Speed @ 10m [m/s]','Avg Wind Speed @ 5m [m/s]',
                    'Avg Wind Speed @ 2m [m/s]'])
secondmonth_data = pd.DataFrame(columns=['Avg Wind Speed @ 80m [m/s]','Avg Wind Speed @ 50m [m/s]','Avg Wind Speed @ 20m [m/s]','Avg Wind Speed @ 10m [m/s]',
                    'Avg Wind Speed @ 5m [m/s]','Avg Wind Speed @ 2m [m/s]'])
thirdmonth_data = pd.DataFrame(columns=['Avg Wind Speed @ 80m [m/s]','Avg Wind Speed @ 50m [m/s]','Avg Wind Speed @ 20m [m/s]','Avg Wind Speed @ 10m [m/s]',
                    'Avg Wind Speed @ 5m [m/s]','Avg Wind Speed @ 2m [m/s]'])
fourthmonth_data = pd.DataFrame(columns=['Avg Wind Speed @ 80m [m/s]','Avg Wind Speed @ 50m [m/s]','Avg Wind Speed @ 20m [m/s]','Avg Wind Speed @ 10m [m/s]',
                    'Avg Wind Speed @ 5m [m/s]','Avg Wind Speed @ 2m [m/s]'])
fifthmonth_data = pd.DataFrame(columns=['Avg Wind Speed @ 80m [m/s]','Avg Wind Speed @ 50m [m/s]','Avg Wind Speed @ 20m [m/s]','Avg Wind Speed @ 10m [m/s]',
                    'Avg Wind Speed @ 5m [m/s]','Avg Wind Speed @ 2m [m/s]'])
sixthmonth_data = pd.DataFrame(columns=['Avg Wind Speed @ 80m [m/s]','Avg Wind Speed @ 50m [m/s]','Avg Wind Speed @ 20m [m/s]','Avg Wind Speed @ 10m [m/s]',
                    'Avg Wind Speed @ 5m [m/s]','Avg Wind Speed @ 2m [m/s]'])
## ============================================================================
#Stores corresponding month rows and wind speed columns to data set
firstmonth_data = data.iloc[0:44643,[4,6,8,10,12,14]]#Days(0 -31] Jan
secondmonth_data = data.iloc[44643:86403,[4,6,8,10,12,14]]#Days (31-60] Feb
thirdmonth_data = data.iloc[86403:131043,[4,6,8,10,12,14]]#Days (60-91] March
fourthmonth_data = data.iloc[131043:174257,[4,6,8,10,12,14]]#Days(91-121] April
fifthmonth_data = data.iloc[174257:218897,[4,6,8,10,12,14]]#Days (31-152] May 
sixthmonth_data = data.iloc[218897:262097,[4,6,8,10,12,14]]#Days (152-182] June

#%%............................................................................
'''Calculates the mean wind speeds for each of the elevation levels then plots
the means with the corresponding error bars ona velocity profile graph'''


writer =pd.ExcelWriter(time.strftime('Results %Y-%m-%d.xlsx')) #Writer for the excel sheet of results

# Make empty lists for wind speed and std windspeed
Ave_Wind_Speeds = []
Ave_STD_Windspeeds = []

#For the corresponding columns in data, calculates the mean and adds to list
for column in data[['Avg Wind Speed @ 80m [m/s]','Avg Wind Speed @ 50m [m/s]',
                    'Avg Wind Speed @ 20m [m/s]','Avg Wind Speed @ 10m [m/s]',
                    'Avg Wind Speed @ 5m [m/s]','Avg Wind Speed @ 2m [m/s]']]:
     Ave_Wind_Speeds.append(data[column].mean())

#For the corresponding columns in data, calculates the mean and adds to list
for column in data[['Avg Wind Speed (std dev) @ 80m [m/s]','Avg Wind Speed (std dev) @ 50m [m/s]',
                    'Avg Wind Speed (std dev) @ 20m [m/s]','Avg Wind Speed (std dev) @ 10m [m/s]',
                    'Avg Wind Speed (std dev) @ 5m [m/s]','Avg Wind Speed (std dev) @ 2m [m/s]']]:
     Ave_STD_Windspeeds.append(data[column].mean())
     
def main1():     
    print("Ave_Wind_Speeds",  Ave_Wind_Speeds)
    print("Ave_STD_Windspeeds", Ave_STD_Windspeeds)
    
    #Plots Average wind speeds and error bars
    plt.figure(1) 
    yy = [80,50,20,10,5,2]
    plt.errorbar(Ave_Wind_Speeds, yy,xerr=Ave_STD_Windspeeds,fmt='.k',
                 markersize='10', ecolor='red',capsize=6, elinewidth=1) 
    plt.ylabel('Height above Ground [m]')
    plt.xlabel('Wind Speed [m/s]')
    plt.savefig('E:\Documents\Waterloo-Masters\ME 765\Figures\ average.png')
    plt.show()
    
    #loads results into dataframe
    results_2_a = pd.DataFrame(Ave_Wind_Speeds, index = ['80m', '50m', '20m', '10m',
                                                         '5m', '2m'])
    results_2_a.columns = ['2.a - Ave_Wind_Speeds']
    results_2_a.insert(1,'2.a - Ave_STD_Windspeeds', Ave_STD_Windspeeds)
    
    results_2_a.to_excel(writer, '2A') #Results dataframe where values are stored
    
    
#%%............................................................................    
'''Alpha from the log law is determined from the mean wind velocities determined'''
yy=[50,80]
xx = [Ave_Wind_Speeds[1], Ave_Wind_Speeds[0]]
plt.scatter(xx,yy)
plt.show()

alpha = (math.log10(Ave_Wind_Speeds[1]/Ave_Wind_Speeds[0]))/math.log10(50/80)
print('Alpha from data --','%0.4f' % alpha)

results_2_d =  pd.DataFrame({'Alpha from data between 50 and 80m':[alpha]} )
results_2_d.to_excel(writer, '2.d') #writes data frame to excel sheet '2.g' 
                                                        #within main excel file
#%%............................................................................
'''Calculates the Weibull parameters of the wind data'''      
                                                  
AvgWindSpeed_eighty = Ave_Wind_Speeds[0]   # Average Wind speed at 80m
AvgSTDWindSpeed_eighty= Ave_STD_Windspeeds[0] # Average STD of wind at 80m
sigma_varience = AvgSTDWindSpeed_eighty**2   # Varience from STD

Weibull_k = (sigma_varience/AvgWindSpeed_eighty)**(-1.086)
Weibull_c = (0.568 +0.433/Weibull_k)**(-1/Weibull_k)
#print ('Weibull Parameter K --', '%0.4f' % Weibull_k)
#print ('Weibull Parameter c --', '%0.4f' % Weibull_c, '[m/s]')

#loads results into dataframe
results_2_g = pd.DataFrame({'Weibull Parameter K':[Weibull_k], 
                            'Weibull Parameter c': [Weibull_c]} )
results_2_g.to_excel(writer, '2.g') #writes data frame to excel sheet '2.g' within main excel file

writer.save()

#%%............................................................................
'''Calculates the average wind velocity for all height levels as a function of the month of the year
and plots the trend'''

#EMPTY LISTS
Ave_firstmonth = []
Ave_secondmonth = []
Ave_thirdmonth = []
Ave_fourthmonth = []
Ave_fifthmonth = []
Ave_sixthmonth = []

for column in firstmonth_data[['Avg Wind Speed @ 80m [m/s]','Avg Wind Speed @ 50m [m/s]',
                    'Avg Wind Speed @ 20m [m/s]','Avg Wind Speed @ 10m [m/s]',
                    'Avg Wind Speed @ 5m [m/s]','Avg Wind Speed @ 2m [m/s]']]:
    Ave_firstmonth.append(firstmonth_data[column].mean())
for column in secondmonth_data[['Avg Wind Speed @ 80m [m/s]','Avg Wind Speed @ 50m [m/s]',
                    'Avg Wind Speed @ 20m [m/s]','Avg Wind Speed @ 10m [m/s]',
                    'Avg Wind Speed @ 5m [m/s]','Avg Wind Speed @ 2m [m/s]']]:
    Ave_secondmonth.append(secondmonth_data[column].mean())
for column in thirdmonth_data[['Avg Wind Speed @ 80m [m/s]','Avg Wind Speed @ 50m [m/s]',
                    'Avg Wind Speed @ 20m [m/s]','Avg Wind Speed @ 10m [m/s]',
                    'Avg Wind Speed @ 5m [m/s]','Avg Wind Speed @ 2m [m/s]']]:
    Ave_thirdmonth.append(thirdmonth_data[column].mean())
for column in fourthmonth_data[['Avg Wind Speed @ 80m [m/s]','Avg Wind Speed @ 50m [m/s]',
                    'Avg Wind Speed @ 20m [m/s]','Avg Wind Speed @ 10m [m/s]',
                    'Avg Wind Speed @ 5m [m/s]','Avg Wind Speed @ 2m [m/s]']]:
    Ave_fourthmonth.append(fourthmonth_data[column].mean())
for column in fifthmonth_data[['Avg Wind Speed @ 80m [m/s]','Avg Wind Speed @ 50m [m/s]',
                    'Avg Wind Speed @ 20m [m/s]','Avg Wind Speed @ 10m [m/s]',
                    'Avg Wind Speed @ 5m [m/s]','Avg Wind Speed @ 2m [m/s]']]:
    Ave_fifthmonth.append(fifthmonth_data[column].mean())
for column in sixthmonth_data[['Avg Wind Speed @ 80m [m/s]','Avg Wind Speed @ 50m [m/s]',
                    'Avg Wind Speed @ 20m [m/s]','Avg Wind Speed @ 10m [m/s]',
                    'Avg Wind Speed @ 5m [m/s]','Avg Wind Speed @ 2m [m/s]']]:
    Ave_sixthmonth.append(sixthmonth_data[column].mean())    
    
# =============================================================================

eighty_m_average = [Ave_firstmonth[0],Ave_secondmonth[0],Ave_thirdmonth[0],
                    Ave_fourthmonth[0], Ave_fifthmonth[0],Ave_sixthmonth[0]]
fifty_m_average = [Ave_firstmonth[1],Ave_secondmonth[1],Ave_thirdmonth[1],
                    Ave_fourthmonth[1], Ave_fifthmonth[1],Ave_sixthmonth[1]]
twenty_m_average = [Ave_firstmonth[2],Ave_secondmonth[2],Ave_thirdmonth[2],
                    Ave_fourthmonth[2],Ave_fifthmonth[2],Ave_sixthmonth[2]]
ten_m_average = [Ave_firstmonth[3],Ave_secondmonth[3],Ave_thirdmonth[3],
                    Ave_fourthmonth[3], Ave_fifthmonth[3],Ave_sixthmonth[3]]
five_m_average = [Ave_firstmonth[4],Ave_secondmonth[4],Ave_thirdmonth[4],
                    Ave_fourthmonth[4], Ave_fifthmonth[4],Ave_sixthmonth[4]]
two_m_average = [Ave_firstmonth[5],Ave_secondmonth[5],Ave_thirdmonth[5],
                    Ave_fourthmonth[5], Ave_fifthmonth[5],Ave_sixthmonth[5]]

months = ['Jan','Feb','March', 'April', 'June', 'July']

#columns = months, index = ['80m', '50m', '20m', '10m', '5m', '2m']
#results_2_b.to_excel(writer, '2.b') #writes data frame to excel sheet '2.g' 
                                                        #within main excel file
#print(results_2_b)

plt.figure(2) 
plt.plot(months,eighty_m_average, label='80m')
plt.plot(months,fifty_m_average,label='50m')
plt.plot(months,twenty_m_average,label='20m')
plt.plot(months,ten_m_average, label='10m')
plt.plot(months,five_m_average,label='5m')
plt.plot(months,two_m_average,label='2m')
plt.legend(loc='upper right')
plt.ylabel('Wind Speed [m/s]')
plt.xlabel('Month')
plt.ylim(-0.5,8)
plt.savefig('E:\Documents\Waterloo-Masters\ME 765\Figures\months.png')
plt.show()

#%%............................................................................
'''Plots the wind speed over a single date at the 80meter level'''
plt.figure(3) 
data.iloc[0:1441,4].plot()
plt.xlabel('Time [min]')
plt.ylabel('Wind Speed [m/s]')
#plt.title('Wind Speed over 1 day (2016-01-01)')
plt.savefig('E:\Documents\Waterloo-Masters\ME 765\Figures\windover1day.png')

#%%............................................................................
'''Plots the turbulence intensity measurements from the data with respect to the IEC high and 
low standards'''

Ave_Turbulence = []
for column in data[['Turbulence Intensity @ 20m','Turbulence Intensity @ 50m',
                    'Turbulence Intensity @ 80m']]:
     Ave_Turbulence.append(data[column].mean())

print('Average Turbulence at 20 metres --', '%0.4f' %Ave_Turbulence[0])
print('Average Turbulence at 50 metres --', '%0.4f' %Ave_Turbulence[1])
print('Average Turbulence at 80 metres --', '%0.4f' %Ave_Turbulence[2])

list(range(30))
speed = list(np.linspace(0.9,25,num=1000))
high = [0.18*(2+15/average)/3 for average in speed]
low = [0.16*(3+15/average)/4 for average in speed]

plt.figure(4)
data.plot.scatter(y='Turbulence Intensity @ 20m',x='Avg Wind Speed @ 20m [m/s]',
                  label='Data')
plt.plot(speed, high, 'k-', color = 'k', label='IEC High')
plt.plot(speed, low, 'k--', color = 'k',label='IEC Low')
plt.legend(loc ='upper right')
plt.savefig('E:\Documents\Waterloo-Masters\ME 765\Figures\Turbulence_intensity@20.png')
plt.show()

plt.figure(5)
data.plot.scatter(y='Turbulence Intensity @ 50m',x='Avg Wind Speed @ 50m [m/s]',
                  label='Data') 
plt.plot(speed, high, 'k-', color = 'k', label='IEC High')
plt.plot(speed, low, 'k--', color = 'k',label='IEC Low')
plt.legend(loc ='upper right')
plt.savefig('E:\Documents\Waterloo-Masters\ME 765\Figures\Turbulence_intensity@50.png')

plt.show()

plt.figure(6)
data.plot.scatter(y='Turbulence Intensity @ 80m',x='Avg Wind Speed @ 80m [m/s]',
                  label='Data')
plt.plot(speed, high, 'k-', color = 'k', label='IEC High')
plt.plot(speed, low, 'k--', color = 'k',label='IEC Low')
plt.legend(loc ='upper right')
plt.savefig('E:\Documents\Waterloo-Masters\ME 765\Figures\Turbulence_intensity@80.png')
plt.show()


#%%.............................................................................
'''Plots the histograms of the wind velocity frequency for each seperate height level'''

plt.figure(7) 
data['Avg Wind Speed @ 80m [m/s]'].plot.hist(bins=25, density=True)
plt.legend()
plt.xlabel('Wind Speed [m/s]', fontsize =14)
plt.ylabel('Frequency [%]', fontsize =14)
plt.ylim(0,0.3)
plt.savefig('E:\Documents\Waterloo-Masters\ME 765\Figures\Histogram_wind80.png')
plt.show()
 
plt.figure(8) 
data['Avg Wind Speed @ 50m [m/s]'].plot.hist(density=True, bins=25)
plt.legend()
plt.xlabel('Wind Speed [m/s]', fontsize =14)
plt.ylabel('Frequency [%]', fontsize =14)
plt.ylim(0,0.3)
plt.savefig('E:\Documents\Waterloo-Masters\ME 765\Figures\Histogram_wind50.png')
plt.show()
# 
plt.figure(9) 
data['Avg Wind Speed @ 20m [m/s]'].plot.hist(density=True, bins=25)
plt.legend()
plt.xlabel('Wind Speed [m/s]', fontsize =14)
plt.ylabel('Frequency [%]', fontsize =14)
plt.ylim(0,0.3)
plt.savefig('E:\Documents\Waterloo-Masters\ME 765\Figures\Histogram_wind20.png')
plt.show()
# 
plt.figure(10) 
data['Avg Wind Speed @ 10m [m/s]'].plot.hist(density=True, bins=25)
plt.legend()
plt.xlabel('Wind Speed [m/s]', fontsize =14)
plt.ylabel('Frequency [%]', fontsize =14)
plt.ylim(0,0.3)
plt.savefig('E:\Documents\Waterloo-Masters\ME 765\Figures\Histogram_wind10.png')
plt.show()
# 
plt.figure(11) 
data['Avg Wind Speed @ 5m [m/s]'].plot.hist(density=True, bins=25)
plt.legend()
plt.xlabel('Wind Speed [m/s]', fontsize =14)
plt.ylabel('Frequency [%]', fontsize =14)
plt.ylim(0,0.3)
plt.savefig('E:\Documents\Waterloo-Masters\ME 765\Figures\Histogram_wind5.png')
plt.show()
# 
plt.figure(12) 
data['Avg Wind Speed @ 2m [m/s]'].plot.hist(density=True, bins=25)
plt.legend()
plt.xlabel('Wind Speed [m/s]', fontsize =14)
plt.ylabel('Frequency [%]', fontsize =14)
plt.ylim(0,0.3)
plt.savefig('E:\Documents\Waterloo-Masters\ME 765\Figures\Histogram_wind2.png')
plt.show()
#
#%%............................................................................ 

#PLOTS HISTOGRAM OF FREQUENCY VS WIND SPEED[M/S]
plt.figure(8) 

#Histrogram of Velocity Data===================================================
Directory = 'E:\Documents\Waterloo-Masters\ME 765\Figures'


def super_hist(data_name):
    wind = data[data_name]
    support = np.linspace(wind.min(), wind.max(), 100)
    p0, p1, p2 = scipy.stats.weibull_min.fit(wind, floc=0)
    plt.plot(support, scipy.stats.weibull_min.pdf(support, p0, p1, p2), 'r-', 
             lw=2, label = 'Weibull')
    
    data[data_name].plot.hist( weights=np.ones_like(data.index)/len(data.index),
        bins=25)

    param = rayleigh.fit(wind) # distribution fitting
    plt.plot(support, rayleigh.pdf(support, loc=param[0],scale=param[1]),lw=2, 
             label = 'Rayleigh')
    plt.legend()
    plt.xlabel("Mean wind speed [m/s]")
    plt.ylabel("Frequency [%]")
    #figure_path = Directory + '\ ' + data_name + '.png'
    plt.savefig(Directory + '\ ' + 'Super_' + data_name[17:19] + '.png')
    plt.show()
    return

print(super_hist('Avg Wind Speed @ 80m [m/s]'))
print(super_hist('Avg Wind Speed @ 50m [m/s]'))
print(super_hist('Avg Wind Speed @ 20m [m/s]'))
print(super_hist('Avg Wind Speed @ 10m [m/s]'))
print(super_hist('Avg Wind Speed @ 5m [m/s]'))
print(super_hist('Avg Wind Speed @ 2m [m/s]'))

#MAIN..........................................................................
main1()    
