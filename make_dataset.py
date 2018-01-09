import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


### DATA PREPROCESSING ################################################

# load csv
data_2007 = pd.read_csv("2007.csv")
data_2008 = pd.read_csv("2008.csv")

test = data_2007.loc[:100][['Year', 'Month', 'DayofMonth', 'DayOfWeek']]
data_2007["UniqueCarrier"].isnull().sum()

# identify carriers
carriers = data_2007.UniqueCarrier.unique()

# choose Delta Airlines
df_dl_2007 = data_2007.loc[data_2007["UniqueCarrier"] == "DL"]
df_dl_2008 = data_2008.loc[data_2008["UniqueCarrier"] == "DL"]

# concat data frames, reset index, save csv
dl_data = pd.concat([df_dl_2007, df_dl_2008])
dl_data.reset_index(level=0, drop=True, inplace=True)
dl_data.to_csv('delta_flights_data.csv')

# load Delta Airlines data
df = pd.read_csv("delta_flights_data.csv")
df.drop('Unnamed: 0', axis=1, inplace=True)

# check types of data
df.dtypes

# check relation between columns 5,6,7,8 and 15,16
delays = df.iloc[:, [4,5,6,7,14,15]]

# show column labels
labels = list(df.columns.values)

# check Nan
df.isnull().any()
df.isnull().sum()           # Nans are 1,6773% of whole data records


### PLANE TAIL NUMBER (column 11 - 'TailNum') ######################################

total_num_flights = len(df.index)   # total number of Delta flights during 2007-2008
df["TailNum"].mode()                # the most popular plane tail number 'N914DE'
df.loc[df["TailNum"] == 'N914DE']   # 4322 flights performed for the most popular one
len(df.TailNum.unique())            # 468 planes in the fleet

planes_vc = df["TailNum"].value_counts()        # how many flights each of the planes was scheduled
planes_vc.describe()

matplotlib.style.use('ggplot')
plt.figure(figsize=(8,5))
plt.hist(planes_vc, bins=100)
plt.title("Number of flights scheduled for each machine during in 2007-2008")
plt.xlabel("Number of flights scheduled for each aircraft")
plt.ylabel("Number of planes in each interval")
plt.show()

"""
results:
- there are 468 planes in the Delta fleet
- on the average every plane performed 1982 flight
- there is a plane which was used only 4 times, but the most popular ones did 4322 flights
- 75% of the planes perform around 2879 flights during 2 years
"""


### ORIGIN/DESTINATION (column 17,18 - 'Origin', 'Dest') ######################################

origin_vc = df["Origin"].value_counts()        # how many flights were scheduled from each origin
origin_vc.describe()
df["Origin"].mode()                             # most popular origin ATL airport

matplotlib.style.use('ggplot')
plt.figure(figsize=(8,5))
plt.hist(origin_vc, bins=10)
plt.title("Number of flights scheduled from each origin in 2007-2008")
plt.xlabel("Number of flights scheduled from each origin")
plt.ylabel("Number of flights in each interval")
plt.show()

"""
results:
- there are 109 origins in the data
- the mean number of flights scheduled per origin is 8512
- there is a great variety in the data (std 30.058), min is 2 and max is 304.250
- ATL is the most popular origin for Delta Airlines (304.250 flights scheduled). The second most popular is only 58.167 (SLC)
- only 15 origin have more than 10k flights scheduled
- only 28 origins are below 1k flights scheduled
"""

dest_vc = df["Dest"].value_counts()        # how many flights were scheduled from each destination
dest_vc.describe()
df['Dest'].mode()                             # most popular origin ATL airport

"""
results:
- there are 109 destinations in the data
- the mean number of flights scheduled per origin is 8512
- there is a great variety in the data (std 30.054), min is 2 and max is 304.202
- ATL is the most popular origin for Delta Airlines (304.202 flights scheduled). The second most popular is only 58.196 (SLC)
- only 15 origin have more than 10k flights scheduled
- only 28 origins are below 1k flights scheduled
"""

# Would be nice to add map plot with bubbles https://python-graph-gallery.com/bubble-map/


### DISTANCE in miles (column 19 - 'Distance') ######################################

df['Distance'].describe()

matplotlib.style.use('ggplot')
plt.figure(figsize=(8,5))
plt.hist(df['Distance'], bins=10)
plt.title("Number of flights scheduled from each origin in 2007-2008")
plt.xlabel("Number of flights scheduled from each origin")
plt.ylabel("Number of flights in each interval")
plt.show()

"""
results:
- there are 927820 flights
- average distance per flight is 931 miles
- variance is high (std is 689 miles)
- the shortest flight was 83 miles, but the longest 4502 miles
- Delta prefers shorter flights - 75% of them are below 1269 miles (NY to SFO is around 2500 miles flight)
"""


### TAXI IN/OUT in minutes (column 20/21 - 'TaxiIn', 'TaxiOut') ######################################

df['TaxiIn'].describe()
df['TaxiIn'].isnull().sum()

"""
results:
- there are some Nan in the data - 7811, which is 0,8418% of data records
- the average time of taxi in is 8,47 minutes
- the variety is quite, std is 5,75 minutes
- there were some flights with 0 taxi, but the max was 252 minutes
- it's not a common situation for long taxi in, because 75% of flights have 10 minutes or less
"""

df['TaxiOut'].describe()
df['TaxiOut'].isnull().sum()

"""
results:
- there are some Nan in the data - 6799, which is 0,7327% of data records
- the average time of taxi in is 20,22 minutes
- the variety is quite, std is 12,63 minutes
- there were some flights with 0 taxi, but the max was 422 minutes
- it's not a common situation for long taxi in, because 75% of flights have 23 minutes or less
"""


### CANCELLATION/DIVERTION (column 22/23/24 - 'Cancelled', 'CancellationCode', 'Diverted') ##############################

cancel_vc = df['Cancelled'].value_counts()
cancel_reason_vc = df['CancellationCode'].value_counts()

plt.bar(cancel_reason_vc.iloc[:, 0], cancel_reason_vc[1], color = (0.5,0.1,0.5,0.6))
plt.title('My title')
plt.xlabel('categories')
plt.ylabel('values')
plt.show()
# dokonczyc wyres

divert_vc = df['Diverted'].value_counts()

"""
results:
- 13334 flights were cancelled, 914486 were performed (cancellation ratio 1,4371%)
- flights cancelled because: of the carrier 6320 (% of all cancellations), of the weather 4583 (% of all cancellations),
 of NAS 2431 (% of all cancellations). None was cancelled because of the security reasons
- 2229 flights were diverted (diversion ratio 0,2402%)
"""

"""
definitions:
test = df.loc[:][['ArrTime', 'DepTime', 'ActualElapsedTime']]
- ArrTime - DepTime -> ActualElapsedTime (from gate to gate)        /// do not use this way
test = df.loc[:][['CRSArrTime', 'CRSDepTime', 'CRSElapsedTime']]
- CRSArrTime - CRSDepTime -> CRSElapsedTime (from gate to gate)     /// do not use this way
test = df.loc[:][['TaxiOut', 'AirTime', 'TaxiIn', 'ActualElapsedTime']]
- Taxi-Out + Air Time + Taxi-In ->  ActualElapsedTime (from gate to gate)   /// works fine
test = df.loc[:][['ArrTime', 'CRSArrTime', 'ArrDelay']]
- ArrTime - CRSArrTime -> ArrDelay (total)  /// do not use this way
test = df.loc[:][['DepTime', 'CRSDepTime', 'DepDelay']]
- DepTime - CRSDepTime -> DepDelay (total)  /// do not use this way
"""

### FLIGHT LENGTH (column 14 - 'AirTime') ##############################

df['AirTime'].describe()

"""
results:
- AirTime has Nan in the number of cancelled + diverted
- average flight took 127,45 minutes, with std 81,75
- the shortest flight was 12 min and the longest 618 min
- 75% of the flights are 175 min and shorter (around 3 hours flights)
"""

### DELAYS (column 12,13,15,16,25,26,27,28,29 - 'ActualElapsedTime', 'CRSElapsedTime', 'ArrDelay', 'DepDelay',
### 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay') ##############################

# total number of delays
total_delay = pd.DataFrame()
total_delay = df['ActualElapsedTime'] - df['CRSElapsedTime']
value_filter = total_delay > 0
total_delay[value_filter]                                           # 372510 total delays

df.loc[(df['ActualElapsedTime'] - df['CRSElapsedTime']) > 0]        # 372510

total_delay.describe()





# when did they happen?
df['ArrDelay'].describe()
df['DepDelay'].describe()
df.loc[df['ArrDelay'] > 15.0]
df.query('(ArrDelay > 15)')
arrival_delay_vc = df['ArrDelay'].value_counts()

"""
results:
- average arrival delay is 7,57 min
- variety is huge, std 34,62 min
- it happened that a flight was 132 minutes before time (CHECK IT), but the biggest arrival delay was 1007 min
- long arrival delays are not common - most flights have only 12 minutes
- 192292 flights have arrival delay over 15 min - they are not considered on-time (20,7155%)
- average departure delay was 7,93 min
- also high variety, std 30,57 min
- it happened that departure was 124 min before scheduled, but the latest was 1003 min delay
- departure delay are not common - not more thatn 6 min for 75% of the flights
"""

# why did they happen?
test = df.loc[:][['ActualElapsedTime', 'CRSElapsedTime','CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']]

car_delay_vc = df['CarrierDelay'].value_counts()                # 489472 carrier delay was 0
weather_delay_vc = df['WeatherDelay'].value_counts()            # 565567 weather delay was 0
nas_delay_vc = df['NASDelay'].value_counts()                    # 427768 NAS delay was 0
sec_delay_vc = df['SecurityDelay'].value_counts()               # 574596 security delay was 0
late_delay_vc = df['LateAircraftDelay'].value_counts()          # 489944 late aircraft was 0





### TIMES od the day(column 5,6,7,8 - 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime') ##############################

test = df.loc[:][['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime']]
df_test = df['DepTime'].copy()
format(df_test, '.2f')
df_test = int(df_test)
df_test = pd.to_datetime(df_test, format='%H%M')


def extractTime(l):
    l = str(l)
    tt = (l.split(' ')[1]).split('.')[1][-4:];
    return tt[:2] + ':' + tt[-2:];

map(extractTime(x), df_test)
df_test = df_test.apply(extractTime)


### DATES in the year, week (column 1,2,3,4 - Year', 'Month', 'DayofMonth', 'DayOfWeek') ##############################

df_head = df.head()         # the dataset begins on Jan 17th 2007
df_tail = df.tail()         # the dataset ends on Dec 13th 2008

year_vc = df['Year'].value_counts()
month_vc = df['Month'].value_counts()
dayofweek_vc = df['DayOfWeek'].value_counts()

# create a datetime object in a new column
df["Date"] = df["Year"].map(str) + "/" + df["Month"].map(str) + "/" + df["DayofMonth"].map(str)
df["Date"] = pd.to_datetime(df["Date"])

# count
date_vc = df['Date'].value_counts()
df_dates = pd.DataFrame(date_vc)
df_dates.rename(columns={'Date':'NumOfFlights'}, inplace=True)
df_dates['Date'] = df_dates.index
df_dates.reset_index(level=0, drop=True, inplace=True)
df_dates.sort_values('Date', inplace=True)
df_dates.reset_index(level=0, drop=True, inplace=True)

df_dates[df_dates["Date"] == '2008-01-01']

df_dates['2008'] = df_dates.loc[365:, 'NumOfFlights']


x = df_dates['Date']
y1 = df_dates[:365]['NumOfFlights']
y2 = df_dates[365:]['NumOfFlights']

plt.plot(x, y1, data=df_dates, marker='o', markerfacecolor='blue',
         markersize=12, color='skyblue', linewidth=4)

plt.plot('x', 'y2', data=df, marker='', color='olive', linewidth=2)
plt.plot('x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
plt.legend()



"""
results:
- There was a decrease in number of flights scheduled for 2008 by 5,0343%
- there is a gap in flights data - it starts on 17/1/2007 and ends 13/12/2008, 
    so it's very hard to say which month of the year is the most popular
- it seems that there is no obvious most popular day of the week, it's quite the same every day
    Saturday is the only day we see some drop by around 20%
- the most popular days are in October

"""








# ActualElapsedTime, AirTime, ArrDelay have Nan in the number of cancelled + diverted
# CancellationCode has Nan in the number of flights performed
# Carrier to LateAircraftDelay has Nan 353091



"""
ideas for correlations:
- plain tail number and origin/sedtination
- number of delays and date of the year
- origin and destination and time of the year and cance lreason
- distance and delay amount 
- origin/destination and delay amount
- weather cancellation and time of the year
- at what time people travel during the week and weekends?
"""


"""
suggestions:
- we do not have data for the amount of passenger carried, condition of machines
"""
