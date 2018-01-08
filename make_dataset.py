import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


### DATA PREPROCESSING ################################################

# load csv
data_2007 = pd.read_csv("2007.csv")
data_2008 = pd.read_csv("2008.csv")

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
for col in df:
    print(col, df[col].dtypes)

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

### FLIGHT LENGTH (column 12,13,14 - 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime') ##############################





### DELAYS (column 15,16,25,26,27,28,29 - ArrDelay', 'DepDelay',
### 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay') ##############################




### TIMES od the day(column 5,6,7,8 - 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime') ##############################




### DATES in the year, week (column 1,2,3,4 - Year', 'Month', 'DayofMonth', 'DayOfWeek') ##############################

# kiedy najwiecej lotow, w roku
# jaki dzien tygodnia jest nnajbardziej popularny






# ActualElapsedTime, AirTime, ArrDelay have Nan in the number of cancelled + diverted
# CancellationCode has Nan in the number of flights performed
# Carrier to LateAircraftDelay has Nan 353091




# convert 3 columns in date time object

