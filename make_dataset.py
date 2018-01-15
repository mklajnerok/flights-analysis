import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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

# separate dataset to analyze dynamic between years
df_2007 = df[df['Year'] == 2007].copy()               # shape (475889, 30)
df_2008 = df[df['Year'] == 2008].copy()                # shape (451931, 30)


### PLANE TAIL NUMBER (column 11 - 'TailNum') ######################################

total_num_flights = len(df.index)   # 927820 total number of Delta flights during 2007-2008
df["TailNum"].mode()                # the most popular plane tail number 'N914DE'
df.loc[df["TailNum"] == 'N914DE']   # 4322 flights performed for the most popular one
len(df.TailNum.unique())            # 468 planes in the fleet

# vc for both years
planes_vc = df["TailNum"].value_counts()        # how many flights each of the planes was scheduled
planes_vc.describe()

# plot hist
matplotlib.style.use('ggplot')
plt.figure(figsize=(8,5))
plt.hist(planes_vc, bins=100)
plt.title("Number of flights scheduled for each plane in 2007-2008")
plt.xlabel("Number of flights per aircraft")
plt.ylabel("Number of aircrafts")
plt.show()

# vc for separate years
planes_vc_2007 = df_2007["TailNum"].value_counts()
planes_vc_2007.describe()

planes_vc_2008 = df_2008["TailNum"].value_counts()
planes_vc_2008.describe()

# plot hist for separate years
matplotlib.style.use('ggplot')
plt.figure(figsize=(8,5))
plt.hist(planes_vc_2007, bins=75, label='2007', alpha=0.7, stacked='True')
plt.hist(planes_vc_2008, bins=75, label='2008', alpha=0.7, stacked='True')
plt.title("Number of flights scheduled for each aircraft in a given year")
plt.xlabel("Number of flights per aircraft")
plt.ylabel("Number of aircrafts")
plt.legend(loc='upper right')
plt.show()


"""
results:
- there were 449 planes in 2007 fleet
- the average plane performed 1059,89 flights
- min 4, max 2254 flights per machine
- 75% of machines performed 2254 and less flights

- there were 461 planes in 2008
- the average plane performed 980,33 flights
- min 4, max 2130 flights per machine
- 75% of machines performed 1425 and less flights

- there are 468 planes in the Delta fleet
- on the average every plane performed 1982 flight
- there is a plane which was used only 4 times, but the most popular ones did 4322 flights
- 75% of the planes perform around 2879 flights during 2 years

suggest:
- relation with cost exploatation of each machine - 
is the use effective (consider miles flight, passengers numebr - how big machine
- 
"""


##### ORIGIN/DESTINATION (column 17,18 - 'Origin', 'Dest') ######################################

origin_vc = df["Origin"].value_counts()        # how many flights were scheduled from each origin
origin_vc.describe()
df["Origin"].mode()                             # most popular origin ATL airport

# plot (not interesting)
matplotlib.style.use('ggplot')
plt.figure(figsize=(8,5))
plt.hist(origin_vc, bins=50)
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
- ATL is the most popular dest for Delta Airlines (304.202 flights scheduled). The second most popular is only 58.196 (SLC)
- only 15 origin have more than 10k flights scheduled
- only 28 origins are below 1k flights scheduled
"""

# vc for separate years
origin_vc_2007 = df_2007["Origin"].value_counts()
origin_vc_2007.describe()
df_2007["Origin"].mode()
origin_vc_2008 = df_2008["Origin"].value_counts()
origin_vc_2008.describe()
df_2008["Origin"].mode()

dest_vc_2007 = df_2007["Dest"].value_counts()
dest_vc_2007.describe()
df_2007['Dest'].mode()
dest_vc_2008 = df_2008["Dest"].value_counts()
dest_vc_2008.describe()
df_2008['Dest'].mode()

# convert dest_vc to df, check the difference between 2008 and 2007 (is it 5% everywhere)
dest_df_2007 = make_df_from_series(dest_vc_2007, {'Dest':'Number of flights'}, 'Dest')
dest_df_2008 = make_df_from_series(dest_vc_2008, {'Dest':'Number of flights'}, 'Dest')
dest_df_2007_2008 = merge_df_on_label(dest_df_2007, dest_df_2008, 'Dest')
dest_df_2007_2008.rename(columns={'Number of flights_x': '2007', 'Number of flights_y': '2008'}, inplace=True)
dest_df_2007_2008['Diff'] = dest_df_2007_2008['2008'] - dest_df_2007_2008['2007']
dest_df_2007_2008['Diff_perc'] = dest_df_2007_2008['Diff'] / dest_df_2007_2008['2007']*100

# plot - not so much
plt.figure(figsize=(8,5))
plt.hist(dest_df_2007_2008['Diff (%)'], bins=50)
plt.title("vvvv")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# descr stat
dest_df_2007_2008['Diff_perc'].describe()

# check for decrease
big_decrease = dest_df_2007_2008.query('(Diff_perc < -20)')
airports_corr['Dest'] = airports_corr['Origin']
big_decrease = merge_df_on_label(big_decrease, airports_corr, 'Dest')

# check for growth
big_growth = dest_df_2007_2008.query('(Diff_perc > 10)')
big_growth = merge_df_on_label(big_growth, airports_corr, 'Dest')

"""
results:
- one less origin airport in 2008 (from 105 to 104)
- average number of flights to single origin is less (from 4532 to 4345)
- the most popular origin airport also got under the trend and have a decrease in flights (from 152999 to 151251)
- the same amount of destination airports
- the same average amount of flghts accepted per destination (around 8512)
- the same min and max (2 and 304202)
"""

# map plot with bubbles https://python-graph-gallery.com/bubble-map/

def make_df_from_series(series_data, new_label, old_index):
    df_data = pd.DataFrame(series_data)
    df_data.rename(columns=new_label, inplace=True)
    df_data[old_index] = df_data.index
    df_data.sort_values(old_index, inplace=True)
    df_data.reset_index(level=0, drop=True, inplace=True)
    return df_data

origin_df = make_df_from_series(origin_vc, {'Origin':'Number of flights'}, 'Origin')

# load airport cordinates csv
airports_corr = pd.read_csv('airport_USA_corr.csv')
airports_corr.rename(columns={'locationID':'Origin'}, inplace=True)
airports_corr['Longitude'] = airports_corr['Longitude'] - 2 * airports_corr['Longitude']


def merge_df_on_label(df_data1, df_data2, label):
    """Merge two data frames on Code column, drop double country column
    :param df_data1: data frame
    :param df_data2: data frame
    :returns df_joined: data frame"""
    df_joined = pd.merge(df_data1, df_data2, on=label)
    #df_joined.drop(['Country_y'], axis=1, inplace=True)
    return df_joined

# merge with origins data
origin_corr = merge_df_on_label(origin_df, airports_corr, 'Origin')

# save to csv
origin_corr.to_csv('origin_with_cordinates.csv')

# plot
from mpl_toolkits.basemap import Basemap

my_dpi = 96
fig = plt.figure(figsize=(800 / my_dpi, 700 / my_dpi), dpi=my_dpi)
fig.text(0.5, 0.87, "Number of flights scheduled from each airport origin in 2007-2008", ha='center', va='center',
         fontsize=14)
m=Basemap(llcrnrlon=-165, llcrnrlat=10,urcrnrlon=-60,urcrnrlat=80)
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='grey', alpha=0.3)
m.drawcoastlines(linewidth=0.1, color="white")
m.readshapefile('st99_d00', name='states', drawbounds=True, color='grey')
m.scatter(origin_corr['Longitude'], origin_corr['Latitude'], s=origin_corr['Number of flights']/100, alpha=0.4, latlon=True)



c = data['labels_enc'], cmap = "Set1"
df.plot.scatter(x='a', y='b', s=df['c']*200);


#plot
map = Basemap(width=9000000,height=8000000,
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',area_thresh=1000.,projection='lcc',\
            lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='#ddaa66', lake_color='aqua')
map.drawcountries()
map.drawstates(color='0.5')
plt.show()

# plot map
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
m.readshapefile('st99_d00', name='states', drawbounds=True)
m.scatter(origin_corr['Longitude'], origin_corr['Latitude'], s=origin_corr['Number of flights']/6, alpha=0.4, latlon=True)
plt.show()







m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)


#check for less popular dest
dest_vc.describe()
less_pop_dest = df.loc[df['Dest'] == 'GUC']


### DISTANCE in miles (column 19 - 'Distance') ######################################

# describe stats
df['Distance'].describe()

# plot hist
matplotlib.style.use('ggplot')
plt.figure(figsize=(8,5))
plt.hist(df['Distance'], bins=30)
plt.title("Flight distances in 2007-2008")
plt.xlabel("Flight distance (miles)")
plt.ylabel("Number of flights")
plt.show()

"""
results:
- there are 927820 flights
- average distance per flight is 931 miles
- variance is high (std is 689 miles)
- the shortest flight was 83 miles, but the longest 4502 miles
- Delta prefers shorter flights - 75% of them are below 1269 miles (NY to SFO is around 2500 miles flight)
"""

df_2007['Distance'].describe()
df_2008['Distance'].describe()

"""
results:
- the average lenghts of flight was shotened (from 941 to 921 miles)
- minimuum flight was 83, in 2008 116 - max 4502, 4502
- around the same amount of flights have the distance below 1200 miles
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
- the average time of taxi out is 20,22 minutes
- the variety is quite, std is 12,63 minutes
- there were some flights with 0 taxi, but the max was 422 minutes
- it's not a common situation for long taxi in, because 75% of flights have 23 minutes or less
"""
# not much change
df_2007['TaxiIn'].describe()
df_2008['TaxiIn'].describe()

# not much change
df_2007['TaxiOut'].describe()
df_2008['TaxiOut'].describe()


### FLIGHT LENGTH (column 14 - 'AirTime') ##############################

df['AirTime'].describe()

"""
results:
- AirTime has Nan in the number of cancelled + diverted
- average flight took 127,45 minutes, with std 81,75
- the shortest flight was 12 min and the longest 618 min
- 75% of the flights are 175 min and shorter (around 3 hours flights)
suggest:
- compare air time for a given plane with its cost of maintaining
"""

df_2007['AirTime'].describe()
total = df_2007.loc[:, "AirTime"].sum()
df_2008['AirTime'].describe()
df_2008['Airtime'].sum()

"""
results:
- not much change between years
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



### DELAYS (column 12,13,15,16,25,26,27,28,29 - 'ActualElapsedTime', 'CRSElapsedTime', 'ArrDelay', 'DepDelay',
### 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay') ##############################

test = df.loc[:][['ArrDelay', 'DepDelay', 'ActualElapsedTime', 'CRSElapsedTime']]

# total number of delays
total_delay = pd.DataFrame()
total_delay = df['ActualElapsedTime'] - df['CRSElapsedTime']
value_filter_1 = total_delay > 0
total_delay[value_filter_1]                                     # 372510 total delays
value_filter_2 = total_delay != 0
total_delay[value_filter_2]                                     # 895082 with any difference in accuracy

df.loc[(df['ActualElapsedTime'] - df['CRSElapsedTime']) > 0]        # 372510 (jaki 40,1%)

total_delay.describe()
"""
results:
- average total delay is around 0, most of the flights have total delay below 6 min. 
    So we can deduct that the flights itself is not prone to delays that much as is the managment of the airport 
    and most delays are probably cause by the dep delays, which cause arr delays by other planes
"""

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
- 
- average departure delay was 7,93 min
- also high variety, std 30,57 min
- it happened that departure was 124 min before scheduled, but the latest was 1003 min delay
- departure delay are not common - not more that 6 min for 75% of the flights
"""

# not much change between years
df_2007['ArrDelay'].describe()
df_2008['ArrDelay'].describe()
df_2007['DepDelay'].describe()
df_2008['DepDelay'].describe()

# change in the on-time ratio
df_2007.loc[df['ArrDelay'] > 15.0].count()      # 97819/475889 = 20,6%
df_2008.loc[df['ArrDelay'] > 15.0].count()      # 94383/451931 = 20,9%
df_2007.loc[df['DepDelay'] > 15.0].count()      # 74622/475889 = 15,7%
df_2008.loc[df['DepDelay'] > 15.0].count()      # 70718/451931 = 15,6%


# why did they happen?
test = df.loc[:][['ActualElapsedTime', 'CRSElapsedTime','CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']]

car_delay_vc = df['CarrierDelay'].value_counts()                # 489472 carrier delay was 0
weather_delay_vc = df['WeatherDelay'].value_counts()            # 565567 weather delay was 0
nas_delay_vc = df['NASDelay'].value_counts()                    # 427768 NAS delay was 0
sec_delay_vc = df['SecurityDelay'].value_counts()               # 574596 security delay was 0
late_delay_vc = df['LateAircraftDelay'].value_counts()          # 489944 late aircraft was 0


# plot arrival delay during the year
fig = plt.figure()
plt.plot(df_2007['Date'], df_2007['ArrDelay'], label='2007', color='skyblue')
plt.show()

# look for the relations with arrival delay
df['ArrDelay'].describe()
late_arrivals = df.query('(ArrDelay > 15)')
late_arrivals.corr()


### CANCELLATION/DIVERTION (column 22/23/24 - 'Cancelled', 'CancellationCode', 'Diverted') ##############################

cancel_vc = df['Cancelled'].value_counts()
cancel_reason_vc = df['CancellationCode'].value_counts()

# donut plot
plt.figure(figsize=(5,4))
plt.pie(cancel_reason_vc, labels=['Carrier', 'Weather', 'NAS'])
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Reasons for a flight cancellation')
plt.show()

divert_vc = df['Diverted'].value_counts()

"""
results:
- 13334 flights were cancelled, 914486 were performed (cancellation ratio 1,4371%)
- flights cancelled because: of the carrier 6320 (47,4% of all cancellations), of the weather 4583 (34,4% of all cancellations),
 of NAS 2431 (18,2% of all cancellations). None was cancelled because of the security reasons
- 2229 flights were diverted (diversion ratio 0,2402%)
"""


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


df_test = df['DepTime'].copy()
format(df_test, '.3f')

time_df = make_df_from_series(origin_vc, {'Origin':'Number of flights'}, 'Origin')

dep_vc_2007 = df_2007['DepTime'].value_counts()
dep_df_2007 = make_df_from_series(dep_vc_2007, {'DepTime':'Number of flights'}, "DepTime")

plt.hist(dep_vc_2007, bins=72, label='2007', alpha=0.7, stacked='True', vertical=True)

arr_vc_2007 = df_2007['ArrTime'].value_counts()



##### DATES in the year, week (column 1,2,3,4 - Year', 'Month', 'DayofMonth', 'DayOfWeek') ##############################

df_head = df.head()         # the dataset begins on Jan 17th 2007
df_tail = df.tail()         # the dataset ends on Dec 13th 2008

# popularity ranking for year, month, day of the week
year_vc = df['Year'].value_counts()
month_vc = df['Month'].value_counts()
dayofweek_vc = df['DayOfWeek'].value_counts()

# create a datetime object in a new column
df_2007["Date"] = df_2007["Year"].map(str) + "/" + df_2007["Month"].map(str) + "/" + df_2007["DayofMonth"].map(str)
df_2007["Date"] = pd.to_datetime(df_2007["Date"])

df_2008["Date"] = df_2008["Year"].map(str) + "/" + df_2008["Month"].map(str) + "/" + df_2008["DayofMonth"].map(str)
df_2008["Date"] = pd.to_datetime(df_2008["Date"])

# create DF from series
date_vc_2007 = df_2007['Date'].value_counts()
date_df_2007 = make_df_from_series(date_vc_2007, {'Date':'Flights'}, 'Date')

date_vc_2008 = df_2008['Date'].value_counts()
date_df_2008 = make_df_from_series(date_vc_2008, {'Date':'Flights'}, 'Date')

# plot comparison between years
fig = plt.figure()
plt.subplot(211)
fig.text(0.5, 0.93, "Comparison of a daily flights frequency\n in 2007-2008", ha='center', va='center',
         fontsize=14)
plt.plot(date_df_2007['Date'], date_df_2007['Flights'], label='2007', color='skyblue')
plt.ylim((500,1600))
plt.subplot(212)
plt.plot(date_df_2008['Date'], date_df_2008['Flights'], label='2008', color='orange')
plt.ylim((500,1600))
fig.text(0.03, 0.5, 'Number of flights per day', ha='center', va='center', rotation='vertical')
plt.show()

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


### Additional Comments ##############################

# add category code column for: flight number, tail number, origin, destination
def add_category_code_column(df_data, features_list):
    for group in features_list:
        df_data[group+'_cat_code'] = df_data[group].astype(dtype='category').cat.codes

add_category_code_column(df, ['FlightNum', 'TailNum', 'Origin', 'Dest'])
add_category_code_column(df, ['CancellationCode'])


# check for corr with arrival delays
check_corr_df = df.loc[:][['DepTime', 'ArrTime', 'FlightNum_cat_code', 'TailNum_cat_code', 'Origin_cat_code', 'Dest_cat_code',
                    'Distance','TaxiIn', 'TaxiOut', 'ArrDelay']]
check_corr_df.corr()['ArrDelay']


# add column with datetime object
df["Date"] = df["Year"].map(str) + "/" + df["Month"].map(str) + "/" + df["DayofMonth"].map(str)
df["Date"] = pd.to_datetime(df["Date"])

# check for corr with date
check_corr_df_2 = df.loc[:][['DepTime', 'ArrTime', 'FlightNum_cat_code', 'TailNum_cat_code', 'Origin_cat_code', 'Dest_cat_code',
                    'Distance','TaxiIn', 'TaxiOut', 'Cancelled', 'Diverted', 'ArrDelay', 'Date']]
check_corr_df_2.corr()['Date']

# corr matrix
corr_df = df.corr()
corr_matrix = corr_df.as_matrix()

# plot corr matrix as heatmap
plt.figure(figsize=(8,8))
plt.pcolor(corr_df)
plt.colorbar()
plt.show()
plt.savefig("corr_flavors.pdf")

"""
high correlations from matrix:
- Scheduled Arrival time with Departure Time = 0,71
- Arr Time with Departure Time = 0,63
- Distance and TailNumber = -0,40
- Origin and Destination = -0,24
- Taxi-in and Destination = -0,21
- Taxi-out and Arrival Delay = 0,36
"""

# corr between month and the rest - no strong correlations
check_corr_df_3 = df.loc[:][['Month', 'DepTime', 'ArrTime', 'FlightNum_cat_code', 'TailNum_cat_code', 'Origin_cat_code',
                             'Dest_cat_code', 'Distance','TaxiIn', 'TaxiOut', 'Cancelled', 'CancellationCode_cat_code',
                             'Diverted', 'ArrDelay', 'Date']]
check_corr_df_3.corr()

"""
ideas for correlations:
- plain tail number and origin/sedtination = -0,10
- number of delays and date of the year
- origin and destination and time of the year and cance lreason
- distance and delay amount = -0,00
- origin/destination and delay amount = almost None
- weather cancellation and time of the year
- at what time people travel during the week and weekends?

- check global air time for a given plane tail number 
- why some of the planes are flying only 50 flight per 2 yeras - check fo r the reason in different coluns
- why some of the airports have only a few flights, but others have plenty 
- opźnienia o jakicho dzinach, porach dnia, porach miesiaca, czy roku, numer samolotu, lotnisko
- to samo z cancel

"""



"""
suggestions:
- we do not have data for the amount of passenger carried, condition of machines
"""
