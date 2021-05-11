import time;
import datetime;

# start_time = time.time()
# print(start_time)


# date_and_time = datetime.datetime(2020, 2, 19, 12, 0, 0)
# print(date_and_time)

# time_change = datetime.timedelta(hours=10)
# new_time = date_and_time + time_change

# print(new_time)


# surprisesize - normalsize 

now = datetime.datetime.now()
print (now)
strfNow = now.strftime("%H:%M:%S")
print (strfNow)
print(now + datetime.timedelta(seconds=20))