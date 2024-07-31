

**Usage for data collection**

1. Start ph_init
2. go to http://192.168.254.130:9090 and see prometheus has started or not, if yes, proceed with next steps
3. Clone this repo
4. Check time on the terminal where the ph_init is started with this command
```
date
```
if the time is 2 hours behind the actual time, then edit the wrapper.sh script and replace the following line on line number 28 and 40 
from 
```
local start_time=$(date -d "-2 hour" +"%Y-%m-%dT%H:%M:%S")
```
to
```
local start_time=$(date +"%Y-%m-%dT%H:%M:%S")
```
5. Run ./wrapper.sh A B, where argument A is the number of registrations, and B is the number of frequencies
6. After all the registraions are over, the script will extract and load the csv files


**for performing single step predictions**
```
cd Python Code
python3 main.py 2024-01-01T00:00:00 2024-01-02T00:00:00 ABC123 100
```

To establish socks proxy
run this command on host
```
ssh -D 1337 saparia@192.168.143.5
```
and change proxy settings on browser and choose 
```
socks host: localhost, port: 1337
```