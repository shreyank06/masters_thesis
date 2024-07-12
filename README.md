

Usage for data collection
```
1. Start ph_init
2. go to http://192.168.254.130:9090 and see prometheus has started
3. Clone this repo
4. Run ./wrapper.sh A B, where argument A is the number of registrations, and B is the number of frequencies
5. After all the registraions are over, the script will extract and load the csv files
```

for performing single step predictions
```
cd Python Code
python3 main.py 2024-01-01T00:00:00 2024-01-02T00:00:00 ABC123 100
```
