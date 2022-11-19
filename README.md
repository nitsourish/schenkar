# Machine Learning Task - DB Schenker

to build an accurate model based on linear regression able to predict the costs using the provided data

### **Steps To **

#### a) All the events are associated with one specific Advertiser(or at least same ad subcategory) and Publisher. Else there will be biased comparison amongst different bid price and the associated win rate(some ads are comparative less demanding, so the average highest bid price may lower and vice-versa)
#### b) Based on the input the upper bound amount of advertiser is willing to pay for a win is 0.50
#### c) There is no other experimented bid_price points other than the mentioned in the data
#### d) There might be domain(real time bidding) specific optimization algorithm, however with time constraint a simple analytical solution is proposed

curl --location -g --request POST 'curl --location --request POST '\''http://localhost:5000/predict'\'' /
--header '\''Content-Type: application/json'\'' /
--data-raw '{
    "origin_latitude" : 26.68,
    "origin_longitude" : 69.43,
    "destination_latitude" :  24.75,
    "destination_longitude" :  59.42,
    "weight": 0.447,
    "loading_meters" : 0.195,
    "shipping_date": 2018-03-22
}'
