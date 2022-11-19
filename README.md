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
