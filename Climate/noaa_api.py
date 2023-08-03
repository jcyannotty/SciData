import requests

headers = {'token':"xcexqdPSjLmWOzMHRIwkmXKQtmmOAlDH"}
resp = requests.get('https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets', headers=headers)
resp.json()

resp = requests.get('https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets?datatypeid=TOBS', headers=headers)
resp.status_code
resp.json()

resp = requests.get('https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid=GSOM', headers=headers)
resp.status_code
resp.json()

resp = requests.get('https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&locationid=ZIP:28801&startdate=2005-05-01&enddate=2006-05-01&limit=10'
, headers=headers)
resp.status_code
len(resp.json()['results'])
resp.json().keys()
resp.json()['metadata']

resp = requests.get('ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/v3', headers=headers)
resp.status_code
len(resp.json()['results'])
resp.json().keys()
resp.json()['metadata']