from newsdataapi import NewsDataApiClient

api = NewsDataApiClient (apikey="pub_6f7ef601e9b14e88a45791a1c1cca9af")

response = api.news_api()

print(response)