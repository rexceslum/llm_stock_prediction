from newsdataapi import NewsDataApiClient

api = NewsDataApiClient (apikey="pub_6f7ef601e9b14e88a45791a1c1cca9af")

response = api.market_api(language="en",
                          symbol="NVDA",
                          # from_date="2023-05-01",
                          # to_date="2026-05-01",
                          removeduplicate=True,
                          size=10)

print(response)