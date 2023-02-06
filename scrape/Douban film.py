import json

import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (<system-information>) <platform> (<platform-details>) <extensions>'
}

url = ' https://movie.douban.com/j/chart/top_list'
parma = {
    'type': '24',
    'interval_id': '100:90',
    'action': '',
    'start': '0',
    'limit': '100',
}

response = requests.get(url=url, params=parma, headers=headers)

list_data = response.json()

fp = open('./douban.json', 'w', encoding='utf-8')
json.dump(list_data, fp=fp, ensure_ascii=False)
