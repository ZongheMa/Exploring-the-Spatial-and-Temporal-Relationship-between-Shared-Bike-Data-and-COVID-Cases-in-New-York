import requests
import json

post_url = 'https://fanyi.baidu.com/sug'

word = input('enter a word:')
data = {'kw': word}

header = {
    'User-Agent': 'Mozilla/5.0 (<system-information>) <platform> (<platform-details>) <extensions>'
}
response = requests.post(post_url, data, header)

dic_obj = response.json()

fp = open(f'./{word}].json', 'w', encoding='utf-8')
json.dump(dic_obj, fp=fp, ensure_ascii=False)
