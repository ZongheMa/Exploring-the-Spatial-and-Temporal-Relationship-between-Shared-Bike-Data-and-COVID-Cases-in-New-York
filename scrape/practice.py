import requests

url = 'https://www.sogou.com'
response = requests.get(url)

page_text = response.text

print(page_text)

with open('./sogou.html', 'w', encoding='utf-8') as fp:
    fp.write(page_text)

print('stage 1 is fnished')

url = 'https://www.sogou.com/web?query='
kw = input('enter a value:')
header = {
    'User-Agent': 'Mozilla/5.0 (<system-information>) <platform> (<platform-details>) <extensions>'

}
param = {
    'query': kw
}
reponse = requests.get(url, param, headers=header)
page_text = reponse.text
filename = kw + '.html'
with open(filename, 'w', encoding='utf-8') as fp:
    fp.write(page_text)
print(filename, 'successfully saved')

