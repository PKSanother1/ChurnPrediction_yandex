## Создание виртуальной машины в Яндекс облаке:

Выбрать образ виртуальной машины: Ubuntu 24.04 LTS
Выбрать минимальную используемую память SSD в 10гб
Выбрать зону доступности: автомат
Выбрать размерности vCPU и Ram: минимальную в 2гб
## Создать ssh ключ для подключения:
```
#Создает ключ
ssh-keygen -t ed25519
#Копирует ключ в буфер обмена
type C:\Users\<имя_пользователя>\.ssh\id_ed25519.pub | clip 


```
## Далее подключиться к VM:
```
ssh -l zutor <IP-адрес ВМ> #Ip-адрес - это Публичный IPv4-адрес VM
```
При подключении попросить ввести keyphrase.

## После введения keyphrase и подключения к VM, устанавливаем Docker
```
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
docker --version
```

## Выгружаем докер образ и запускаем:

```
docker pull avikinrok/imo-exam-mii
docker run -p 9696:9696 avikinrok/imo-exam-mii
```

## Проверяем работоспособность POST метода:

Запускаем скрипт, который с помощью библиотеки requests делает post запрос по ip нашей VM

```
import requests

#где hostname указываем Публичный IPv4-адрес VM
url = 'http://84.201.181.89:9696/predict'

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'no',
    'tenure': 41,
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'no',
    'deviceprotection': 'yes',
    'techsupport': 'yes',
    'streamingtv': 'yes',
    'streamingmovies': 'yes',
    'contract': 'one_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'bank_transfer_(automatic)',
    'monthlycharges': 79.85,
    'totalcharges': 3320.75,
}

response = requests.post(url, json=customer)
print(f"Код ответа: {response.status_code}")

print(f"Ответ модельки{response.json()}")
```
