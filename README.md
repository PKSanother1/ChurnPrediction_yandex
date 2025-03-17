## Готовим модель
### Загрузка датасета
```
url = 'https://github.com/alexeygrigorev/mlbookcamp-code/raw/refs/heads/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(url)
```
 загружаем датасет по указанной ссылке и сохраняем его в переменную `df`.

### Приведение данных к единому виду
```
df.columns = df.columns.str.lower().str.replace(' ', '_')
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')
```
Здесь  приводим названия столбцов и строковые значения к нижнему регистру и заменяем пробелы на подчеркивания. Это делается для удобства работы с данными.

### Преобразование типов данных
```
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce').fillna(0)
```
 преобразуем столбец `totalcharges` в числовой формат, а если встречаются ошибки (например, пустые строки), то заменяем их на 0.

### Разделение данных на обучающую и тестовую выборки
```
from sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
```
 разделяем данные на три части:
- `df_train` (60% данных) — обучающая выборка.
- `df_val` (20% данных) — валидационная выборка.
- `df_test` (20% данных) — тестовая выборка.

### Подготовка целевой переменной
```
y_train = (df_train.churn == 'yes').astype('int')
y_val = (df_val.churn == 'yes').astype('int')
y_test = (df_test.churn == 'yes').astype('int')
```
 преобразуем целевой столбец `churn` в бинарный формат: `1` для `yes` и `0` для `no`.

### Удаление ненужных столбцов
```
del df_train['customerid']
del df_train['churn']
del df_val['customerid']
del df_val['churn']
```
Удаляем столбцы `customerid` и `churn` из обучающей и валидационной выборок, так как они не нужны для обучения модели.

### Векторизация данных
```
from sklearn.feature_extraction import DictVectorizer
dict_train = df_train.to_dict(orient='records')
dv = DictVectorizer()
dv.fit(dict_train)
X_train = dv.transform(dict_train)
```
 преобразуем данные в формат словаря, а затем векторизуем их с помощью `DictVectorizer`. Это нужно для того, чтобы модель могла работать с категориальными и числовыми данными.

### Обучение модели
```
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```
создаем и обучаем модель логистической регрессии на обучающих данных.

### Предсказание на валидационной выборке
```
dict_val = df_val.to_dict(orient='records')
X_val = dv.transform(dict_val)
y_pred = model.predict(X_val)
```
 преобразуем валидационные данные в формат словаря, векторизуем их и делаем предсказания с помощью обученной модели.

### Оценка точности модели
```
from sklearn.metrics import accuracy_score
accuracy_score(y_val, y_pred)
```
 оцениваем точность модели на валидационной выборке.

### Сохранение модели
```
import pickle
with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)
```
 сохраняем модель и векторизатор в файл `model.bin` с помощью библиотеки `pickle`.


## Поднимаем API 
### Импорт библиотек
```
from flask import Flask, request, jsonify
import pickle
```
- **Flask**:  микрофреймворк для создания веб-приложений на Python
- **request**: Используется для обработки входящих HTTP-запросов
- **jsonify**: Преобразует Python-объекты в JSON-ответы.
- **pickle**:  используется для загрузки модели и векторизатора.



### Загрузка модели и векторизатора
```
with open('model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)
```
- открываем файл `model.bin` в режиме чтения (`'rb'` — read binary).
- С помощью `pickle.load` загружаем из файла два объекта: векторизатор (`dv`) и модель (`model`), которые были сохранены ранее.



### Создание Flask-приложения
```
app = Flask('churn')
```
- Создаем экземпляр Flask-приложения с именем `'churn'`.



### Определение маршрута для предсказаний
```
@app.route('/predict', methods=['POST'])
def predict():
```
- создаем маршрут (`/predict`), который будет обрабатывать POST-запросы.
- Когда на этот маршрут приходит POST-запрос, вызывается функция `predict()`.



### Получение данных от клиента
```
    customer = request.get_json()
```
- извлекаем JSON-данные из тела POST-запроса. Эти данные представляют собой информацию о клиенте, для которого нужно сделать предсказание.



### Преобразование данных с помощью векторизатора
```
    X = dv.transform(customer)
```
- используем векторизатор (`dv`), чтобы преобразовать данные клиента в формат, который может обработать модель.



### Предсказание с помощью модели
```
    y_pred = model.predict(X)
```
- передаем преобразованные данные (`X`) в модель (`model`) и получаем предсказание (`y_pred`).
- `y_pred` — это массив, где каждый элемент соответствует предсказанию для одного клиента. В данном случае массив содержит только один элемент, так как обрабатываем одного клиента.



### Формирование результата
```
    if y_pred[0] == 0:
        result = 'No'
    else:
        result = 'Yes'
```
- проверяем значение предсказания:
  - Если `y_pred[0] == 0`, то клиент не уйдет (`No`).
  - Если `y_pred[0] == 1`, то клиент уйдет (`Yes`).



### Возврат результата
```
    return {
        'churn': result
    }
```
- возвращаем JSON-ответ с результатом предсказания:
  ```
  {
      "churn": "Yes"
  }
  ```



### Запуск Flask-приложения
```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
```
  - `debug=True`: Включает режим отладки (удобно для разработки).
  - `host='0.0.0.0'`: Приложение будет доступно на всех IP-адресах сервера.
  - `port=9696`: Приложение будет слушать порт 9696.

## Загрузка образа в докер хаю
###  Сборка Docker-образа
```
docker build -t predict_app .
```
### Тест запуск образа
```
docker run predict_app
```
### Выгрузка образа в Docker Hub
```
docker login
docker push [название образа]
```

## Работа с виртуальной машиной в Яндекс облака и использование докера как сервис
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
