import pandas as pd
import requests
import uuid
import json
import ssl
import os
from dotenv import load_dotenv
import urllib3
import time

# Загрузка переменных окружения
load_dotenv(dotenv_path="key.env")

# Настройка API (выберите один из вариантов)
USE_GIGACHAT = True  # True - использовать GigaChat, False - использовать OpenAI

if USE_GIGACHAT:
    API_KEY = os.getenv("GIGACHAT_API_KEY")
    if not API_KEY:
        raise ValueError("❌ ОШИБКА: Не найден GIGACHAT_API_KEY в файле .env")
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("❌ ОШИБКА: Не найден OPENAI_API_KEY в файле .env")

INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.json"

# Отключение предупреждений SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def get_gigachat_token():
    """Получает токен доступа к GigaChat API"""
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    payload = "scope=GIGACHAT_API_PERS"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {API_KEY}"
    }
    response = requests.post(url, data=payload, headers=headers, verify=False)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print(f"Ошибка получения токена: {response.text}")
        return None


def extract_product_info_gigachat(text):
    """Извлекает информацию о товаре через GigaChat API"""
    token = get_gigachat_token()
    if not token:
        return None

    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "RqUID": str(uuid.uuid4())
    }
    
    prompt = f"""
    Извлеки информацию о товаре из следующего описания на русском языке.
    Ответь ТОЛЬКО валидным JSON объектом без пояснений и дополнительного текста.
    
    Формат ответа:
    {{
        "product_name": "полное название товара",
        "brand": "бренд/производитель",
        "category": "категория товара",
        "price": числовое значение цены (без валюты и пробелов),
        "currency": "валюта (RUB, USD, EUR и т.д.)",
        "key_features": ["характеристика 1", "характеристика 2"]
    }}
    
    Если какая-то информация отсутствует, укажи null для этого поля.
    
    Описание товара: "{text}"
    """

    body = {
        "model": "GigaChat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }

    response = requests.post(url, json=body, headers=headers, verify=False)
    
    if response.status_code == 200:
        result = response.json()['choices'][0]['message']['content']
        try:
            # Очистка от markdown-разметки
            if "```" in result:
                result = result.split("```")[1]
                if "json" in result.lower():
                    result = result.lower().replace("json", "", 1)
            return json.loads(result.strip())
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}")
            print(f"Полученный ответ: {result}")
            return {
                "product_name": text,
                "brand": None,
                "category": None,
                "price": None,
                "currency": None,
                "key_features": [],
                "parse_error": result
            }
    else:
        print(f"Ошибка API: {response.status_code} - {response.text}")
        return None


def extract_product_info_openai(text):
    """Извлекает информацию о товаре через OpenAI API"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    prompt = f"""
    Extract product information from the following description in Russian.
    Respond ONLY with a valid JSON object, no explanations or additional text.
    
    Response format:
    {{
        "product_name": "full product name",
        "brand": "brand/manufacturer",
        "category": "product category",
        "price": numeric price value (without currency and spaces),
        "currency": "currency (RUB, USD, EUR, etc.)",
        "key_features": ["feature 1", "feature 2"]
    }}
    
    If some information is missing, use null for that field.
    
    Product description: "{text}"
    """

    body = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a product information extractor. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    response = requests.post(url, json=body, headers=headers)
    
    if response.status_code == 200:
        result = response.json()['choices'][0]['message']['content']
        try:
            return json.loads(result.strip())
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}")
            return {
                "product_name": text,
                "brand": None,
                "category": None,
                "price": None,
                "currency": None,
                "key_features": [],
                "parse_error": result
            }
    else:
        print(f"Ошибка API: {response.status_code} - {response.text}")
        return None


def main():
    print("=" * 60)
    print("📦 PIPELINE: Извлечение характеристик товаров")
    print("=" * 60)
    
    # Чтение входных данных
    print(f"\n1️⃣  Чтение входных данных из {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        print(f"   ✅ Загружено {len(df)} записей")
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {INPUT_FILE} не найден!")
        return
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return
    
    # Проверка наличия нужной колонки
    if 'description' not in df.columns:
        print("❌ Ошибка: В CSV файле отсутствует колонка 'description'")
        return
    
    results = []
    total = len(df)
    api_function = extract_product_info_gigachat if USE_GIGACHAT else extract_product_info_openai
    api_name = "GigaChat" if USE_GIGACHAT else "OpenAI"
    
    print(f"\n2️⃣  Отправка данных в LLM ({api_name}) для извлечения характеристик...")
    print("-" * 60)
    
    for index, row in df.iterrows():
        description = row['description']
        print(f"\n📝 Обработка товара #{index + 1}/{total}...")
        
        # Извлечение информации
        extracted_info = api_function(description)
        
        if extracted_info:
            result_entry = {
                "id": index + 1,
                "original_description": description,
                "extracted_data": extracted_info
            }
            results.append(result_entry)
            
            # Вывод краткой информации
            brand = extracted_info.get("brand", "N/A")
            category = extracted_info.get("category", "N/A")
            price = extracted_info.get("price", "N/A")
            print(f"   ✅ Извлечено: {brand} | {category} | {price}")
        else:
            print("   ⚠️  Пропуск из-за ошибки API")
            results.append({
                "id": index + 1,
                "original_description": description,
                "extracted_data": None,
                "error": "API error"
            })
        
        # Пауза между запросами (чтобы не превысить лимиты)
        time.sleep(0.5)
    
    # Сохранение результатов
    print("Сохранение результатов...")
    
    output_data = {
        "metadata": {
            "total_items": len(results),
            "processed_items": sum(1 for r in results if r.get("extracted_data") is not None),
            "api_used": api_name,
            "input_file": INPUT_FILE,
            "output_file": OUTPUT_FILE
        },
        "results": results
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Результаты сохранены в {OUTPUT_FILE}")
   
    print(f"\n ИТОГИ:")
    print(f"   • Всего обработано: {len(results)}")
    print(f"   • Успешно извлечено: {output_data['metadata']['processed_items']}")
    print(f"   • Использован API: {api_name}")
    print(f"\n Готово!")


if __name__ == "__main__":
    main()