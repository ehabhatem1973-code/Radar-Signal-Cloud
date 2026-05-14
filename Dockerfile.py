# استخدام نسخة بايثون خفيفة جداً
FROM python:3.9-slim

# تحديد مكان العمل داخل الـ Container
WORKDIR /app

# تسطيب أدوات النظام الضرورية
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف الطلبات أولاً لتسريع الـ Build
COPY requirements.txt .

# تسطيب المكتبات
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع (الأكواد والموديل)
COPY . .

# فتح البورت اللي الـ Streamlit هيشتغل عليه
EXPOSE 8501

# أمر التشغيل
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]