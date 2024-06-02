# استخدام صورة أساسية مع Python
FROM python:3.9

# تعيين دليل العمل داخل الحاوية
WORKDIR /app

# نسخ ملفات المشروع إلى دليل العمل
COPY . /app

# تثبيت التبعيات
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# تعيين المنفذ الذي سيتم استخدامه
EXPOSE 8501

# الأمر الافتراضي لتشغيل التطبيق
CMD ["streamlit", "run", "fimall.py"]
