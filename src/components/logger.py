import logging
import os

# إنشاء مجلد للـ Logs إذا لم يكن موجودًا
os.makedirs("logs", exist_ok=True)

# إنشاء Logger
logger = logging.getLogger("ML_Project")
logger.setLevel(logging.INFO)

# منع تكرار الرسائل
logger.propagate = False

# لو الـ Handlers موجودة بالفعل، متضيفهاش مرة تانية
if not logger.handlers:

    # حفظ اللوج في ملف
    file_handler = logging.FileHandler(
        "logs/project.log",
        mode="a",
        encoding="utf-8"
    )

    # عرض اللوج في الـ Terminal
    console_handler = logging.StreamHandler()

    # شكل الرسالة
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s | Line:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)