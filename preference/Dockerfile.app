FROM recommender-base

COPY preference/ ./preference/

CMD ["uvicorn", "preference.app:app", "--host", "0.0.0.0", "--port", "8000"]
