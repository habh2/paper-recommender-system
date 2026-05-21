FROM recommender-base

LABEL org.opencontainers.image.source=https://github.com/habh2/recommender-system

COPY preference/ preference/

CMD ["uvicorn", "preference.app:app", "--host", "0.0.0.0", "--port", "8000"]
