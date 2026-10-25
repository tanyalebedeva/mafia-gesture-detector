.PHONY: install dev train tune eval predict leakage app test lint

DATA ?= datasets/mafia-cv-1/data.yaml

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

train:            ## скачать датасет и обучить
	python -m mafia_cv.train

tune:             ## подбор гиперпараметров + обучение
	python -m mafia_cv.train --tune

eval:             ## метрики на test
	python -m mafia_cv.evaluate --data $(DATA) --split test

predict:          ## make predict SRC=photo.jpg
	python -m mafia_cv.predict $(SRC) --save

leakage:          ## проверить пересечение train/test по видео
	python -m mafia_cv.leakage $(dir $(DATA))

app:
	python app.py

test:
	pytest -q

lint:
	ruff check .
