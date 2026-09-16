"""Интерфейс «список покупок по блюду»: ввод → обработчик → вывод."""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from llm import (
    FORMAT_LIST,
    FORMAT_STEPS,
    AppError,
    generate_shopping_list,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(page_title="Список покупок по блюду", layout="centered")
st.title("Список покупок по блюду")
st.caption("Напишите, что приготовить — получите список продуктов от DeepSeek.")

dish = st.text_area(
    "Что приготовить",
    height=140,
    placeholder="Например: борщ",
)
people = st.selectbox("На сколько человек", ["1", "2", "4", "6"])
answer_format = st.selectbox("Формат", [FORMAT_LIST, FORMAT_STEPS])

if st.button("Сгенерировать список покупок", type="primary"):
    try:
        result = generate_shopping_list(dish, people, answer_format)
        st.text_area("Список покупок", value=result, height=360, disabled=True)
    except AppError as exc:
        st.error(exc.message)
    except Exception:
        st.error("Не удалось получить ответ. Попробуйте ещё раз.")
