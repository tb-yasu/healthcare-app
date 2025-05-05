# GNU GENERAL PUBLIC LICENSE
# Version 3, 29 June 2007
#
# Copyright (C) 2025-2030 Yasuo Tabei
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import datetime
import streamlit as st
import pandas as pd
from database import session, MealRecord, BodyComposition, WorkoutRecord, UserInfo
import requests
import base64
import re

import matplotlib.pyplot as plt
import matplotlib


import json

import openai
from openai import OpenAI

from database import MealRecord  # database.pyからインポート

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_advice(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # gpt-4.1 から gpt-4o に変更
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"エラー: {str(e)}"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_advice_with_image(prompt, image_path):
    base64_image = encode_image(image_path)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"エラー: {str(e)}"

def parse_nutrition_json(response_text):
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        json_str = response_text[start:end]
        nutrition_data = json.loads(json_str)
        return nutrition_data
    except json.JSONDecodeError as e:
        st.error(f"栄養情報の解析に失敗しました: {e}")
        return None

def parse_nutrient_value(value):
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str):
        try:
            numbers = re.findall(r"[\d\.]+", value)
            numbers = list(map(float, numbers))
            if len(numbers) == 2:
                return sum(numbers) / 2
            elif len(numbers) == 1:
                return numbers[0]
        except:
            return 0
    return 0

def get_nutrient_value(nutrition_data, keys):
    for key in keys:
        if key in nutrition_data:
            return parse_nutrient_value(nutrition_data[key])
    return 0

def calculate_bmr(gender, age, height, weight):
    if gender == "男性":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    elif gender == "女性":
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    else:  # その他の場合は中間値をとる
        bmr_male = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        bmr_female = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        bmr = (bmr_male + bmr_female) / 2
    return bmr

def calculate_bmi(height, weight):
    """
    BMI（ボディマス指数）を計算する関数
    height: 身長（cm）
    weight: 体重（kg）
    """
    # 身長をcmからmに変換
    height_m = height / 100
    # BMIを計算（体重÷身長の2乗）
    bmi = weight / (height_m * height_m)
    
    # BMI値に基づいて評価を返す
    if bmi < 18.5:
        category = "低体重（痩せ型）"
    elif bmi < 25:
        category = "普通体重"
    elif bmi < 30:
        category = "肥満（1度）"
    elif bmi < 35:
        category = "肥満（2度）"
    elif bmi < 40:
        category = "肥満（3度）"
    else:
        category = "肥満（4度）"
    
    return bmi, category

# GPT-4o API連携用関数の実装
#def generate_advice(prompt):
#    try:
#        response = client.chat.completions.create(
#            #model="gpt-4o",
#            model="gpt-4.1",
#            messages=[{"role": "user", "content": prompt}],
#            max_tokens=800,
#            temperature=0.5,
#        )
#        return response.choices[0].message.content.strip()
#    except Exception as e:
#        return f"エラー: {str(e)}"

# Gemma3 LLM API連携用関数の実装
#def generate_advice(prompt):
#    api_url = "http://localhost:11434/api/generate"
#    headers = {"Content-Type": "application/json"}
#    data = {"model": "gemma3:27b", "prompt": prompt, "stream": False}
#
#    response = requests.post(api_url, json=data, headers=headers)
#
#
#    if response.status_code == 200:
#        result = response.json()
#        return result.get("response", "[No response from Gemma3]")
#    else:
#        return f"エラー: ステータスコード {response.status_code}、メッセージ: {response.text}"

st.title("ヘルスケアアプリケーション")

st.sidebar.header("ユーザー情報")
selected_date = st.sidebar.date_input("記録日を選択", datetime.date.today(), key="sidebar_date")
age = st.sidebar.number_input("年齢", min_value=1, max_value=120, value=30, key="sidebar_age")
gender = st.sidebar.selectbox("性別", ["男性", "女性", "その他"], key="sidebar_gender")
height = st.sidebar.number_input("身長 (cm)", min_value=50.0, max_value=250.0, step=0.1, key="sidebar_height")

if st.sidebar.button("ユーザー情報を保存", key="save_user_info"):
    user_info = UserInfo(age=age, gender=gender, height=height)
    session.add(user_info)
    session.commit()
    st.sidebar.success("ユーザー情報を保存しました。")


# タブを使って入力カテゴリを整理
tabs = st.tabs(["食事記録", "体組成記録", "筋トレ記録", "データ表示・可視化", "評価とアドバイス", "ヘルスケア相談", "データ一元管理"])

# 画像保存ディレクトリ
UPLOAD_DIR = 'uploaded_images'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 食事記録フォーム
with tabs[0]:
    st.header("食事記録フォーム")

    if 'meal_list' not in st.session_state:
        st.session_state.meal_list = []

    with st.form("meal_form", clear_on_submit=True):
        meal_type = st.selectbox("食事タイプ", ["朝", "昼", "夕", "間食"], key="meal_type")
        content = st.text_area("食事内容を入力してください", key="meal_content")
        uploaded_file = st.file_uploader("画像をアップロード（任意）", type=["jpg", "jpeg", "png"], key="meal_image")
        add_meal = st.form_submit_button("食事を追加")

        if add_meal and (content or uploaded_file):
            file_path = None
            if uploaded_file:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_extension = uploaded_file.name.split('.')[-1]
                filename = f"{meal_type}_{timestamp}.{file_extension}"
                file_path = os.path.join(UPLOAD_DIR, filename)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
                # 画像がある場合、画像分析を行う
#                nutrition_prompt = """
#                この食事のカロリー、タンパク質(g)、脂質(g)、炭水化物(g)を数値のみで返してください。
#                説明文や数値の範囲ではなく、それぞれ単一の具体的な数値で回答してください。
#                以下の形式で返してください：
#                カロリー,タンパク質,脂質,炭水化物
#                食事内容：
#                {}
#                """.format(content if content else "画像に写っている料理")

                nutrition_prompt = "この画像の食事の食事名、カロリー、タンパク質、炭水化物、脂質に加えて、以下の栄養素も大まかに数値(文字列はやめてください)で推定してください：\n1. ビタミン類（主要なビタミンA、B群、C、D、E、Kなど）\n2. ミネラル類（カルシウム、鉄、亜鉛、マグネシウムなど）\n3. 食物繊維（水溶性・不溶性）\n4. 水分量と電解質（ナトリウム、カリウムなど）\n5. フィトケミカル/抗酸化物質\n 最後に栄養の一覧をjsonで表示してください。"
                nutrition_info = generate_advice_with_image(nutrition_prompt, file_path)
                nutrition_data = parse_nutrition_json(nutrition_info)
                if nutrition_data:
                    pass
#                    st.write("### 栄養情報")
#                    st.json(nutrition_data)
                else:
                    st.error("栄養情報を取得できませんでした。")

                print(nutrition_data)
                meal_record = MealRecord(
                    date=selected_date,
                    meal_type=meal_type,
                    input_type="画像" if file_path else "テキスト",
                    content=content or file_path,
                    nutrition_data=nutrition_data  # JSONで栄養情報を保存
                )
                session.add(meal_record)
                session.commit()
            else:
#                nutrition_prompt = """
#                この食事のカロリー、タンパク質(g)、脂質(g)、炭水化物(g)を数値のみで返してください。
#                説明文や数値の範囲ではなく、それぞれ単一の具体的な数値で回答してください。
#                以下の形式で返してください：
#                カロリー,タンパク質,脂質,炭水化物
#
#                食事内容：
#                {}
#                """.format(content if content else "画像に写っている料理")

                # テキストのみの場合
                nutrition_prompt = f"この食事の食事名、カロリー、タンパク質、炭水化物、脂質に加えて、以下の栄養素も大まかに数値(文字列はやめてください)で推定してください：\n1. ビタミン類（主要なビタミンA、B群、C、D、E、Kなど）\n2. ミネラル類（カルシウム、鉄、亜鉛、マグネシウムなど）\n3. 食物繊維（水溶性・不溶性）\n4. 水分量と電解質（ナトリウム、カリウムなど）\n5. フィトケミカル/抗酸化物質\n 最後に栄養の一覧をjsonで表示してください。 食事内容: {content}"
                nutrition_info = generate_advice(nutrition_prompt)  # 画像なしの関数を使用

                nutrition_data = parse_nutrition_json(nutrition_info)

                if nutrition_data:
                    pass    
#                    st.write("### 栄養情報")
#                    st.json(nutrition_data)
                else:
                    st.error("栄養情報を取得できませんでした。")

                meal_record = MealRecord(
                    date=selected_date,
                    meal_type=meal_type,
                    input_type="画像" if file_path else "テキスト",
                    content=content or file_path,
                    nutrition_data=nutrition_data  # JSONで栄養情報を保存
                )
                session.add(meal_record)
                session.commit()


#            try:
#                cal, prot, fat, carb = map(float, nutrition_info.split(","))
#            except ValueError as e:
#                st.error(f"栄養情報の解析に失敗しました: {nutrition_info}")
#                cal = prot = fat = carb = 0  # 解析失敗時のデフォルト値を設定

            st.session_state.meal_list.append({"meal_type": meal_type, "content": content, "image": file_path, "nutrition_info": nutrition_info})

    # 削除用のセッション状態を初期化
    if 'to_delete' not in st.session_state:
        st.session_state.to_delete = None

    st.write("### 本日の食事内容と栄養情報")
    total_nutrition = ""
    for idx, meal in enumerate(st.session_state.meal_list):
        st.write(f"{idx+1}. {meal['meal_type']}: {meal['content']}")
        if meal['image']:
            st.image(meal['image'], width=200)
        st.write(f"栄養情報: {meal['nutrition_info']}")
        
        # 削除ボタンを追加
        if st.button(f"削除 {idx+1}", key=f"delete_{idx}"):
            st.session_state.to_delete = idx

        total_nutrition += meal['nutrition_info'] + "\n"

    # 削除処理をループの外で実行
    if st.session_state.to_delete is not None:
        if st.session_state.to_delete < len(st.session_state.meal_list):
            del st.session_state.meal_list[st.session_state.to_delete]
        st.session_state.to_delete = None
        st.rerun()

    if st.button("1日分の食事記録を保存"):
        for meal in st.session_state.meal_list:
            meal_record = MealRecord(
                date=selected_date,
                meal_type=meal["meal_type"],
                input_type="画像" if meal["image"] else "テキスト",
                content=meal["content"] or meal["image"]
            )
            session.add(meal_record)
        session.commit()
        st.session_state.meal_list.clear()
        st.success("食事記録を保存しました。")


# 体組成記録フォーム
with tabs[1]:
    st.header("体組成記録フォーム")
    with st.form("body_composition_form", clear_on_submit=True):
        weight = st.number_input("体重 (kg)", min_value=10.0, max_value=300.0, step=0.1, key="weight")
        fat_percentage = st.number_input("体脂肪率 (%)", min_value=1.0, max_value=60.0, step=0.1, key="fat_percentage")
        muscle_mass = st.number_input("筋肉量 (kg)", min_value=1.0, max_value=150.0, step=0.1, key="muscle_mass")
        bone_mass = st.number_input("骨量 (kg)", min_value=0.5, max_value=10.0, step=0.1, key="bone_mass")
        water_percentage = st.number_input("体水分率 (%)", min_value=10.0, max_value=90.0, step=0.1, key="water_percentage")

        submitted = st.form_submit_button("体組成記録を保存")

        if submitted:
            body_record = BodyComposition(
                date=selected_date,
                weight=weight,
                fat_percentage=fat_percentage,
                muscle_mass=muscle_mass,
                bone_mass=bone_mass,
                water_percentage=water_percentage
            )
            session.add(body_record)
            session.commit()
            st.success("体組成記録を保存しました。")

# 筋トレ記録フォーム
with tabs[2]:
    st.header("筋トレ記録フォーム")
    if 'workout_list' not in st.session_state:
        st.session_state.workout_list = []

    with st.form("workout_form", clear_on_submit=True):
        exercise_name = st.text_input("種目名", key="exercise_name")
        weight_used = st.number_input("使用重量 (kg)", 0.0, 500.0, step=0.5, key="weight_used")
        reps = st.number_input("レップ数", 1, 100, key="reps")
        sets = st.number_input("セット数", 1, 20, key="sets")
        duration_min = st.number_input("運動時間 (分)", 1.0, 180.0, step=0.5, key="duration_min")

        add_exercise = st.form_submit_button("種目を追加")

        if add_exercise:
            st.session_state.workout_list.append({
                "exercise_name": exercise_name,
                "weight_used": weight_used,
                "reps": reps,
                "sets": sets,
                "duration_min": duration_min
            })

    if st.session_state.workout_list:
        st.write("### 本日のトレーニング内容")
        for idx, workout in enumerate(st.session_state.workout_list):
            st.write(f"{idx+1}. {workout['exercise_name']} - {workout['weight_used']}kg, {workout['reps']}レップ, {workout['sets']}セット, {workout['duration_min']}分")

        if st.button("1日分の筋トレ記録を保存"):
            for workout in st.session_state.workout_list:
                workout_record = WorkoutRecord(
                    date=selected_date,
                    exercise_name=workout["exercise_name"],
                    weight_used=workout["weight_used"],
                    reps=workout["reps"],
                    sets=workout["sets"],
                    duration_min=workout["duration_min"]
                )
                session.add(workout_record)
            session.commit()
            st.session_state.workout_list.clear()
            st.success("筋トレ記録を保存しました.")

with tabs[3]:  # 「データ表示・可視化」タブの位置に合わせて調整
    st.header("データ表示・可視化")

    vis_category = st.selectbox("表示するデータの種類を選択してください", ["食事", "体組成", "筋トレ"])

    start_date = st.date_input("開始日", datetime.date.today() - datetime.timedelta(days=30), key="start_date_vis")
    end_date = st.date_input("終了日", datetime.date.today(), key="end_date_vis")

    if vis_category == "食事":
        meals = session.query(MealRecord).filter(MealRecord.date.between(start_date, end_date)).all()
        if meals:
            dates, calories, proteins, carbs, fats = [], [], [], [], []

            for meal in meals:
                nutrition = meal.nutrition_data
                if nutrition:
                    cal = parse_nutrient_value(nutrition.get("カロリー") or nutrition.get("calories"))
                    prot = parse_nutrient_value(nutrition.get("タンパク質") or nutrition.get("protein"))
                    fat = parse_nutrient_value(nutrition.get("脂質") or nutrition.get("fat"))
                    carb = parse_nutrient_value(nutrition.get("炭水化物") or nutrition.get("carbohydrates"))
                else:
                    continue

                dates.append(meal.date)
                calories.append(cal)
                proteins.append(prot)
                carbs.append(carb)
                fats.append(fat)

            if dates:
                df_meal = pd.DataFrame({
                    "日付": dates,
                    "カロリー": calories,
                    "タンパク質": proteins,
                    "炭水化物": carbs,
                    "脂質": fats
                }).groupby("日付").sum()

                # インデックスを文字列の日付に変更（時刻を削除）
#                df_meal.index = df_meal.index.strftime('%Y-%m-%d')
                df_meal.index = pd.to_datetime(df_meal.index).strftime('%Y-%m-%d')

                st.bar_chart(df_meal)
            else:
                st.warning("表示できる栄養情報がありません。")
        else:
            st.info("指定された期間の食事記録がありません。")

    elif vis_category == "体組成":
        body_data = session.query(BodyComposition).filter(
            BodyComposition.date.between(start_date, end_date)).order_by(BodyComposition.date).all()
        if body_data:
            dates = [data.date for data in body_data]
            weights = [data.weight for data in body_data]
            fat_percentages = [data.fat_percentage for data in body_data]
            muscle_masses = [data.muscle_mass for data in body_data]

            # 日本語フォントを指定（Macの場合の例）
            matplotlib.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'  # Macの例
            # matplotlib.rcParams['font.family'] = 'Meiryo'  # Windowsの例
            # matplotlib.rcParams['font.family'] = 'IPAPGothic'  # Linuxの例

            fig, ax = plt.subplots(3, 1, figsize=(10, 12))
            ax[0].plot(dates, weights, marker="o")
            ax[0].set_title("体重推移")
            ax[0].set_ylabel("体重 (kg)")

            ax[1].plot(dates, fat_percentages, marker="o", color="orange")
            ax[1].set_title("体脂肪率推移")
            ax[1].set_ylabel("体脂肪率 (%)")

            ax[2].plot(dates, muscle_masses, marker="o", color="green")
            ax[2].set_title("筋肉量推移")
            ax[2].set_ylabel("筋肉量 (kg)")

            plt.tight_layout()
            st.pyplot(fig)

    elif vis_category == "筋トレ":
        workouts = session.query(WorkoutRecord).filter(WorkoutRecord.date.between(start_date, end_date)).all()
        if workouts:
            workout_data = {}
            for workout in workouts:
                volume = workout.weight_used * workout.reps * workout.sets
                if workout.exercise_name not in workout_data:
                    workout_data[workout.exercise_name] = volume
                else:
                    workout_data[workout.exercise_name] += volume

            workout_df = pd.DataFrame(workout_data.items(), columns=["種目", "総重量"])
            workout_df.set_index("種目", inplace=True)

            st.bar_chart(workout_df)

            total_duration = sum(workout.duration_min for workout in workouts)
            total_sessions = len(set(workout.date for workout in workouts))

            st.metric("期間内の総運動時間 (分)", total_duration)
            st.metric("期間内のトレーニング日数", total_sessions)

# 評価とアドヴァイス
with tabs[4]:

    st.header("評価とアドヴァイス")

    evaluation_criteria = st.selectbox("評価基準を選択してください", ["糖質制限", "PFCバランス", "減量", "ビタミン・ミネラルバランス", "抗酸化物質"])

    meal_eval_btn = st.button("食事の評価を実行")
    body_eval_btn = st.button("体組成の評価を実行")
    workout_eval_btn = st.button("筋トレの評価を実行")
    total_eval_btn = st.button("全体の評価を実行")

    user_info = session.query(UserInfo).order_by(UserInfo.id.desc()).first()
    body_data = session.query(BodyComposition).filter(BodyComposition.date <= selected_date)\
                        .order_by(BodyComposition.date.desc()).first()


    # 変更後
    if user_info:
        st.write(f"ユーザー情報: {user_info.gender}, {user_info.age}, {user_info.height}")
    else:
        st.warning("ユーザー情報が登録されていません。サイドバーから登録してください。")

#    bmr = 0
#    if user_info and body_data:
#        bmr = calculate_bmr(user_info.gender, user_info.age, user_info.height, body_data.weight)

#    st.write(f"基礎代謝量(BMR)は約{bmr:.1f} kcalです。")

    if meal_eval_btn:
        today_meals = session.query(MealRecord).filter(MealRecord.date == selected_date).all()

        st.write("デバッグ: 今日の食事記録")
        for meal in today_meals:
            st.write(f"日付: {meal.date}, 食事タイプ: {meal.meal_type}, 内容: {meal.content}, 栄養データ: {meal.nutrition_data}")


        bmr = 0
        if user_info and body_data:
            bmr = calculate_bmr(user_info.gender, user_info.age, user_info.height, body_data.weight)

        # デバッグ: 今日の食事記録の内容を表示
        print(today_meals)
        
        if today_meals:
            total_calories = total_proteins = total_fats = total_carbs = 0

            for meal in today_meals:
                nutrition = meal.nutrition_data
                if nutrition:
                    cal = get_nutrient_value(nutrition, ["calories", "カロリー"])
                    prot = get_nutrient_value(nutrition, ["protein", "タンパク質"])
                    fat = get_nutrient_value(nutrition, ["fat", "脂質"])
                    carb = get_nutrient_value(nutrition, ["carbohydrates", "炭水化物"])
                else:
                    continue

                total_calories += cal
                total_proteins += prot
                total_fats += fat
                total_carbs += carb

            nutrition_summary = (
                f"本日の総栄養摂取量:\n"
                f"- カロリー: {total_calories:.1f} kcal\n"
                f"- タンパク質: {total_proteins:.1f} g\n"
                f"- 脂質: {total_fats:.1f} g\n"
                f"- 炭水化物: {total_carbs:.1f} g\n"

            )

            meal_prompt = f"""
            本日の食事の総栄養摂取量は以下の通りです：

            {nutrition_summary}

            基礎代謝量(BMR)は約{bmr:.1f} kcalです。これらを考慮し、
            {evaluation_criteria}の観点から評価し、改善点やアドヴァイスを具体的に提案してください。
            """

            meal_advice = generate_advice(meal_prompt)

            st.subheader("食事の評価とアドヴァイス")
            st.write(nutrition_summary)
            st.write(meal_advice)
        else:
            st.info("本日の食事記録がありません。")



#        for meal in today_meals:
#            if meal.input_type == "画像" and os.path.isfile(meal.content):
#                meal_prompt = f"{meal.meal_type}の食事内容を{evaluation_criteria}の観点から評価してください。また、カロリー、PFC、ビタミン類、ミネラル類、食物繊維、水分・電解質、抗酸化物質のバランスについても分析してください。"
#                meal_advice = generate_advice_with_image(meal_prompt, meal.content)
#                st.subheader(f"{meal.meal_type}（画像あり）評価")
#                st.image(meal.content, width=200)
#                st.write(meal_advice)
#            else:
#                meal_prompt = f"{meal.meal_type}の食事内容: {meal.content}\n\n上記を{evaluation_criteria}の観点から評価してください。また、カロリー、PFC、ビタミン類、ミネラル類、食物繊維、水分・電解質、抗酸化物質のバランスについても分析してください。"
#                meal_advice = generate_advice(meal_prompt)
#                st.subheader(f"{meal.meal_type}評価")
#                st.write(meal_advice)

    if body_eval_btn:
        body_changes = session.query(BodyComposition).filter(BodyComposition.date == selected_date).all()
        
        body_summary = "\n".join([f"身長:{user_info.height}cm, 体重:{b.weight}kg, 体脂肪率:{b.fat_percentage}%, 筋肉量:{b.muscle_mass}kg, 骨量:{b.bone_mass}kg, 体水分率:{b.water_percentage}%" for b in body_changes])
        if body_summary:
            body_prompt = f"本日の体組成:\n{body_summary}\n\n以上を{evaluation_criteria}の観点から評価してください。"
            body_advice = generate_advice(body_prompt)
            st.subheader("体組成評価")
            st.write(body_advice)
        else:
            st.info("本日の体組成記録がありません。")

    if workout_eval_btn:
        today_workouts = session.query(WorkoutRecord).filter(WorkoutRecord.date == selected_date).all()
        workout_summary = "\n".join([f"種目: {w.exercise_name}, 重量: {w.weight_used}kg, レップ: {w.reps}, セット: {w.sets}, 時間: {w.duration_min}分" for w in today_workouts])
        if workout_summary:
            workout_prompt = f"本日の筋トレ内容:\n{workout_summary}\n\n以上を{evaluation_criteria}の観点から評価してください。"
            workout_advice = generate_advice(workout_prompt)
            st.subheader("筋トレ評価")
            st.write(workout_advice)
        else:
            st.info("本日の筋トレ記録がありません。")

    if total_eval_btn:
        prompt = f"""以下の情報を総合して{evaluation_criteria}の観点から評価してください。また、カロリー、PFC、ビタミン類、ミネラル類、食物繊維、水分・電解質、抗酸化物質のバランスについても分析してください。食事面だけでなく、体組成や筋トレ記録も考慮に入れて分析してください。


        以下の形式で回答してください：

        ## 総合評価
        [全体的な評価]

        ## 栄養素分析
        ### マクロ栄養素
        - カロリー：[評価]
        - タンパク質：[評価]
        - 脂質：[評価]
        - 炭水化物：[評価]

        ### ビタミン
        [主要なビタミンの評価]

        ### ミネラル
        [主要なミネラルの評価]

        ### 食物繊維
        [評価]

        ### 水分・電解質
        [評価]

        ### フィトケミカル/抗酸化物質
        [評価]

        ## 改善提案
        [具体的な改善点]

        ## 体組成分析
        ### 体重、体脂肪率、筋肉量
        [評価]


        ## 筋トレ分析
        ### トレーニング内容
        [評価]

        ## 総合評価
        [総合評価]
        """

        today_meals = session.query(MealRecord).filter(MealRecord.date == selected_date).all()
        today_workouts = session.query(WorkoutRecord).filter(WorkoutRecord.date == selected_date).all()
        body_changes = session.query(BodyComposition).filter(BodyComposition.date == selected_date).all()

        meal_summary_lines = []
        for meal in today_meals:
            if meal.nutrition_data:
                nutrition = meal.nutrition_data
                cal = nutrition.get("カロリー") or nutrition.get("calories", "不明")
                prot = nutrition.get("タンパク質") or nutrition.get("protein", "不明")
                fat = nutrition.get("脂質") or nutrition.get("fat", "不明")
                carb = nutrition.get("炭水化物") or nutrition.get("carbohydrates", "不明")
                meal_summary_lines.append(
                    f"{meal.meal_type}: カロリー:{cal} kcal, タンパク質:{prot} g, 脂質:{fat} g, 炭水化物:{carb} g"
                )
            else:
                meal_summary_lines.append(f"{meal.meal_type}: 栄養情報がありません")

        workout_summary = "\n".join(
            [f"種目: {w.exercise_name}, 重量: {w.weight_used}kg, レップ: {w.reps}, セット: {w.sets}, 時間: {w.duration_min}分" for w in today_workouts]
        ) if today_workouts else "記録なし"

        body_summary = "\n".join(
            [f"身長:{user_info.height}cm, 体重:{b.weight}kg, 体脂肪率:{b.fat_percentage}%, 筋肉量:{b.muscle_mass}kg, 骨量:{b.bone_mass}kg, 体水分率:{b.water_percentage}%" for b in body_changes]
        ) if body_changes else "記録なし"

        meal_summary = "\n".join(meal_summary_lines)

        prompt += f"【食事】\n{meal_summary}\n\n【体組成】\n{body_summary}\n\n【筋トレ】\n{workout_summary}\n"

        total_advice = generate_advice(prompt)
        st.subheader("総合評価")
        st.write(total_advice)


# LLM相談タブの追加
with tabs[5]:
    st.header("最近の状況を踏まえたLLM相談")

    one_month_ago = datetime.date.today() - datetime.timedelta(days=30)

    meals = session.query(MealRecord).filter(MealRecord.date >= one_month_ago).all()
    body_data = session.query(BodyComposition).filter(BodyComposition.date >= one_month_ago).all()
    workouts = session.query(WorkoutRecord).filter(WorkoutRecord.date >= one_month_ago).all()

    meal_summary = "\n".join([f"{meal.date}: {meal.meal_type} - {meal.content}" for meal in meals]) if meals else "記録なし"
    body_summary = "\n".join([f"{data.date}: 体重:{data.weight}kg, 体脂肪率:{data.fat_percentage}%, 筋肉量:{data.muscle_mass}kg" for data in body_data]) if body_data else "記録なし"
    workout_summary = "\n".join([f"{workout.date}: 種目:{workout.exercise_name}, 重量:{workout.weight_used}kg, レップ:{workout.reps}, セット:{workout.sets}" for workout in workouts]) if workouts else "記録なし"

    user_query = st.text_area("相談したい内容を入力してください")
    if st.button("LLMに相談"):
        consultation_prompt = f"以下は最近1ヶ月間の食事、体組成、筋トレのデータです。これらを踏まえてユーザーの質問に答えてください。\n\n【食事】\n{meal_summary}\n\n【体組成】\n{body_summary}\n\n【筋トレ】\n{workout_summary}\n\nユーザーの質問：{user_query}"

        advice = generate_advice(consultation_prompt)
        st.write("### LLMからの回答")
        st.write(advice)


with tabs[6]:  # タブの番号は適宜調整してください
    st.header("データ一元管理")

    data_category = st.selectbox("表示したいデータを選択", ["食事記録", "体組成記録", "筋トレ記録"], key="data_category_select")

    start_date = st.date_input("開始日", datetime.date.today() - datetime.timedelta(days=30), key="start_date_manage")
    end_date = st.date_input("終了日", datetime.date.today(), key="end_date_manage")

    if data_category == "食事記録":
        records = session.query(MealRecord).filter(
            MealRecord.date.between(start_date, end_date)).order_by(MealRecord.date.desc()).all()
        data = [{
            "id": r.id, "日付": r.date, "タイプ": r.meal_type, "内容": r.content, "種類": r.input_type
        } for r in records]

    elif data_category == "体組成記録":
        records = session.query(BodyComposition).filter(
            BodyComposition.date.between(start_date, end_date)).order_by(BodyComposition.date.desc()).all()
        data = [{
            "id": r.id, "日付": r.date, "体重": r.weight,
            "体脂肪率": r.fat_percentage, "筋肉量": r.muscle_mass,
            "骨量": r.bone_mass, "体水分率": r.water_percentage
        } for r in records]

    elif data_category == "筋トレ記録":
        records = session.query(WorkoutRecord).filter(
            WorkoutRecord.date.between(start_date, end_date)).order_by(WorkoutRecord.date.desc()).all()
        data = [{
            "id": r.id, "日付": r.date, "種目": r.exercise_name, "重量": r.weight_used,
            "レップ数": r.reps, "セット数": r.sets, "時間": r.duration_min
        } for r in records]

    if data:
        df = pd.DataFrame(data).drop(columns=['id'])
        df["削除する"] = [False] * len(df)  # チェックボックス用のカラムを追加

        # DataFrameを編集可能な形式で表示（チェックボックスで行を選択可能に）
        edited_df = st.data_editor(df, key="editable_table", hide_index=False)

        if st.button("選択した記録を一括削除"):
            ids_to_delete = [record["id"] for idx, record in enumerate(data) if edited_df.loc[idx, "削除する"]]

            if ids_to_delete:
                for record_id in ids_to_delete:
                    if data_category == "食事記録":
                        session.query(MealRecord).filter(MealRecord.id == record_id).delete()
                    elif data_category == "体組成記録":
                        session.query(BodyComposition).filter(BodyComposition.id == record_id).delete()
                    elif data_category == "筋トレ記録":
                        session.query(WorkoutRecord).filter(WorkoutRecord.id == record_id).delete()
                session.commit()
                st.success(f"{len(ids_to_delete)}件の記録を削除しました。")
                st.rerun()
            else:
                st.warning("削除する記録が選択されていません。")
    else:
        st.info("指定した期間の記録はありません。")
