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

from sqlalchemy import create_engine, Column, Integer, Float, String, Date, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime

engine = create_engine('sqlite:///healthcare.db', connect_args={'check_same_thread': False})
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

# 個人情報
class UserInfo(Base):
    __tablename__ = 'user_info'
    id = Column(Integer, primary_key=True)
    age = Column(Integer)
    gender = Column(String)
    height = Column(Float)  

# 体組成データ
class BodyComposition(Base):
    __tablename__ = 'body_composition'
    id = Column(Integer, primary_key=True)
    date = Column(Date, default=datetime.date.today)
    fat_percentage = Column(Float)
    muscle_mass = Column(Float)
    weight = Column(Float)
    bone_mass = Column(Float)
    water_percentage = Column(Float)

# 食事記録データ
class MealRecord(Base):
    __tablename__ = 'meal_record'
    id = Column(Integer, primary_key=True)
    date = Column(Date, default=datetime.date.today)
    meal_type = Column(String)  # 朝, 昼, 夕, 間食
    input_type = Column(String)  # text or image
    content = Column(String)  # 食事内容または画像のパス
    nutrition_data = Column(JSON)  # JSONで栄養情報を保存

# 筋トレ記録データ
class WorkoutRecord(Base):
    __tablename__ = 'workout_record'
    id = Column(Integer, primary_key=True)
    date = Column(Date, default=datetime.date.today)
    exercise_name = Column(String)
    weight_used = Column(Float)
    reps = Column(Integer)
    sets = Column(Integer)
    duration_min = Column(Float)

Base.metadata.create_all(engine)
