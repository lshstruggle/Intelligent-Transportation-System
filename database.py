# database.py
import pymysql
from datetime import datetime

class Database:
    def __init__(self, host, user, password, database, port=3306):
        """初始化数据库连接配置"""
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = None

    def connect(self):
        """连接到 MySQL 数据库"""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            print("成功连接到 MySQL 数据库")
        except Exception as e:
            print(f"连接 MySQL 失败: {e}")

    def disconnect(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            print("已关闭 MySQL 连接")

    def insert_vehicle_data(self, current_datetime, camera_id, track_id, vehicle_class, vehicle_count, speed, congestion_level, congestion_time, latest_congestion_level, predicted_congestion_level):
        """插入车辆数据到数据库"""
        if not self.connection:
            print("未连接到数据库，请先调用 connect() 方法")
            return

        try:
            with self.connection.cursor() as cursor:
                sql = """
                INSERT INTO vehicle_data (current_datetime, camera_id, track_id, vehicle_class, vehicle_count, speed, congestion_level, congestion_time, latest_congestion_level, predicted_congestion_level)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (current_datetime, camera_id, track_id, vehicle_class, vehicle_count, speed, congestion_level, congestion_time, latest_congestion_level, predicted_congestion_level))
            self.connection.commit()
            print(f"成功插入数据: 车辆 {track_id} 的信息")
        except Exception as e:
            print(f"插入数据失败: {e}")